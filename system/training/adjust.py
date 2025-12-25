from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from system.hardware import get_system_info
from system.training.schema import TrainingConfig


_VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_relpath(root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


@dataclass
class DatasetStats:
    image_count: int
    caption_count: int
    caption_coverage: float
    sample_count: int
    min_side: Optional[int]
    max_side: Optional[int]
    avg_min_side: Optional[float]
    avg_aspect: Optional[float]
    mostly_square: Optional[bool]

    # Quick image content stats (computed on a random sample)
    luma_mean_avg: Optional[float]
    luma_mean_std: Optional[float]
    luma_p10: Optional[float]
    luma_p50: Optional[float]
    luma_p90: Optional[float]
    luma_contrast_avg: Optional[float]
    entropy_avg: Optional[float]
    entropy_std: Optional[float]


def _percentile(sorted_vals: List[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    q = _clamp(float(q), 0.0, 1.0)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = idx - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _shannon_entropy_from_hist(hist: List[int]) -> float:
    total = float(sum(hist) or 0)
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in hist:
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return ent


def _analyze_sample_images(sample_paths: List[Path], downsample_max: int = 256) -> Dict[str, Optional[float]]:
    luma_means: List[float] = []
    luma_stds: List[float] = []
    entropies: List[float] = []

    for p in sample_paths:
        try:
            with Image.open(p) as im:
                # Deterministic, cheap-ish stats: resize to a fixed small grid.
                # (Requested: avoid relying on ImageStat quirks.)
                gray = im.convert("L")
                if downsample_max:
                    gray = gray.resize((100, 100), resample=Image.Resampling.LANCZOS)

                px = gray.tobytes()
                n = len(px)
                if n <= 0:
                    continue
                s = float(sum(px))
                ss = float(sum(v * v for v in px))
                mean = s / n
                var = max(0.0, (ss / n) - (mean * mean))
                std = math.sqrt(var)
                luma_means.append(mean)
                luma_stds.append(std)
                entropies.append(_shannon_entropy_from_hist(gray.histogram()))
        except Exception:
            continue

    if not luma_means:
        return {
            "luma_mean_avg": None,
            "luma_mean_std": None,
            "luma_p10": None,
            "luma_p50": None,
            "luma_p90": None,
            "luma_contrast_avg": None,
            "entropy_avg": None,
            "entropy_std": None,
        }

    mean_avg = sum(luma_means) / len(luma_means)
    mean_std = math.sqrt(sum((x - mean_avg) ** 2 for x in luma_means) / max(1, (len(luma_means) - 1)))

    ent_avg = sum(entropies) / len(entropies)
    ent_std = math.sqrt(sum((x - ent_avg) ** 2 for x in entropies) / max(1, (len(entropies) - 1)))

    sorted_means = sorted(luma_means)

    return {
        "luma_mean_avg": float(mean_avg),
        "luma_mean_std": float(mean_std),
        "luma_p10": _percentile(sorted_means, 0.10),
        "luma_p50": _percentile(sorted_means, 0.50),
        "luma_p90": _percentile(sorted_means, 0.90),
        "luma_contrast_avg": float(sum(luma_stds) / len(luma_stds)),
        "entropy_avg": float(ent_avg),
        "entropy_std": float(ent_std),
    }


def scan_dataset(dataset_dir: Path, caption_extension: str, sample_limit: int = 200) -> DatasetStats:
    if not dataset_dir.exists():
        return DatasetStats(
            image_count=0,
            caption_count=0,
            caption_coverage=0.0,
            sample_count=0,
            min_side=None,
            max_side=None,
            avg_min_side=None,
            avg_aspect=None,
            mostly_square=None,
            luma_mean_avg=None,
            luma_mean_std=None,
            luma_p10=None,
            luma_p50=None,
            luma_p90=None,
            luma_contrast_avg=None,
            entropy_avg=None,
            entropy_std=None,
        )

    image_count = 0
    caption_count = 0

    # Reservoir sampling so huge datasets don't require loading all paths into RAM.
    sample_paths: List[Path] = []
    seen_images = 0

    for p in dataset_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in _VALID_IMAGE_EXTS:
            continue
        image_count += 1
        seen_images += 1

        cap = p.with_suffix(caption_extension)
        if cap.exists():
            caption_count += 1

        if sample_limit > 0:
            if len(sample_paths) < sample_limit:
                sample_paths.append(p)
            else:
                j = random.randint(0, seen_images - 1)
                if j < sample_limit:
                    sample_paths[j] = p
    min_sides: List[int] = []
    max_sides: List[int] = []
    aspects: List[float] = []
    sq: List[bool] = []

    for p in sample_paths:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            continue

        if w <= 0 or h <= 0:
            continue

        min_sides.append(min(w, h))
        max_sides.append(max(w, h))
        aspects.append(w / h)
        sq.append(0.9 <= (min(w, h) / max(w, h)) <= 1.0)

    sample_count = len(min_sides)
    min_side = min(min_sides) if min_sides else None
    max_side = max(max_sides) if max_sides else None
    avg_min_side = (sum(min_sides) / len(min_sides)) if min_sides else None
    avg_aspect = (sum(aspects) / len(aspects)) if aspects else None
    mostly_square = (sum(1 for x in sq if x) / len(sq) >= 0.6) if sq else None

    coverage = (caption_count / image_count) if image_count else 0.0

    content = _analyze_sample_images(sample_paths, downsample_max=256)

    return DatasetStats(
        image_count=image_count,
        caption_count=caption_count,
        caption_coverage=coverage,
        sample_count=sample_count,
        min_side=min_side,
        max_side=max_side,
        avg_min_side=avg_min_side,
        avg_aspect=avg_aspect,
        mostly_square=mostly_square,
        luma_mean_avg=content["luma_mean_avg"],
        luma_mean_std=content["luma_mean_std"],
        luma_p10=content["luma_p10"],
        luma_p50=content["luma_p50"],
        luma_p90=content["luma_p90"],
        luma_contrast_avg=content["luma_contrast_avg"],
        entropy_avg=content["entropy_avg"],
        entropy_std=content["entropy_std"],
    )


def _estimate_steps_per_epoch(
    *,
    instance_images: int,
    class_images: int,
    batch_size: int,
    grad_accum: int,
) -> int:
    effective_len = max(1, max(instance_images, class_images))
    bs = max(1, int(batch_size or 1))
    ga = max(1, int(grad_accum or 1))
    dataloader_len = int(math.ceil(effective_len / bs))
    steps = dataloader_len // ga
    return max(1, steps)


def _model_family(model_type: str) -> str:
    mt = (model_type or "").lower().strip()
    if mt in {"sd15", "sd1.5", "sd_legacy", "sd_1_5", "sd_15"}:
        return "sd15"
    if mt == "sdxl":
        return "sdxl"
    if mt in {"flux", "flux1", "flux2"}:
        return "flux"
    if mt in {"sd3", "sd3.0", "sd3.5"}:
        return "sd3"
    return mt or "unknown"


def _risk_score(*, stats: DatasetStats, train_text_encoder: bool, use_prior_preservation: bool) -> int:
    score = 0

    # Dataset size
    if stats.image_count <= 20:
        score += 2
    elif stats.image_count <= 60:
        score += 1

    # Caption coverage (missing captions -> weaker conditioning, easier to overfit)
    if stats.caption_coverage < 0.30:
        score += 2
    elif stats.caption_coverage < 0.70:
        score += 1

    # Diversity proxy
    if stats.entropy_avg is not None:
        if stats.entropy_avg < 4.25:
            score += 2
        elif stats.entropy_avg < 4.70:
            score += 1

    # Lighting extremes / unusual exposure
    brightness_extreme = False
    if stats.luma_p10 is not None and stats.luma_p10 < 35.0:
        brightness_extreme = True
    if stats.luma_p90 is not None and stats.luma_p90 > 220.0:
        brightness_extreme = True
    if stats.luma_mean_avg is not None and (stats.luma_mean_avg < 70.0 or stats.luma_mean_avg > 185.0):
        brightness_extreme = True
    if brightness_extreme:
        score += 1

    if train_text_encoder:
        score += 1

    # Prior preservation usually helps stability / reduces drift.
    if use_prior_preservation:
        score -= 1

    return int(_clamp(float(score), 0.0, 10.0))


def _target_steps_for(
    *,
    model_type: str,
    stats: DatasetStats,
    train_text_encoder: bool,
    use_prior_preservation: bool,
) -> int:
    family = _model_family(model_type)
    n = int(stats.image_count or 0)

    # Character/identity LoRAs are easy to overcook; start conservative.
    if family == "sdxl":
        if n <= 5:
            base = 300
        elif n <= 10:
            base = 450
        elif n <= 20:
            base = 700
        elif n <= 40:
            base = 1000
        elif n <= 80:
            base = 1500
        elif n <= 200:
            base = 2000
        elif n <= 1000:
            base = 2400
        elif n <= 2000:
            base = 2700
        elif n <= 10000:
            base = 3300
        else:
            base = 3600
    elif family == "sd15":
        if n <= 5:
            base = 500
        elif n <= 10:
            base = 800
        elif n <= 20:
            base = 1200
        elif n <= 40:
            base = 1700
        elif n <= 80:
            base = 2300
        elif n <= 200:
            base = 3000
        elif n <= 1000:
            base = 3600
        elif n <= 2000:
            base = 4000
        elif n <= 10000:
            base = 4600
        else:
            base = 5200
    elif family in {"flux", "sd3"}:
        if n <= 5:
            base = 220
        elif n <= 10:
            base = 350
        elif n <= 20:
            base = 550
        elif n <= 40:
            base = 800
        elif n <= 80:
            base = 1100
        elif n <= 200:
            base = 1600
        elif n <= 1000:
            base = 2000
        elif n <= 2000:
            base = 2200
        elif n <= 10000:
            base = 2600
        else:
            base = 3000
    else:
        base = 1200

    risk = _risk_score(stats=stats, train_text_encoder=train_text_encoder, use_prior_preservation=use_prior_preservation)
    if risk >= 5:
        base = int(base * 0.70)
    elif risk >= 3:
        base = int(base * 0.82)
    elif risk == 0 and n >= 200:
        base = int(base * 1.08)

    # A tiny nudge upward if PP is enabled (usually keeps identity from drifting).
    if use_prior_preservation:
        base = int(base * 1.05)

    return max(150, int(base))


def _recommended_learning_rate(
    *,
    model_type: str,
    stats: DatasetStats,
    train_text_encoder: bool,
    use_prior_preservation: bool,
) -> float:
    family = _model_family(model_type)
    n = int(stats.image_count or 0)

    if family == "sdxl":
        if n <= 10:
            lr = 1.6e-5 if train_text_encoder else 4.0e-5
        elif n <= 20:
            lr = 2.2e-5 if train_text_encoder else 5.0e-5
        elif n <= 40:
            lr = 3.0e-5 if train_text_encoder else 6.0e-5
        elif n <= 80:
            lr = 4.0e-5 if train_text_encoder else 8.0e-5
        elif n <= 200:
            lr = 5.0e-5 if train_text_encoder else 1.0e-4
        else:
            lr = 6.0e-5 if train_text_encoder else 1.2e-4
    elif family == "sd15":
        if n <= 10:
            lr = 5.0e-5 if train_text_encoder else 1.0e-4
        elif n <= 20:
            lr = 7.0e-5 if train_text_encoder else 1.5e-4
        elif n <= 40:
            lr = 1.0e-4 if train_text_encoder else 2.0e-4
        elif n <= 80:
            lr = 1.2e-4 if train_text_encoder else 2.5e-4
        elif n <= 200:
            lr = 1.5e-4 if train_text_encoder else 3.0e-4
        else:
            lr = 1.8e-4 if train_text_encoder else 3.0e-4
    elif family in {"flux", "sd3"}:
        if n <= 20:
            lr = 2.0e-5
        elif n <= 80:
            lr = 5.0e-5
        elif n <= 200:
            lr = 7.0e-5
        else:
            lr = 1.0e-4
    else:
        lr = 1.0e-4

    # Risk-driven LR dampening
    if stats.entropy_avg is not None and stats.entropy_avg < 4.70:
        lr *= 0.85
    if stats.caption_coverage < 0.70:
        lr *= 0.90
    if stats.caption_coverage < 0.30:
        lr *= 0.85

    brightness_extreme = False
    if stats.luma_p10 is not None and stats.luma_p10 < 35.0:
        brightness_extreme = True
    if stats.luma_p90 is not None and stats.luma_p90 > 220.0:
        brightness_extreme = True
    if stats.luma_mean_avg is not None and (stats.luma_mean_avg < 70.0 or stats.luma_mean_avg > 185.0):
        brightness_extreme = True
    if brightness_extreme:
        lr *= 0.92

    # Prior preservation generally tolerates slightly higher LR, but don't be aggressive.
    if use_prior_preservation:
        lr *= 1.03

    return float(_clamp(float(lr), 1e-6, 5e-4))


def recommend_training_patch(config: TrainingConfig, root_path: Path) -> Dict[str, Any]:
    """Return a patch dict (subset of TrainingConfig fields) tuned for the current dataset + model_type.

    Constraints:
    - Does NOT pick the model or architecture; uses whatever the user selected in config.
    - Does NOT download anything.
    """

    # Resolve dataset dir
    ds_path = Path(config.dataset_path)
    if not ds_path.is_absolute():
        ds_path = root_path / ds_path

    stats = scan_dataset(ds_path, config.caption_extension or ".txt", sample_limit=500)
    sysinfo = get_system_info()
    vram_gb = 0.0
    try:
        if sysinfo.get("gpu", {}).get("has_cuda") and sysinfo["gpu"]["gpus"]:
            vram_gb = float(sysinfo["gpu"]["gpus"][0].get("total_memory", 0.0))
    except Exception:
        vram_gb = 0.0

    notes: List[str] = []
    patch: Dict[str, Any] = {}
    model_type_norm = (config.model_type or "").lower().strip()
    model_family = _model_family(config.model_type)

    # ---- Dataset sanity ----
    if stats.image_count <= 0:
        notes.append(f"No images found in dataset dir: {_safe_relpath(root_path, ds_path)}")
        return {"patch": patch, "stats": asdict(stats), "notes": notes}

    # ---- Resolution ----
    # Keep the user's choice unless the dataset is clearly too small to justify it.
    if stats.min_side is not None:
        if model_family == "sdxl":
            if stats.min_side < 768:
                patch["resolution"] = 768
                notes.append("Dataset images are small; reduced SDXL resolution to 768 for stability.")
        elif model_family == "sd15":
            patch["resolution"] = 512
        elif model_family in {"sd3", "flux"}:
            if stats.min_side < 896:
                patch["resolution"] = 768

    # ---- Prior preservation (DreamBooth-style reg images) ----
    # Only enable if prompts exist; otherwise leave it as-is.
    if (config.instance_prompt or "").strip() and (config.class_prompt or "").strip():
        if stats.image_count <= 200:
            patch["use_prior_preservation"] = True
            patch["auto_generate_reg_images"] = True
            patch["reg_data_dir"] = config.reg_data_dir or "project/reg"

            # Keep a reasonable amount; clamp to avoid long reg generation.
            if stats.image_count <= 10:
                patch["num_class_images"] = 100
            elif stats.image_count <= 30:
                patch["num_class_images"] = 150
            elif stats.image_count <= 80:
                patch["num_class_images"] = 200
            else:
                patch["num_class_images"] = 300

            patch["prior_loss_weight"] = 1.0

            # Reg gen settings: keep stable defaults
            patch["reg_infer_steps"] = int(config.reg_infer_steps or 20)
            patch["reg_guidance_scale"] = float(config.reg_guidance_scale or 7.5)
            patch["reg_scheduler"] = config.reg_scheduler or "euler_a"

            # If class prompt looks like a full detailed caption, warn.
            cp = (config.class_prompt or "")
            if "," in cp and len([t for t in cp.split(",") if t.strip()]) > 3:
                notes.append("Class Prompt looks very detailed. For prior preservation, a generic class like 'girl' often works better.")

    use_pp = bool(patch.get("use_prior_preservation", config.use_prior_preservation))

    # ---- Text encoder training ----
    # For small datasets, training TE tends to overfit hard.
    train_te = bool(config.train_text_encoder)
    if stats.image_count <= 30:
        patch["train_text_encoder"] = False
        patch["train_text_encoder_frac"] = 0.0
        train_te = False
        notes.append("Disabled Text Encoder training (small dataset).")
    else:
        # Mid-size datasets: only train TE when captions/diversity support it.
        if 31 <= stats.image_count <= 80:
            ok_captions = stats.caption_coverage >= 0.70
            ok_diversity = (stats.entropy_avg is None) or (stats.entropy_avg >= 4.70)
            if train_te and not (ok_captions and ok_diversity):
                patch["train_text_encoder"] = False
                patch["train_text_encoder_frac"] = 0.0
                train_te = False
                notes.append("Disabled Text Encoder training (captions/diversity too weak for this dataset size).")

        # If TE training is enabled, cap it to a conservative fraction.
        if train_te:
            if model_family == "sdxl":
                frac_lo, frac_hi = 0.20, 0.35
            elif model_family == "sd15":
                frac_lo, frac_hi = 0.30, 0.50
            else:
                frac_lo, frac_hi = 0.20, 0.30
            patch["train_text_encoder_frac"] = float(_clamp(config.train_text_encoder_frac or 1.0, frac_lo, frac_hi))

    # ---- Learning rate ----
    lr = float(
        _recommended_learning_rate(
            model_type=config.model_type,
            stats=stats,
            train_text_encoder=train_te,
            use_prior_preservation=use_pp,
        )
    )
    patch["learning_rate"] = lr

    # Set a safer TE LR when TE training is on (engines use text_encoder_lr if set)
    if train_te:
        if model_family == "sdxl":
            patch["text_encoder_lr"] = float(lr * 0.35)
        elif model_family == "sd15":
            patch["text_encoder_lr"] = float(lr * 0.50)
        else:
            patch["text_encoder_lr"] = float(lr * 0.35)

    # ---- Epoch budget (SDXL/SD15 engines ignore max_train_steps) ----
    # Estimate steps/epoch using num_class_images when prior preservation is enabled.
    class_images = 0
    if patch.get("use_prior_preservation") or config.use_prior_preservation:
        class_images = int(patch.get("num_class_images") or config.num_class_images or 0)

    steps_per_epoch = _estimate_steps_per_epoch(
        instance_images=stats.image_count,
        class_images=class_images,
        batch_size=int(patch.get("batch_size") or config.batch_size or 1),
        grad_accum=int(patch.get("gradient_accumulation_steps") or config.gradient_accumulation_steps or 1),
    )

    target_steps = _target_steps_for(
        model_type=config.model_type,
        stats=stats,
        train_text_encoder=train_te,
        use_prior_preservation=use_pp,
    )

    # VRAM-aware nudge: if low VRAM and large per-epoch, reduce target steps slightly.
    if vram_gb and vram_gb <= 8:
        target_steps = int(target_steps * 0.85)

    max_train_steps = int(max(1, target_steps))
    patch["max_train_steps"] = max_train_steps

    epochs = int(math.ceil(max_train_steps / steps_per_epoch))
    epochs = int(_clamp(float(epochs), 1, 50))
    patch["max_train_epochs"] = epochs
    patch["save_every_n_epochs"] = 1

    # ---- Scheduler ----
    total_steps = int(max_train_steps)
    risk = _risk_score(stats=stats, train_text_encoder=train_te, use_prior_preservation=use_pp)
    if total_steps <= 350:
        patch["lr_scheduler"] = "constant"
        patch["lr_warmup_steps"] = 0
        patch["lr_warmup_ratio"] = 0.0
    else:
        patch["lr_scheduler"] = "cosine"
        # Prefer ratio over fixed warmup steps so it scales across budgets.
        patch["lr_warmup_steps"] = 0
        warmup_ratio = 0.03
        if total_steps <= 800:
            warmup_ratio = 0.05
        if risk >= 3:
            warmup_ratio = max(warmup_ratio, 0.06)
        patch["lr_warmup_ratio"] = float(_clamp(warmup_ratio, 0.0, 0.10))
        patch["lr_scheduler_num_cycles"] = 1
        patch["lr_scheduler_power"] = 1.0

    # ---- Loss / noise strategy ----
    luma_p10 = stats.luma_p10
    luma_p90 = stats.luma_p90
    luma_mean = stats.luma_mean_avg
    entropy_avg = stats.entropy_avg

    brightness_extreme = False
    if luma_p10 is not None and luma_p10 < 35.0:
        brightness_extreme = True
    if luma_p90 is not None and luma_p90 > 220.0:
        brightness_extreme = True
    if luma_mean is not None and (luma_mean < 70.0 or luma_mean > 185.0):
        brightness_extreme = True

    low_diversity = entropy_avg is not None and entropy_avg < 4.6

    if model_family in {"sdxl", "sd15"}:
        # Commonly stable defaults for kohya-style LoRA training.
        if stats.image_count <= 40:
            sg = 5.0
        elif stats.image_count <= 200:
            sg = 4.0
        else:
            sg = 3.0
        patch["min_snr_gamma"] = float(sg)
        patch["snr_gamma"] = float(sg)

        if model_family == "sdxl":
            patch["zero_terminal_snr"] = True

        if brightness_extreme:
            patch["noise_offset_type"] = "original"
            patch["noise_offset_strength"] = 0.10
        elif model_family == "sdxl":
            patch["noise_offset_type"] = "original"
            patch["noise_offset_strength"] = 0.0357

        if (stats.image_count <= 80 and low_diversity) or (risk >= 4):
            patch["loss_type"] = "huber"
            hc = float(config.huber_c or 0.1)
            if entropy_avg is not None and entropy_avg < 4.25:
                hc = max(hc, 0.2)
            patch["huber_c"] = float(_clamp(hc, 0.05, 0.3))

    # ---- Network scaling defaults ----
    # Rank guidance: small dataset -> smaller rank to reduce overfitting.
    if config.network_type == "lora":
        if stats.image_count <= 15:
            patch["network_dim"] = 8
        elif stats.image_count <= 60:
            patch["network_dim"] = 16
        # else leave user value

        # Alpha: keep <= dim and non-zero.
        dim = int(patch.get("network_dim") or config.network_dim or 32)
        alpha = float(config.network_alpha or 0)
        if alpha <= 0.0 or alpha > dim:
            patch["network_alpha"] = float(dim)

    # ---- Augmentation & caching ----
    # For identity/character LoRAs with small datasets, aggressive aug hurts more than helps.
    if stats.image_count <= 40:
        patch["flip_aug_probability"] = 0.0
        patch["color_aug_strength"] = 0.0
        patch["caption_dropout_rate"] = 0.0
    else:
        patch["flip_aug_probability"] = float(_clamp(config.flip_aug_probability or 0.5, 0.0, 0.5))
        patch["color_aug_strength"] = float(_clamp(config.color_aug_strength or 0.5, 0.0, 0.5))
        patch["caption_dropout_rate"] = float(_clamp(config.caption_dropout_rate or 0.05, 0.0, 0.1))

    # Cache latents to disk is generally a win on Windows (faster restarts), but avoid if aug is active.
    if float(patch.get("flip_aug_probability", config.flip_aug_probability)) == 0.0 and float(
        patch.get("color_aug_strength", config.color_aug_strength)
    ) == 0.0:
        patch["cache_latents_to_disk"] = True

    # ---- Misc safety ----
    # Make sure dataloader workers stay sane on Windows.
    patch["dataloader_num_workers"] = int(_clamp(float(config.dataloader_num_workers or 1), 1, 4))

    return {
        "patch": patch,
        "stats": asdict(stats),
        "notes": notes,
    }
