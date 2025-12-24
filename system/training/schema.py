from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Union, Dict

class TrainingConfig(BaseModel):
    # --- Model Config ---
    base_model_name: str = Field(..., description="Path to the base model (safetensors)")
    model_type: str = Field("sdxl", description="sdxl, sd_legacy, sd3, flux1")
    quantization_bit: Optional[str] = Field(None, description="4, 8, or None")
    output_name: str = Field("my_lora", description="Name of the output LoRA")
    output_dir: str = Field("project/outputs", description="Directory to save outputs")
    save_precision: str = Field("float16", description="float16, bf16, float32")
    save_model_as: str = Field("safetensors", description="safetensors, ckpt, diffusers")
    save_every_n_epochs: int = Field(1, description="Save every N epochs")
    save_every_n_steps: Optional[int] = Field(None, description="Save every N steps")
    save_best_only: bool = Field(False, description="Save only best models")
    checkpoints_total_limit: Optional[int] = Field(None, description="Max number of checkpoints to store")
    resume_from_checkpoint: Optional[str] = Field(None, description="Path to checkpoint or 'latest'")

    # --- Network Config ---
    network_type: str = Field("lora", description="lora, loha, lycoris, oft")
    network_dim: int = Field(32, description="Network dimension (rank)")
    network_alpha: float = Field(16.0, description="Network alpha")
    network_module: str = Field("networks.lora", description="Network module path")
    network_args: Optional[Union[str, List[str]]] = Field(None, description="Extra network arguments")
    
    # LyCORIS Specific
    algo: Optional[str] = Field("lora", description="LyCORIS algorithm")
    conv_dim: Optional[int] = Field(None, description="Conv dimension")
    conv_alpha: Optional[float] = Field(None, description="Conv alpha")
    
    # Dropout
    module_dropout: float = Field(0.0, description="Module dropout")
    rank_dropout: float = Field(0.0, description="Rank dropout")
    network_dropout: float = Field(0.0, description="Network dropout")
    dora_weight_decay: Optional[float] = Field(None, description="DoRA weight decay")

    # --- Dataset Config ---
    dataset_path: str = Field("project/dataset", description="Path to the dataset directory")
    use_prior_preservation: bool = Field(False, description="Enable prior preservation")
    instance_prompt: Optional[str] = Field(None, description="Instance prompt")
    class_prompt: Optional[str] = Field(None, description="Class prompt")
    reg_negative_prompt: Optional[str] = Field(None, description="Negative prompt for reg generation")
    reg_data_dir: Optional[str] = Field(None, description="Reg images directory")
    num_class_images: Optional[int] = Field(0, description="Number of class images")
    auto_generate_reg_images: bool = Field(False, description="Auto-generate reg images")
    reg_infer_steps: Optional[int] = Field(20, description="Reg inference steps")
    reg_guidance_scale: float = Field(7.5, description="Reg guidance scale")
    reg_scheduler: str = Field("euler_a", description="Scheduler for reg generation")
    reg_seed: Optional[int] = Field(-1, description="Reg seed")
    
    resolution: int = Field(1024, description="Training resolution")
    batch_size: int = Field(1, description="Batch size")
    max_train_epochs: int = Field(10, description="Max train epochs")
    max_train_steps: Optional[int] = Field(None, description="Max train steps")
    center_crop: bool = Field(True, description="Center crop images to 1:1 if they are close to square")
    enable_bucket: bool = Field(True, description="Enable aspect ratio bucketing")
    min_bucket_reso: int = Field(256, description="Min bucket resolution")
    max_bucket_reso: int = Field(2048, description="Max bucket resolution")
    bucket_reso_steps: int = Field(64, description="Bucket resolution step")
    bucket_no_upscale: bool = Field(False, description="No upscale")
    persistent_data_loader_workers: bool = Field(False, description="Persistent data loader workers")
    dataloader_num_workers: int = Field(8, description="DataLoader workers")
    dataloader_shuffle: bool = Field(True, description="DataLoader shuffle")

    # --- Text Encoder Config ---
    train_text_encoder: bool = Field(True, description="Train text encoder")
    train_text_encoder_ti: bool = Field(False, description="Train textual inversion (pivotal tuning)")
    token_abstraction: str = Field("TOK", description="Token abstraction for TI (e.g. TOK)")
    num_new_tokens_per_abstraction: Optional[int] = Field(2, description="Number of new tokens per abstraction")
    train_text_encoder_ti_frac: float = Field(0.5, description="Fraction of training for TI")
    train_text_encoder_frac: float = Field(1.0, description="Fraction of training for TE")
    clip_skip: Optional[int] = Field(0, description="Clip skip")
    max_token_length: Optional[int] = Field(None, description="Max token length")
    stop_text_encoder_training_pct: Optional[int] = Field(0, description="Stop TE training at %")
    weighted_captions: bool = Field(False, description="Weighted captions")
    emphasis_strength: float = Field(1.2, description="Emphasis strength")
    de_emphasis_strength: float = Field(0.8, description="De-emphasis strength")

    # --- Augmentation & Caching ---
    cache_latents: bool = Field(True, description="Cache latents")
    cache_latents_to_disk: bool = Field(False, description="Cache latents to disk")
    cache_text_embeddings: bool = Field(False, description="Cache text embeddings")
    vae_batch_size: Optional[int] = Field(None, description="VAE batch size")
    
    # Noise Offset & Multires
    noise_offset_type: str = Field("original", description="original, alternative")
    noise_offset_strength: float = Field(0.0, description="Noise offset strength")
    noise_offset_random_strength: float = Field(0.0, description="Noise offset random strength")
    adaptive_noise_scale: float = Field(0.0, description="Adaptive noise scale")
    multires_noise_iterations: int = Field(0, description="Multires noise iterations")
    multires_noise_discount: float = Field(0.0, description="Multires noise discount")
    min_timestep: Optional[int] = Field(None, description="Minimum diffusion timestep to sample (inclusive)")
    max_timestep: Optional[int] = Field(None, description="Maximum diffusion timestep to sample (inclusive)")
    v_pred_like_loss: float = Field(0.0, description="Add v-prediction-like loss scaling (epsilon/noise prediction only)")
    min_snr_gamma: Optional[float] = Field(None, description="Min SNR gamma")
    snr_gamma: Optional[float] = Field(None, description="SNR gamma (alias for min_snr_gamma)")
    do_edm_style_training: bool = Field(False, description="Enable EDM-style training")

    # Image Augmentation
    augmentation_mode: str = Field("always", description="always, per_epoch, random")
    color_aug_strength: float = Field(0.5, description="Color augmentation strength")
    flip_aug_probability: float = Field(0.5, description="Flip augmentation probability")
    random_crop_scale: float = Field(1.0, description="Random crop scale")

    # Caption Augmentation
    caption_dropout_rate: float = Field(0.0, description="Caption dropout rate")
    caption_dropout_every_n_epochs: int = Field(0, description="Caption dropout every N epochs")
    keep_tokens: int = Field(0, description="Keep tokens")
    shuffle_caption: bool = Field(False, description="Shuffle captions")
    caption_extension: str = Field(".txt", description="Caption extension")

    # --- Learning (Basic) ---
    learning_rate: float = Field(1e-4, description="Learning rate")
    unet_lr: Optional[float] = Field(None, description="UNet learning rate")
    text_encoder_lr: Optional[float] = Field(None, description="Text encoder learning rate")
    lr_scheduler: str = Field("constant", description="LR scheduler")
    lr_warmup_steps: int = Field(0, description="LR warmup steps")
    lr_warmup_ratio: float = Field(0.0, description="LR warmup ratio")
    lr_scheduler_num_cycles: int = Field(1, description="LR scheduler cycles")
    lr_scheduler_power: float = Field(1.0, description="LR scheduler power")
    optimizer_type: str = Field("AdamW8bit", description="Optimizer type")
    optimizer_args: Optional[Union[str, List[str]]] = Field(None, description="Optimizer args")
    
    # EMA
    ema_unet: bool = Field(False, description="EMA for UNet")
    ema_text_encoder: bool = Field(False, description="EMA for Text Encoder")
    ema_decay: float = Field(0.995, description="EMA decay")

    # --- Advanced ---
    mixed_precision: str = Field("fp16", description="fp16, bf16, no")
    gradient_accumulation_steps: int = Field(1, description="Gradient accumulation steps")
    gradient_checkpointing: bool = Field(True, description="Gradient checkpointing")
    full_fp16: bool = Field(False, description="Full fp16")
    full_bf16: bool = Field(False, description="Full bf16")
    no_half_vae: bool = Field(False, description="No half VAE")
    xformers: bool = Field(True, description="Use xformers")
    attention_backend: str = Field("sdpa", description="sdpa, xformers, none")
    allow_tf32: bool = Field(True, description="Allow TF32")
    enable_aggressive_memory_saving: bool = Field(False, description="Aggressive memory saving")
    mem_eff_attn: bool = Field(False, description="Memory efficient attention")
    seed: Optional[int] = Field(None, description="Seed")
    max_grad_norm: Optional[float] = Field(1.0, description="Max gradient norm")
    
    # Loss
    loss_type: str = Field("l2", description="Loss type: l2, huber, smooth_l1")
    huber_c: float = Field(0.1, description="Huber loss C parameter")
    debiased_estimation_loss: bool = Field(False, description="Debiased estimation loss")
    scale_v_pred_loss_like_noise_pred: bool = Field(False, description="Scale v-prediction loss like noise prediction")
    zero_terminal_snr: bool = Field(False, description="Fix scheduler betas for zero terminal SNR")
    ip_noise_gamma: float = Field(0.0, description="Input perturbation noise gamma (0 = disabled)")
    masked_loss: bool = Field(False, description="Use alpha channel as loss mask")
    prior_loss_weight: float = Field(1.0, description="Prior loss weight")
    weighting_scheme: str = Field("none", description="none, logit_normal, mode")
    logit_mean: float = Field(0.0, description="Logit mean for logit_normal weighting")
    logit_std: float = Field(1.0, description="Logit std for logit_normal weighting")
    mode_scale: float = Field(1.29, description="Mode scale for mode weighting")
    guidance_scale: float = Field(3.5, description="Guidance scale for Flux/SD3")
    scheduled_huber_schedule: str = Field("constant", description="Scheduled Huber schedule")
    scheduled_huber_c: float = Field(0.1, description="Scheduled Huber C")
    scheduled_huber_scale: float = Field(1.0, description="Scheduled Huber scale")
    scale_weight_norms: Optional[float] = Field(None, description="Scale weight norms")
    
    # SD3/Flux Specific
    lora_layers: Optional[str] = Field(None, description="Target layers for LoRA (comma separated)")
    lora_blocks: Optional[str] = Field(None, description="Target blocks for LoRA (comma separated)")
    max_sequence_length: int = Field(256, description="Max sequence length for T5 (Flux/SD3)")

    # --- Samples ---
    sample_every_n_epochs: Optional[int] = Field(None, description="Sample every N epochs")
    sample_every_n_steps: Optional[int] = Field(None, description="Sample every N steps")
    sample_prompts: str = Field("", description="Sample prompts")
    sample_negative_prompt: Optional[str] = Field(None, description="Sample negative prompt")
    num_validation_images: int = Field(4, description="Number of validation images")
    sample_sampler: str = Field("euler_a", description="Sample sampler")
    sample_num_inference_steps: int = Field(20, description="Sample inference steps")
    sample_guidance_scale: float = Field(7.5, description="Sample guidance scale")
    sample_seed: int = Field(-1, description="Sample seed")

    # --- Metadata ---
    training_comment: Optional[str] = Field(None, description="Training comment")
    metadata_title: Optional[str] = Field(None, description="Metadata title")
    metadata_author: Optional[str] = Field(None, description="Metadata author")
    metadata_description: Optional[str] = Field(None, description="Metadata description")
    metadata_license: Optional[str] = Field(None, description="Metadata license")
    metadata_tags: Optional[str] = Field(None, description="Metadata tags")

    @model_validator(mode="after")
    def _normalize_latent_cache_flags(self):
        # Keep behavior compatible with kohya-style semantics: disk caching implies caching is enabled.
        if self.cache_latents_to_disk:
            self.cache_latents = True
        return self
