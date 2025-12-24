from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests

from system.log import error, info, success, warning
from system.coordinator_settings import SETTINGS

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
REMOTE_CONFIG_PATH = "config/configs.json" # Changed from system_config.json to match Onika
LOCAL_CHECKSUM_PATH = os.path.join(ROOT_DIR, "config", "checksum.json")
DEFAULT_HEADERS = {"User-Agent": "Onika-Updater/1.0"}


def _get_http_timeout() -> int:
    # Default to 30 seconds if not specified
    return SETTINGS.get('http_timeout', 30)


def _get_remote_base_url() -> str:
    url = SETTINGS.get('update_repository_url', '')
    if url and not url.endswith('/'):
        url += '/'
    return url


def _get_skip_paths() -> Set[str]:
    paths = SETTINGS.get('update_skip_paths', ["config/configs.json"])
    return set(paths) if isinstance(paths, list) else {"config/configs.json"}


class UpdateError(RuntimeError):
    pass


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _safe_join(path: str) -> str:
    normalized = _normalize_path(path)
    full_path = os.path.abspath(os.path.join(ROOT_DIR, normalized))
    if not full_path.startswith(ROOT_DIR):
        raise UpdateError(f"Unsafe path detected: {normalized}")
    return full_path


def _load_local_checksums() -> Dict[str, str]:
    try:
        if not os.path.exists(LOCAL_CHECKSUM_PATH):
            warning("Local checksum file missing; assuming empty manifest.")
            return {}
            
        with open(LOCAL_CHECKSUM_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise UpdateError("Local checksum file is not a valid mapping")
        cleaned = {_normalize_path(key): str(value) for key, value in data.items()}
        return cleaned
    except FileNotFoundError:
        warning("Local checksum file missing; assuming empty manifest.")
        return {}
    except json.JSONDecodeError as exc:
        raise UpdateError(f"Local checksum file is invalid: {exc}") from exc


def _write_local_checksums(entries: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(LOCAL_CHECKSUM_PATH), exist_ok=True)
    with open(LOCAL_CHECKSUM_PATH, "w", encoding="utf-8") as handle:
        json.dump(dict(sorted(entries.items())), handle, indent=2, sort_keys=True)


def _http_get(url: str) -> bytes:
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=_get_http_timeout())
        response.raise_for_status()
        return response.content
    except requests.RequestException as exc:
        raise UpdateError(str(exc)) from exc


def _http_get_json(url: str) -> Dict[str, Any]:
    payload = _http_get(url)
    try:
        data = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise UpdateError(str(exc)) from exc
    if not isinstance(data, dict):
        raise UpdateError(f"Remote payload is not a valid mapping: {url}")
    return data


def _fetch_remote_config() -> Dict[str, Any]:
    url = f"{_get_remote_base_url()}{REMOTE_CONFIG_PATH}"
    try:
        return _http_get_json(url)
    except UpdateError as exc:
        raise UpdateError(f"Failed to fetch remote config: {exc}") from exc


def _resolve_remote_checksum_path(remote_config: Dict[str, Any]) -> str:
    # In Onika, we assume config/checksum.json
    return "config/checksum.json"


def _fetch_remote_manifest(remote_config: Dict[str, Any]) -> Tuple[Dict[str, str], str]:
    checksum_rel = _resolve_remote_checksum_path(remote_config)
    checksum_url = f"{_get_remote_base_url()}{checksum_rel}"
    try:
        payload = _http_get_json(checksum_url)
    except UpdateError as exc:
        raise UpdateError(f"Failed to fetch remote manifest: {exc}") from exc

    manifest: Dict[str, str] = {}
    for key, value in payload.items():
        normalized = _normalize_path(key)
        manifest[normalized] = str(value)
    return manifest, checksum_url


def _build_plan(
    selected_paths: Optional[Iterable[str]] = None
) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, str], List[Dict[str, Any]], List[str], List[str], str]:
    remote_config = _fetch_remote_config()
    remote_manifest, checksum_url = _fetch_remote_manifest(remote_config)
    local_manifest = _load_local_checksums()

    skip_set = _get_skip_paths()
    normalized_skip_set = {_normalize_path(item) for item in skip_set}
    normalized_selection: Optional[Set[str]] = None
    if selected_paths is not None:
        normalized_selection = {_normalize_path(path) for path in selected_paths}

    plan_entries: List[Dict[str, Any]] = []
    for path, remote_hash in remote_manifest.items():
        if path in normalized_skip_set:
            continue
        if normalized_selection is not None and path not in normalized_selection:
            continue
        local_hash = local_manifest.get(path)
        if local_hash == remote_hash:
            continue
        status_key = "updates.status.missing" if local_hash is None else "updates.status.outdated"
        plan_entries.append(
            {
                "path": path,
                "status": "missing" if local_hash is None else "outdated",
                "statusLangKey": status_key,
                "local_checksum": local_hash,
                "remote_checksum": remote_hash,
            }
        )

    orphaned = sorted(path for path in local_manifest.keys() if path not in remote_manifest and path not in normalized_skip_set)
    missing_targets: List[str] = []
    if normalized_selection is not None:
        missing_targets = sorted(path for path in normalized_selection if path not in remote_manifest)

    return (
        remote_config,
        remote_manifest,
        local_manifest,
        plan_entries,
        orphaned,
        missing_targets,
        checksum_url,
    )


def check_for_updates(selected_paths: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    info("Starting update check...")
    timestamp = datetime.utcnow().isoformat() + "Z"

    try:
        (
            remote_config,
            remote_manifest,
            local_manifest,
            entries,
            orphaned,
            missing_targets,
            checksum_url,
        ) = _build_plan(selected_paths)
    except UpdateError as exc:
        error(f"Update check failed: {exc}")
        return {
            "status": "error",
            "messageKey": "updates.api.check_failed",
            "messageParams": {"error": str(exc)},
            "checked_at": timestamp,
        }

    count = len(entries)
    info(f"Update check complete: {count} file(s) to update.")

    return {
        "status": "success",
        "messageKey": "updates.api.check_success" if count else "updates.api.check_no_updates",
        "messageParams": {"count": count} if count else {},
        "has_updates": bool(count),
        "files": entries,
        "orphaned": orphaned,
        "missing_targets": missing_targets,
        "skipped": sorted(_get_skip_paths()),
        "reference": {
            "base_url": _get_remote_base_url(),
            "config_url": f"{_get_remote_base_url()}{REMOTE_CONFIG_PATH}",
            "checksum_url": checksum_url,
        },
        "checked_at": timestamp,
        "remote_manifest_size": len(remote_manifest),
        "local_manifest_size": len(local_manifest),
        "remote_version_hint": remote_config.get("version"),
    }


def _download_and_store(path: str, remote_manifest: Dict[str, str], local_manifest: Dict[str, str]) -> None:
    url = f"{_get_remote_base_url()}{path}"
    payload = _http_get(url)
    target_path = _safe_join(path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "wb") as handle:
        handle.write(payload)
    local_manifest[path] = remote_manifest[path]


def apply_updates(selected_paths: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    info("Starting update apply...")
    timestamp = datetime.utcnow().isoformat() + "Z"

    try:
        (
            remote_config,
            remote_manifest,
            local_manifest,
            entries,
            orphaned,
            missing_targets,
            checksum_url,
        ) = _build_plan(selected_paths)
    except UpdateError as exc:
        error(f"Update plan failed: {exc}")
        return {
            "status": "error",
            "messageKey": "updates.api.apply_failed",
            "messageParams": {"error": str(exc)},
            "checked_at": timestamp,
        }

    if not entries:
        info("No updates to apply.")
        return {
            "status": "noop",
            "messageKey": "updates.api.apply_nothing",
            "updated": [],
            "failed": [],
            "orphaned": orphaned,
            "missing_targets": missing_targets,
            "skipped": sorted(_get_skip_paths()),
            "reference": {
                "base_url": _get_remote_base_url(),
                "config_url": f"{_get_remote_base_url()}{REMOTE_CONFIG_PATH}",
                "checksum_url": checksum_url,
            },
            "checked_at": timestamp,
        }

    updated: List[str] = []
    failed: List[Dict[str, Any]] = []

    for entry in entries:
        path = entry["path"]
        try:
            _download_and_store(path, remote_manifest, local_manifest)
            updated.append(path)
            success(f"Updated: {path}")
        except UpdateError as exc:
            failed.append({"path": path, "error": str(exc)})
            error(f"Failed to update {path}: {exc}")
        except OSError as exc:
            failed.append({"path": path, "error": str(exc)})
            error(f"Failed to update {path}: {exc}")

    if updated:
        _write_local_checksums(local_manifest)

    if failed:
        warning(f"Partial update completed: {len(updated)} updated, {len(failed)} failed.")
        status = "partial"
        message_key = "updates.api.apply_partial"
        message_params = {"updated": len(updated), "failed": len(failed)}
    else:
        success(f"Update complete: {len(updated)} file(s) updated.")
        status = "success"
        message_key = "updates.api.apply_success"
        message_params = {"updated": len(updated)}

    return {
        "status": status,
        "messageKey": message_key,
        "messageParams": message_params,
        "updated": updated,
        "failed": failed,
        "orphaned": orphaned,
        "missing_targets": missing_targets,
        "skipped": sorted(_get_skip_paths()),
        "reference": {
            "base_url": _get_remote_base_url(),
            "config_url": f"{_get_remote_base_url()}{REMOTE_CONFIG_PATH}",
            "checksum_url": checksum_url,
        },
        "checked_at": timestamp,
    }
