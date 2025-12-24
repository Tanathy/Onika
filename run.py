"""FlixyAI main entry point.

This script defines the project ROOT directory (the directory containing
run.py) and loads the merged configuration from config.json and config_user.json.
All modules can access the merged configuration through coordinator_settings.SETTINGS.

CLI-only behavior note: All log and error messages in this file are
intentionally hard-coded in English per the system requirement that
command-line flows do not use localization.
"""


from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
import json
import re
import platform

# Import coordinator settings module (used for package metadata/settings)
from system import coordinator_settings as cs

# Define project root as the directory containing this script
ROOT = Path(__file__).resolve().parent

# Add the project root to sys.path so modules can be imported
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from system.log import info, warning, error, success, progress_context


def main() -> None:
	# --- Auto-create virtual environment if missing ---
	venv_dir = os.path.join(ROOT, "venv")

	# Determine venv python path early so we can use it both when creating and when venv exists
	venv_created = False

	# determine platform-specific paths
	sys_platform = platform.system().lower()
	if sys_platform == "windows":
		venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
	else:
		venv_python = os.path.join(venv_dir, "bin", "python")

	# Helper: path to pip via venv python is venv_python -m pip
	if not os.path.isdir(venv_dir):
		info(f"Creating Python virtual environment at {venv_dir}...")
		result = subprocess.run([sys.executable, "-m", "venv", venv_dir], capture_output=True, text=True)
		if result.returncode == 0:
			success("Virtual environment ready.")
		else:
			error(f"Virtual environment creation failed: {result.stderr or result.stdout}")
			error(result.stderr or result.stdout)
			raise RuntimeError("Virtualenv creation failed")
		venv_created = True
	else:
		info("Virtual environment already present.")

	# --- PyTorch/xFormers auto-install if venv was just created (config-driven) ---
	if venv_created:
		import shutil

		# First, upgrade pip in the venv
		info("Upgrading pip inside the virtual environment...")
		result = subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], capture_output=True, text=True)
		if result.returncode == 0:
			success("Pip upgrade completed.")
		else:
			error(f"Pip upgrade failed: {result.stderr or result.stdout}")
			error(result.stderr or result.stdout)
			raise RuntimeError("Pip upgrade failed")

		def detect_cuda_version():
			# Try to detect CUDA version from nvcc or nvidia-smi
			cuda_version = None
			# Try nvcc
			nvcc = shutil.which("nvcc")
			if nvcc:
				try:
					out = subprocess.check_output([nvcc, "--version"], encoding="utf-8", errors="ignore")
					info(f"nvcc --version output:\n{out}")
					match = re.search(r"release (\d+\.\d+)", out)
					if match:
						cuda_version = match.group(1)
				except Exception as e:
					warning(f"Failed to query nvcc: {e}")
					pass
			# Try nvidia-smi
			if not cuda_version:
				nvsmi = shutil.which("nvidia-smi")
				if nvsmi:
					try:
						out = subprocess.check_output([nvsmi], encoding="utf-8", errors="ignore")
						info(f"nvidia-smi output:\n{out}")
						match = re.search(r"CUDA Version: (\d+\.\d+)", out)
						if match:
							cuda_version = match.group(1)
					except Exception as e:
						warning(f"Failed to query nvidia-smi: {e}")
						pass
			info(f"Detected CUDA version: {cuda_version}")
			return cuda_version

		system = platform.system().lower()
		cuda_version = detect_cuda_version()
		# Always invoke pip through the venv python -m pip for consistency
		pip_module_cmd = [venv_python, "-m", "pip"]
		torch_cmd = None

		if cuda_version:
			if cuda_version.startswith("12.6"):
				torch_cmd = pip_module_cmd + ["install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu126"]
			elif cuda_version.startswith("12.8"):
				torch_cmd = pip_module_cmd + ["install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128"]
			elif cuda_version.startswith("12.9"):
				torch_cmd = pip_module_cmd + ["install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu129"]
			elif cuda_version.startswith("13"):
				torch_cmd = pip_module_cmd + ["install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu130"]
			else:
				# Fallback for other CUDA versions - try the standard PyTorch with latest CUDA support
				torch_cmd = pip_module_cmd + ["install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu130"]

		# Always install torch when venv is created
		if torch_cmd:
			info(f"Installing PyTorch for CUDA {cuda_version} on {system}...")
			result = subprocess.run(torch_cmd, shell=False, capture_output=True, text=True)
			if result.returncode == 0:
				success("PyTorch installation finished.")
			else:
				error(f"PyTorch installation failed: {result.stderr or result.stdout}")
				error(result.stderr or result.stdout)
				raise RuntimeError("PyTorch install failed")
			# --- Always install xformers matching CUDA/ROCm version ---
			xformers_cmd = None
			if cuda_version:
				if cuda_version.startswith("12.6"):
					xformers_cmd = pip_module_cmd + ["install", "-U", "xformers", "--index-url", "https://download.pytorch.org/whl/cu126"]
				elif cuda_version.startswith("12.8"):
					xformers_cmd = pip_module_cmd + ["install", "-U", "xformers", "--index-url", "https://download.pytorch.org/whl/cu128"]
				elif cuda_version.startswith("12.9"):
					xformers_cmd = pip_module_cmd + ["install", "-U", "xformers", "--index-url", "https://download.pytorch.org/whl/cu129"]
				elif cuda_version.startswith("13"):
					xformers_cmd = pip_module_cmd + ["install", "-U", "xformers", "--index-url", "https://download.pytorch.org/whl/cu130"]
			elif system == "linux":
				# Experimental ROCm support
				try:
					import torch  # type: ignore
					_tver = getattr(torch, "version", None)
					if _tver is not None and getattr(_tver, "hip", None):
						xformers_cmd = pip_module_cmd + ["install", "-U", "xformers", "--index-url", "https://download.pytorch.org/whl/rocm6.4"]
				except Exception:
					pass
			if xformers_cmd:
				# Only install if not already present
				try:
					out = subprocess.check_output([venv_python, "-m", "pip", "show", "xformers"], encoding="utf-8", stderr=subprocess.DEVNULL)
					if out:
						info("xFormers is already installed and up to date.")
					else:
						raise Exception("not installed")
				except Exception:
					info(f"Installing xFormers (CUDA {cuda_version or 'unknown'})...")
					result = subprocess.run(xformers_cmd, shell=False, capture_output=True, text=True)
					if result.returncode == 0:
						success("xFormers installation finished.")
					else:
						error(f"xFormers installation failed: {result.stderr or result.stdout}")
						# Don't raise, just warn; user can fix manually if needed
		else:
			info(f"Skipping PyTorch install (detected CUDA={cuda_version}, platform={system}).")

	info("Loading environment configuration...")

	# Ensure environment packages declared in PACKAGES are present in the venv.
	def ensure_environment_packages(venv_path: str) -> None:
		"""Ensure packages from PACKAGES['environment_packages'] are installed into venv.

		Behavior:
		- Refreshes PACKAGES by calling coordinator_settings.reload_packages().
		- Reads PACKAGES['environment_packages'] (expects list of pip-style specs, e.g. "pkg==1.2.3").
		- If PACKAGES['pytorch_cuda_version'] is defined, uses that for PyTorch/xformers installation instead of autodetection.
		- Uses the venv pip to query installed packages and installs missing ones one-by-one.
		- Emits hard-coded English progress messages for CLI clarity.
		"""
		try:
			cs.reload_packages()
		except Exception:
			# Non-fatal: proceed with whatever PACKAGES is available
			pass

		env_pkgs = []
		pytorch_cuda_version = None
		if isinstance(cs.PACKAGES, dict):
			env_pkgs = cs.PACKAGES.get("environment_packages") or []
			pytorch_cuda_version = cs.PACKAGES.get("pytorch_cuda_version")

		if not env_pkgs:
			info("No environment packages defined in configuration.")
			return

		# Use venv python to run pip commands (python -m pip ...)
		sys_platform = platform.system().lower()
		if sys_platform == "windows":
			venv_python_local = os.path.join(venv_path, "Scripts", "python.exe")
		else:
			venv_python_local = os.path.join(venv_path, "bin", "python")

		info("Checking installed environment packages...")

		# Probe installed packages inside venv
		installed_map: dict[str, str] = {}
		try:
			out = subprocess.check_output([venv_python_local, "-m", "pip", "list", "--format=json"], encoding="utf-8", stderr=subprocess.DEVNULL)
			data = json.loads(out or "[]")
			for entry in data:
				name = (entry.get("name") or "").lower().replace("-", "_")
				ver = entry.get("version") or ""
				installed_map[name] = ver
			info(f"Found {len(installed_map)} installed packages in virtual environment.")
		except Exception:
			# If we couldn't query, treat as empty -> install all
			warning("Could not query installed packages; will attempt to verify all requirements.")
			installed_map = {}

		# If user pinned a CUDA build, ensure torch stack matches it even if not listed in environment_packages
		# CRITICAL: Check if installed torch is CPU-only (+cpu suffix) and force reinstall if user wants CUDA
		if pytorch_cuda_version:
			torch_index = f"https://download.pytorch.org/whl/{pytorch_cuda_version}"
			torch_stack = ["torch", "torchvision", "torchaudio"]
			
			# Check if torch is installed and whether it's a CPU build
			need_torch_install = False
			torch_ver = installed_map.get("torch", "")
			if not torch_ver:
				# Not installed at all
				need_torch_install = True
			elif "+cpu" in torch_ver:
				# CPU-only build installed but user wants CUDA - FORCE REINSTALL
				info(f"Detected CPU-only PyTorch ({torch_ver}), but CUDA version {pytorch_cuda_version} requested. Force reinstalling...")
				need_torch_install = True
			elif f"+{pytorch_cuda_version}" not in torch_ver and "+cu" not in torch_ver:
				# Has torch but might be wrong CUDA version or no CUDA suffix
				# Check by actually testing CUDA availability
				try:
					cuda_check = subprocess.run(
						[venv_python_local, "-c", "import torch; print('1' if torch.cuda.is_available() else '0')"],
						capture_output=True, text=True, timeout=10
					).stdout.strip()
					if cuda_check != "1":
						info(f"PyTorch installed but CUDA not available. Force reinstalling for {pytorch_cuda_version}...")
						need_torch_install = True
				except Exception:
					need_torch_install = True
			
			if need_torch_install:
				info(f"Installing PyTorch stack for {pytorch_cuda_version}...")
				result = subprocess.run(
					[venv_python_local, "-m", "pip", "install", "--force-reinstall", *torch_stack, "--index-url", torch_index],
					shell=False, capture_output=True, text=True
				)
				if result.returncode == 0:
					success(f"PyTorch stack installed for {pytorch_cuda_version}")
					# Refresh installed map after torch install
					try:
						out = subprocess.check_output([venv_python_local, "-m", "pip", "list", "--format=json"], encoding="utf-8", stderr=subprocess.DEVNULL)
						data = json.loads(out or "[]")
						for entry in data:
							name = (entry.get("name") or "").lower().replace("-", "_")
							installed_map[name] = entry.get("version") or ""
					except Exception: pass
				else:
					warning(f"PyTorch stack install failed: {result.stderr or result.stdout}")

		# Helper to extract base name from spec (before any comparison operator)
		def _spec_base_name(spec: str) -> str:
			m = re.match(r"^\s*([A-Za-z0-9_.+-]+)", spec)
			return (m.group(1) if m else spec).lower().replace("-", "_")

		# Process packages with special handling for PyTorch/xformers if cuda version is specified
		processed_pkgs = []
		for spec in env_pkgs:
			if isinstance(spec, str):
				base_name = _spec_base_name(spec)
				# Handle PyTorch installation with specified CUDA version
				if pytorch_cuda_version and base_name in ["torch", "torchvision", "torchaudio"]:
					# Replace with CUDA-specific version
					cuda_spec = f"{base_name} --index-url https://download.pytorch.org/whl/{pytorch_cuda_version}"
					processed_pkgs.append(cuda_spec)
				# Handle xformers with compatible CUDA version
				elif pytorch_cuda_version and base_name == "xformers":
					# Map CUDA version to compatible xformers index URL
					cuda_index_map = {
						"cu121": "https://download.pytorch.org/whl/cu121",
						"cu122": "https://download.pytorch.org/whl/cu122",
						"cu123": "https://download.pytorch.org/whl/cu123",
						"cu124": "https://download.pytorch.org/whl/cu124",
						"cu125": "https://download.pytorch.org/whl/cu125",
						"cu126": "https://download.pytorch.org/whl/cu126",
						"cu127": "https://download.pytorch.org/whl/cu127",
						"cu128": "https://download.pytorch.org/whl/cu128",
						"cu129": "https://download.pytorch.org/whl/cu129",
						"cu130": "https://download.pytorch.org/whl/cu130"
					}
					xformers_index = cuda_index_map.get(pytorch_cuda_version, "https://download.pytorch.org/whl/cu130")
					xformers_spec = f"{spec} --index-url {xformers_index}"
					processed_pkgs.append(xformers_spec)
				else:
					processed_pkgs.append(spec)
			else:
				processed_pkgs.append(spec)

		# Identify which packages actually need installation
		to_install = []
		import shlex
		for spec in processed_pkgs:
			if isinstance(spec, str):
				base = _spec_base_name(spec)
				spec_args = shlex.split(spec, posix=True)
			elif isinstance(spec, list):
				try:
					base = _spec_base_name(str(spec[0]))
					spec_args = [str(x) for x in spec]
				except Exception: continue
			else: continue

			need_install = False
			if base not in installed_map:
				need_install = True
			else:
				raw_spec = spec if isinstance(spec, str) else (spec_args[0] if spec_args else "")
				if isinstance(raw_spec, str) and "==" in raw_spec:
					req_ver = raw_spec.split("==", 1)[1]
					if installed_map.get(base) != req_ver:
						need_install = True
			
			if need_install:
				to_install.append((spec, spec_args))

		if not to_install:
			success("All environment packages are already installed and up to date.")
			return

		info(f"Found {len(to_install)} missing or outdated packages. Starting installation...")

		# Use a progress bar while attempting package installs
		desc_text = "Installing environment packages"
		with progress_context(total=len(to_install), desc=desc_text, unit="pkgs") as pbar:
			for spec, spec_args in to_install:
				info(f"Installing environment package {spec}...")
				result = subprocess.run([venv_python_local, "-m", "pip", "install", *spec_args], shell=False, capture_output=True, text=True)
				if result.returncode == 0:
					success(f"Environment package installed: {spec}")
				else:
					error(f"Failed to install environment package {spec}: {result.stderr or result.stdout}")
				pbar.update(1)


	# Always ensure environment packages after venv is created and after PyTorch install (if any)
	ensure_environment_packages(venv_dir)

	# Build environment map for the child process (without mutating current os.environ)
	# Handle CUDA allocator configuration from settings
	child_env = {}
	try:
		settings = cs.reload_settings()
		perf = (settings or {}).get("performance", {}) or {}
		alloc_conf = str(perf.get("cuda_allocator", "")).strip()

		if alloc_conf:
			# Set PYTORCH_CUDA_ALLOC_CONF for the child process only
			if alloc_conf == "cudaMallocAsync":
				child_env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
			elif alloc_conf == "native":
				child_env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native"
			elif alloc_conf == "malloc":
				child_env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:malloc"

		# CUDA launch blocking for debugging
		if perf.get("cuda_launch_blocking", False):
			child_env["CUDA_LAUNCH_BLOCKING"] = "1"  # Forces synchronous CUDA ops for debugging

		# cuBLAS workspace configuration for better performance
		cublas_config = str(perf.get("cublas_workspace_config", "")).strip()
		if cublas_config:
			child_env["CUBLAS_WORKSPACE_CONFIG"] = cublas_config  # Optimizes cuBLAS performance

		# CUDA Device-Side Assertions for debugging GPU kernels
		if perf.get("torch_use_cuda_dsa", False):
			child_env["TORCH_USE_CUDA_DSA"] = "1"  # Enable CUDA device-side assertions

		# Disable CUDA memory caching to reduce fragmentation
		if perf.get("pytorch_no_cuda_memory_caching", False):
			child_env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"  # Disable CUDA memory caching

		# xformers Triton backend configuration
		if perf.get("xformers_enable_triton", False):
			child_env["XFORMERS_ENABLE_TRITON"] = "1"  # Force Triton backend
		
		# On Windows, Triton is not supported, so we force disable it unless explicitly enabled
		if platform.system() == "Windows" or perf.get("xformers_force_disable_triton", False):
			child_env["XFORMERS_FORCE_DISABLE_TRITON"] = "1"  # Disable Triton backend

		# xformers performance and compatibility settings
		if perf.get("disable_fused_sequence_parallel", False):
			child_env["DISABLE_FUSED_SEQUENCE_PARALLEL"] = "1"  # Disable fused operations
		if perf.get("xformers_ignore_flash_version_check", False):
			child_env["XFORMERS_IGNORE_FLASH_VERSION_CHECK"] = "1"  # Skip version checks

		if child_env:
			info(f"Prepared {len(child_env)} environment variable(s) for the training process.")
	except Exception as ex:
		warning(f"Failed to configure environment variables: {ex}")

	success("Environment configuration loaded.")

	# --- Re-exec into venv Python to run system/entry.py, passing the root directory ---
	entry_py = os.path.join(ROOT, "system", "entry.py")
	if not os.path.isfile(entry_py):
		error(f"Entry script not found at {entry_py}.")
		return
	
	# Validate venv python exists before attempting exec
	if not os.path.isfile(venv_python):
		error(f"Virtual environment Python executable missing: {venv_python}.")
		return

	# Force flush all streams multiple times to ensure output is visible
	for _ in range(3):
		sys.stdout.flush()
		sys.stderr.flush()
		import time
		time.sleep(0.1)  # Small delay to ensure flush completes
	
	try:
		# Test that the venv python can at least run --version first
		test_result = subprocess.run([venv_python, "--version"], capture_output=True, text=True, timeout=5)
		if test_result.returncode != 0:
			error(f"Virtual environment Python failed --version sanity check: {test_result.stderr}")
			return
			
		sys.stdout.flush(); sys.stderr.flush()
		
		# Use subprocess.run instead of execve for Windows compatibility
		# Pass the child environment with CUDA allocator configuration
		if child_env:
			# Merge current environment with our custom settings
			full_env = {**os.environ, **child_env}
			result = subprocess.run([venv_python, entry_py, str(ROOT)], env=full_env)
		else:
			result = subprocess.run([venv_python, entry_py, str(ROOT)])
		sys.exit(result.returncode)
	except subprocess.TimeoutExpired:
		error("Re-execution timed out.")
		sys.exit(1)
	except OSError as e:
		error(f"Failed to launch training process: {e}")
		sys.exit(1)
	except Exception as e:
		error(f"Unexpected error while launching training process: {e}")
		sys.exit(1)


if __name__ == "__main__":
	main()

