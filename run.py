from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
import json
import re
import platform
import shutil

# NOTE: run.py is a pure bootstrapper. It must not import internal project modules.

# Define project root as the directory containing this script
ROOT = Path(__file__).resolve().parent

def info(msg: str) -> None:
	print(f"[INFO] {msg}")


def warning(msg: str) -> None:
	print(f"[WARNING] {msg}", file=sys.stderr)


def error(msg: str) -> None:
	print(f"[ERROR] {msg}", file=sys.stderr)


def success(msg: str) -> None:
	print(f"[SUCCESS] {msg}")


class ProgressBar:
	"""Clean single-line progress bar that works on all terminals including Windows."""
	
	def __init__(self, total: int | None, desc: str, unit: str = "it", width: int = 20) -> None:
		self.total = total if (isinstance(total, int) and total > 0) else None
		self.desc = desc
		self.unit = unit
		self.width = width
		self.count = 0
		self.current_item = ""
		self._active = True
		self._last_len = 0
	
	def _render(self) -> None:
		"""Render the progress bar on a single line."""
		if not self._active:
			return
		
		# Get terminal width to maximize usage
		term_cols = shutil.get_terminal_size((80, 20)).columns
		
		if self.total is None:
			# Indeterminate
			line = f"{self.desc}: {self.count} {self.unit}"
			if self.current_item:
				line += f" - {self.current_item}"
		else:
			# Determinate with block style bar
			pct = self.count / self.total if self.total > 0 else 0
			pct = min(1.0, max(0.0, pct))
			
			# Create the bar using block characters: █ and ░
			filled = int(pct * self.width)
			filled = min(filled, self.width)
			bar_content = "█" * filled + "░" * (self.width - filled)
			pct_int = int(pct * 100)
			
			# Format: "Installing packages [████████          ] 50% | 22/44 pkgs | scikit-learn"
			line = f"{self.desc} [{bar_content}] {pct_int}% | {self.count}/{self.total} {self.unit}"
			
			# Add item if it fits
			if self.current_item:
				# Calculate remaining space (-3 for safety)
				remaining = term_cols - len(line) - 3
				if remaining > 0:
					item_str = f" | {self.current_item}"
					# Truncate if necessary
					if len(item_str) > remaining:
						item_str = item_str[:remaining].rstrip()
					line += item_str

		# Pad with spaces to clear previous content
		if len(line) < self._last_len:
			line += " " * (self._last_len - len(line))
		
		# Final safety truncate to avoid wrapping
		if len(line) > term_cols:
			line = line[:term_cols-1]
			
		sys.stdout.write(f"\r{line}")
		sys.stdout.flush()
		self._last_len = len(line)
	
	def set_item(self, name: str) -> None:
		"""Set the current item being processed."""
		self.current_item = name
		self._render()
	
	def update(self, n: int = 1) -> None:
		try:
			n_int = int(n)
		except Exception:
			n_int = 1
		if n_int <= 0:
			n_int = 1
		self.count += n_int
		self.current_item = ""
		self._render()
	
	def close(self, final_message: str = "") -> None:
		"""Close the progress bar with an optional final message."""
		if not self._active:
			return
		self._active = False
		# Clear line
		sys.stdout.write(f"\r{' ' * self._last_len}\r")
		if final_message:
			sys.stdout.write(f"{final_message}\n")
		sys.stdout.flush()


def main() -> None:
	def _strip_jsonc(text: str) -> str:
		# Minimal JSONC stripper that preserves strings.
		out: list[str] = []
		i = 0
		in_str = False
		str_quote = ""
		escape = False
		while i < len(text):
			ch = text[i]
			nxt = text[i + 1] if i + 1 < len(text) else ""
			if in_str:
				out.append(ch)
				if escape:
					escape = False
				elif ch == "\\":
					escape = True
				elif ch == str_quote:
					in_str = False
				i += 1
				continue

			if ch in ("\"", "'"):
				in_str = True
				str_quote = ch
				out.append(ch)
				i += 1
				continue

			# line comment //...
			if ch == "/" and nxt == "/":
				i += 2
				while i < len(text) and text[i] not in "\r\n":
					i += 1
				continue
			# block comment /* ... */
			if ch == "/" and nxt == "*":
				i += 2
				while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
					i += 1
				i += 2
				continue

			out.append(ch)
			i += 1
		return "".join(out)

	def _load_packages_config() -> dict[str, Any]:
		cfg_path = ROOT / "config" / "packages.jsonc"
		if not cfg_path.is_file():
			raise FileNotFoundError(f"Missing packages config: {cfg_path}")
		raw = cfg_path.read_text(encoding="utf-8")
		clean = _strip_jsonc(raw)
		data = json.loads(clean)
		if not isinstance(data, dict):
			raise ValueError("packages.jsonc must contain a JSON object")
		return data

	def _run_quiet(cmd: list[str], label: str, keep_last_lines: int = 200, show_spinner: bool = True) -> tuple[int, str]:
		"""Run a command with an optional single-line spinner; capture last N lines in-memory.

		No temp files. Keeps console clean. On failure, caller can print captured output.
		If show_spinner is False, runs completely silently (for use with progress bars).
		"""
		import time
		import threading
		from collections import deque

		buf: deque[str] = deque(maxlen=max(50, int(keep_last_lines)))
		lock = threading.Lock()

		proc = subprocess.Popen(
			cmd,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			text=True,
			encoding="utf-8",
			errors="replace",
			bufsize=1,
			shell=False,
		)

		def _reader() -> None:
			try:
				assert proc.stdout is not None
				for line in proc.stdout:
					with lock:
						buf.append(line.rstrip("\n"))
			except Exception:
				pass

		t = threading.Thread(target=_reader, daemon=True)
		t.start()

		start = time.time()
		pb = None
		if show_spinner:
			pb = ProgressBar(total=None, desc=f"[RUN] {label}", unit="s")

		while True:
			ret = proc.poll()
			if pb:
				elapsed = int(time.time() - start)
				pb.count = elapsed
				pb._render()
			
			if ret is not None:
				break
			time.sleep(0.15)

		if pb:
			pb.close()

		try:
			proc.wait(timeout=1)
		except Exception:
			pass
		try:
			t.join(timeout=1)
		except Exception:
			pass

		with lock:
			out = "\n".join(buf).strip()
		return int(ret), out
	def detect_cuda_version():
		# Try to detect CUDA version from nvcc or nvidia-smi
		cuda_version = None
		# Try nvcc
		nvcc = shutil.which("nvcc")
		if nvcc:
			try:
				out = subprocess.check_output([nvcc, "--version"], encoding="utf-8", errors="ignore")
				# info(f"nvcc --version output:\n{out}") # Reduce noise
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
					# info(f"nvidia-smi output:\n{out}") # Reduce noise
					match = re.search(r"CUDA Version: (\d+\.\d+)", out)
					if match:
						cuda_version = match.group(1)
				except Exception as e:
					warning(f"Failed to query nvidia-smi: {e}")
					pass
		
		return cuda_version

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
		print("Creating virtual environment...", end=" ", flush=True)
		result = subprocess.run([sys.executable, "-m", "venv", venv_dir], capture_output=True, text=True)
		if result.returncode == 0:
			print("OK")
		else:
			print("FAILED")
			error(result.stderr or result.stdout)
			raise RuntimeError("Virtualenv creation failed")
		venv_created = True

	# --- PyTorch/xFormers auto-install if venv was just created (config-driven) ---
	if venv_created:
		# Upgrade pip silently
		subprocess.run([venv_python, "-m", "pip", "install", "-q", "--upgrade", "pip"], capture_output=True, text=True)

	# Ensure environment packages declared in PACKAGES are present in the venv.
	def ensure_environment_packages(venv_path: str) -> None:
		"""Ensure packages from config/packages.jsonc are installed into venv.

		IMPORTANT: run.py is a bootstrapper and must not import/use internal modules.
		"""
		cfg = _load_packages_config()
		env_pkgs = cfg.get("environment_packages") or []
		pytorch_cuda_version = cfg.get("pytorch_cuda_version")
		if not isinstance(env_pkgs, list):
			env_pkgs = []
		# normalize to list[str]
		env_pkgs = [p for p in env_pkgs if isinstance(p, str)]
		if isinstance(pytorch_cuda_version, str):
			pytorch_cuda_version = pytorch_cuda_version.strip()
		else:
			pytorch_cuda_version = None

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
			"cu130": "https://download.pytorch.org/whl/cu130",
		}

		def _parse_ver_tuple(ver: str) -> tuple[int, int] | None:
			m = re.match(r"^\s*(\d+)\.(\d+)", str(ver))
			if not m:
				return None
			try:
				return (int(m.group(1)), int(m.group(2)))
			except Exception:
				return None

		def _cuda_tag_tuple(tag: str) -> tuple[int, int] | None:
			m = re.match(r"^cu(\d+)$", str(tag).strip())
			if not m:
				return None
			digits = m.group(1)
			if len(digits) < 3:
				return None
			try:
				major = int(digits[:2])
				minor = int(digits[2:])
				return (major, minor)
			except Exception:
				return None

		def _pick_supported_cuda_tag(detected: str | None) -> str:
			supported = []
			for tag in cuda_index_map.keys():
				tv = _cuda_tag_tuple(tag)
				if tv is not None:
					supported.append((tv, tag))
			supported.sort()
			if not supported:
				return "cu121"

			if not detected:
				return supported[0][1]
			dv = _parse_ver_tuple(detected)
			if dv is None:
				return supported[0][1]

			best = None
			for tv, tag in supported:
				if tv <= dv:
					best = tag
				else:
					break
			return best or supported[0][1]

		# Handle "auto" CUDA version
		if pytorch_cuda_version == "auto":
			detected_ver = detect_cuda_version()
			chosen = _pick_supported_cuda_tag(detected_ver)
			if not detected_ver:
				warning(f"Could not detect CUDA, defaulting to {chosen}")
			pytorch_cuda_version = chosen

		# Handle triton-windows
		if platform.system() == "Windows":
			# Ensure triton-windows is in the list if we are on Windows
			has_triton = any("triton-windows" in pkg for pkg in env_pkgs)
			if not has_triton:
				env_pkgs.append("triton-windows")
		else:
			# Remove triton-windows if we are NOT on Windows
			env_pkgs = [pkg for pkg in env_pkgs if "triton-windows" not in pkg]

		if not env_pkgs:
			return

		# Use venv python to run pip commands (python -m pip ...)
		sys_platform = platform.system().lower()
		if sys_platform == "windows":
			venv_python_local = os.path.join(venv_path, "Scripts", "python.exe")
		else:
			venv_python_local = os.path.join(venv_path, "bin", "python")

		# Probe installed packages inside venv
		installed_map: dict[str, str] = {}
		try:
			out = subprocess.check_output([venv_python_local, "-m", "pip", "list", "--format=json"], encoding="utf-8", stderr=subprocess.DEVNULL)
			data = json.loads(out or "[]")
			for entry in data:
				name = (entry.get("name") or "").lower().replace("-", "_")
				ver = entry.get("version") or ""
				installed_map[name] = ver
		except Exception:
			installed_map = {}

		# If user pinned a CUDA build, ensure torch stack matches it even if not listed in environment_packages
		# CRITICAL: Check if installed torch is CPU-only (+cpu suffix) and force reinstall if user wants CUDA
		if pytorch_cuda_version:
			torch_index = f"https://download.pytorch.org/whl/{pytorch_cuda_version}"
			extra_index = "https://pypi.org/simple"
			torch_stack = ["torch", "torchvision", "torchaudio"]
			
			# Check if torch is installed and whether it's a CPU build
			need_torch_install = False
			torch_ver = installed_map.get("torch", "")
			if not torch_ver:
				need_torch_install = True
			elif "+cpu" in torch_ver:
				need_torch_install = True
			elif f"+{pytorch_cuda_version}" not in torch_ver and "+cu" not in torch_ver:
				try:
					cuda_check = subprocess.run(
						[venv_python_local, "-c", "import torch; print('1' if torch.cuda.is_available() else '0')"],
						capture_output=True, text=True, timeout=10
					).stdout.strip()
					if cuda_check != "1":
						need_torch_install = True
				except Exception:
					need_torch_install = True
			
			if need_torch_install:
				print(f"Installing PyTorch ({pytorch_cuda_version})...")
				cmd = [
					venv_python_local,
					"-m",
					"pip",
					"install",
					"-q",
					"--disable-pip-version-check",
					"--no-input",
					"--progress-bar",
					"off",
					"--force-reinstall",
					*torch_stack,
					"--index-url",
					torch_index,
					"--extra-index-url",
					extra_index,
				]
				retcode, out = _run_quiet(cmd, "torch", show_spinner=True)
				if retcode == 0:
					print("OK")
					# Refresh installed map after torch install
					try:
						out = subprocess.check_output([venv_python_local, "-m", "pip", "list", "--format=json"], encoding="utf-8", stderr=subprocess.DEVNULL)
						data = json.loads(out or "[]")
						for entry in data:
							name = (entry.get("name") or "").lower().replace("-", "_")
							installed_map[name] = entry.get("version") or ""
					except Exception: pass
				else:
					print("FAILED")
					if out:
						error(out.strip())

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
					xformers_index = cuda_index_map.get(pytorch_cuda_version, cuda_index_map.get("cu121", "https://download.pytorch.org/whl/cu121"))
					xformers_spec = f"{spec} --index-url {xformers_index} --extra-index-url https://pypi.org/simple"
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
			return

		# Use a clean progress bar - NO other prints during installation!
		desc_text = "Installing packages"
		pbar = ProgressBar(total=len(to_install), desc=desc_text, unit="pkgs")
		failed_packages: list[tuple[str, int, str]] = []
		
		try:
			for spec, spec_args in to_install:
				# Show current package in progress bar
				base_name = _spec_base_name(spec) if isinstance(spec, str) else str(spec)
				pbar.set_item(base_name)
				
				cmd = [
					venv_python_local,
					"-m",
					"pip",
					"install",
					"-q",
					"--disable-pip-version-check",
					"--no-input",
					"--progress-bar",
					"off",
					*spec_args,
				]
				retcode, out = _run_quiet(cmd, f"pip install {base_name}", show_spinner=False)
				
				if retcode != 0:
					failed_packages.append((spec, retcode, out))
					# Print error above the bar
					pbar._last_len = 0  # Clear tracking so next render doesn't leave artifacts
					sys.stdout.write(f"\r{' ' * pbar._last_len}\r")
					error(f"{base_name}: {out.strip().split(chr(10))[-1][:100]}")
				
				pbar.update(1)
		finally:
			# Close progress bar with appropriate message
			if failed_packages:
				pbar.close(f"[WARNING] Installed {len(to_install) - len(failed_packages)}/{len(to_install)} packages ({len(failed_packages)} failed)")
			else:
				pbar.close(f"[SUCCESS] All {len(to_install)} packages installed successfully")


	# Always ensure environment packages after venv is created and after PyTorch install (if any)
	ensure_environment_packages(venv_dir)

	# --- Create folder structure from config ---
	def ensure_folder_structure() -> None:
		"""Create folders defined in folder_structure from packages.jsonc."""
		try:
			cfg = _load_packages_config()
			folders = cfg.get("folder_structure", [])
			if not isinstance(folders, list):
				return
			
			created_count = 0
			for folder in folders:
				if not isinstance(folder, str):
					continue
				# Clean up the folder path (remove trailing slashes for consistency)
				folder = folder.rstrip("/\\")
				if not folder:
					continue
				
				folder_path = ROOT / folder
				if not folder_path.exists():
					try:
						folder_path.mkdir(parents=True, exist_ok=True)
						created_count += 1
					except Exception as e:
						warning(f"Failed to create folder '{folder}': {e}")
			
			if created_count > 0:
				success(f"Created {created_count} folder(s) from folder_structure config")
		except Exception as e:
			warning(f"Failed to process folder_structure: {e}")
	
	ensure_folder_structure()

	# IMPORTANT: Per requirement, run.py must depend only on packages.jsonc.
	# That means we do NOT read other configs here and we do NOT set extra runtime env vars.
	child_env: dict[str, str] = {}

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

