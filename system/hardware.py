import gc
import os
import time
import psutil
import torch
import platform
from typing import Optional, Tuple, Dict, Any, List
from enum import Enum
from system.coordinator_settings import SETTINGS
from system.log import info, warning, error, success


class VRAMState(Enum):
    """VRAM state management for advanced memory optimization."""
    DISABLED = 0  # No vram present: no need to move models to vram
    NO_VRAM = 1  # Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5  # No dedicated vram: memory shared between CPU and GPU but models still need to be moved between both.


class CPUState(Enum):
    """CPU/GPU state management."""
    GPU = 0
    CPU = 1
    MPS = 2


# Global state
vram_state = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU
total_vram = 0
total_ram = 0
_TOTAL_MEMORY_LOGGED = False

# Device capability detection
try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        vram_state = VRAMState.SHARED
except:
    pass

# Calculate total memory
try:
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
    total_ram = psutil.virtual_memory().total / (1024**3)
    if not _TOTAL_MEMORY_LOGGED:
        info(f"Total Memory: {total_vram:.2f}GB VRAM, {total_ram:.2f}GB RAM")
        _TOTAL_MEMORY_LOGGED = True
except:
    pass

def get_gpu_info() -> Dict[str, Any]:
    info = {
        "has_cuda": torch.cuda.is_available(),
        "gpu_count": 0,
        "gpus": []
    }
    
    if info["has_cuda"]:
        info["gpu_count"] = torch.cuda.device_count()
        for i in range(info["gpu_count"]):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "name": props.name,
                "total_memory": props.total_memory / (1024**3), # GB
                "free_memory": torch.cuda.mem_get_info(i)[0] / (1024**3), # GB
                "major": props.major,
                "minor": props.minor
            })
    
    return info

def get_cpu_info() -> Dict[str, Any]:
    return {
        "cpu_count": psutil.cpu_count(logical=False),
        "logical_count": psutil.cpu_count(logical=True),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "total_ram": psutil.virtual_memory().total / (1024**3), # GB
        "available_ram": psutil.virtual_memory().available / (1024**3) # GB
    }

def get_system_info() -> Dict[str, Any]:
    return {
        "gpu": get_gpu_info(),
        "cpu": get_cpu_info()
    }

# Current loaded models tracking (advanced memory management)
current_loaded_models = []
current_inference_memory = 1024 * 1024 * 1024  # 1GB default


class LoadedModel:
    """Advanced model tracking for automatic memory management."""

    def __init__(self, model, memory_required=0):
        self.model = model
        self.memory_required = memory_required
        self.model_accelerated = False
        self.device = model.load_device if hasattr(model, 'load_device') else get_torch_device()

    def model_memory(self):
        """Get model memory size."""
        return module_size(self.model)

    def model_memory_required(self, device):
        """Get memory required for model on specific device."""
        return module_size(self.model, exclude_device=device)

    def model_load(self, model_gpu_memory_when_using_cpu_swap=-1):
        """Load model with automatic CPU swap if needed."""
        patch_model_to = None
        do_not_need_cpu_swap = model_gpu_memory_when_using_cpu_swap < 0

        if do_not_need_cpu_swap:
            patch_model_to = self.device

        # Move model to device with CPU swap if needed
        if not do_not_need_cpu_swap:
            real_async_memory = 0
            mem_counter = 0

            for module in self.model.modules():
                if hasattr(module, "parameters_manual_cast"):
                    # Store previous value and enable manual cast
                    prev_value = getattr(module, "parameters_manual_cast", None)
                    setattr(module, "prev_parameters_manual_cast", prev_value)
                    setattr(module, "parameters_manual_cast", True)

                    module_mem = module_size(module)
                    if mem_counter + module_mem < model_gpu_memory_when_using_cpu_swap:
                        module.to(self.device)
                        mem_counter += module_mem
                        info(f"Moved {type(module).__name__} to GPU ({module_mem / (1024**2):.2f}MB)")
                    else:
                        real_async_memory += module_mem
                        module.to(torch.device("cpu"))
                        info(f"Moved {type(module).__name__} to CPU ({module_mem / (1024**2):.2f}MB)")

            info(f"CPU Swap Stats: {real_async_memory / (1024**2):.2f}MB on CPU, {mem_counter / (1024**2):.2f}MB on GPU")
            self.model_accelerated = True
        else:
            self.model.to(self.device)

        return self.model

    def model_unload(self, avoid_model_moving=False):
        """Unload model and restore CPU swap settings."""
        if self.model_accelerated:
            for module in self.model.modules():
                if hasattr(module, "prev_parameters_manual_cast"):
                    module.parameters_manual_cast = module.prev_parameters_manual_cast
                    del module.prev_parameters_manual_cast
            self.model_accelerated = False

        if not avoid_model_moving:
            self.model.to(torch.device("cpu"))

    def __eq__(self, other):
        """Check if models are the same."""
        return self.model is other.model


def load_models_gpu(models, memory_required=0):
    """Advanced automatic model loading with memory management."""
    global current_loaded_models

    execution_start_time = time.perf_counter()
    extra_mem = max(current_inference_memory, memory_required)

    models_to_load = []
    models_already_loaded = []

    for model in models:
        loaded_model = LoadedModel(model, memory_required=memory_required)

        if loaded_model in current_loaded_models:
            # Move to front (most recently used)
            index = current_loaded_models.index(loaded_model)
            current_loaded_models.insert(0, current_loaded_models.pop(index))
            models_already_loaded.append(loaded_model)
        else:
            models_to_load.append(loaded_model)

    if len(models_to_load) == 0:
        # Clean up memory for existing models
        for loaded_model in models_already_loaded:
            if loaded_model.device != torch.device("cpu"):
                free_memory(extra_mem, loaded_model.device, models_already_loaded)
        return

    info(f"Loading {len(models_to_load)} model{'s' if len(models_to_load) > 1 else ''}...")

    # Calculate total memory required
    total_memory_required = {}
    for loaded_model in models_to_load:
        device = loaded_model.device
        if device != torch.device("cpu"):
            total_memory_required[device] = total_memory_required.get(device, 0) + loaded_model.model_memory_required(device)

    # Free memory as needed
    for device, mem_required in total_memory_required.items():
        free_memory(mem_required * 1.3 + extra_mem, device, models_already_loaded)

    # Load models with CPU swap if needed
    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = loaded_model.device

        model_gpu_memory_when_using_cpu_swap = -1

        if vram_state in [VRAMState.LOW_VRAM, VRAMState.NORMAL_VRAM]:
            model_memory = loaded_model.model_memory_required(torch_dev)
            current_free_mem = get_free_memory(torch_dev)
            inference_memory = current_inference_memory
            estimated_remaining_memory = current_free_mem - model_memory - inference_memory

            if estimated_remaining_memory < 0:
                model_gpu_memory_when_using_cpu_swap = compute_model_gpu_memory_when_using_cpu_swap(
                    current_free_mem, inference_memory
                )

        loaded_model.model_load(model_gpu_memory_when_using_cpu_swap)
        current_loaded_models.insert(0, loaded_model)

    moving_time = time.perf_counter() - execution_start_time
    if moving_time > 0.1:
        info(f"Model loading took {moving_time:.2f}s")


def free_memory(memory_required, device, keep_loaded=[]):
    """Free memory by unloading models."""
    global current_loaded_models

    unloaded_model = False
    for i in range(len(current_loaded_models) - 1, -1, -1):
        if get_free_memory(device) > memory_required:
            break

        shift_model = current_loaded_models[i]
        if shift_model.device == device and shift_model not in keep_loaded:
            unloaded_model = True
            m = current_loaded_models.pop(i)
            m.model_unload()
            del m

    if unloaded_model:
        soft_empty_cache()
    else:
        if vram_state != VRAMState.HIGH_VRAM:
            mem_free_total, mem_free_torch = get_free_memory(device), get_free_memory(device)
            if mem_free_torch > mem_free_total * 0.25:
                soft_empty_cache()


def compute_model_gpu_memory_when_using_cpu_swap(current_free_mem, inference_memory):
    """Compute optimal GPU memory allocation when using CPU swap."""
    maximum_memory_available = current_free_mem - inference_memory

    k_1GB = float(inference_memory / (1024 * 1024 * 1024))
    k_1GB = max(0.0, min(1.0, k_1GB))

    adaptive_safe_factor = 1.0 - 0.23 * k_1GB
    suggestion = maximum_memory_available * adaptive_safe_factor

    return int(max(0, suggestion))


def unload_all_models():
    """Unload all loaded models."""
    global current_loaded_models
    for loaded_model in current_loaded_models:
        loaded_model.model_unload()
    current_loaded_models.clear()
    soft_empty_cache()


def cleanup_models():
    """Clean up models with low reference count."""
    import sys
    to_delete = []
    for i in range(len(current_loaded_models)):
        if sys.getrefcount(current_loaded_models[i].model) <= 2:
            to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        x.model_unload()
        del x


def state_dict_size(sd, exclude_device=None):
    """Calculate state dict size with device exclusion."""
    module_mem = 0
    for k in sd:
        t = sd[k]
        if exclude_device is not None:
            if t.device == exclude_device:
                continue
        module_mem += t.nelement() * t.element_size()
    return module_mem


def should_use_fp16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    """Enhanced FP16 detection with advanced compatibility checks."""
    if device is not None:
        if hasattr(device, 'type') and device.type == 'cpu':
            return False

    if device is not None:
        if hasattr(device, 'type') and device.type == 'mps':
            return True

    if cpu_state == CPUState.MPS:
        return True

    if cpu_state == CPUState.CPU:
        return False

    try:
        props = torch.cuda.get_device_properties(device or "cuda")
        if props.major >= 8:
            return True

        if props.major < 6:
            return False

        # Check for problematic cards
        nvidia_10_series = ["1080", "1070", "titan x", "p3000", "p3200", "p4000", "p4200", "p5000", "p5200", "p6000", "1060", "1050", "p40", "p100", "p6", "p4"]
        nvidia_16_series = ["1660", "1650", "1630", "T500", "T550", "T600", "MX550", "MX450", "CMP 30HX", "T2000", "T1000", "T1200"]

        fp16_works = any(x in props.name.lower() for x in nvidia_10_series)

        if fp16_works or manual_cast:
            free_model_memory = (get_free_memory(device) * 0.9 - current_inference_memory)
            if (not prioritize_performance) or model_params * 4 > free_model_memory:
                return True

        if props.major < 7:
            return False

        # FP16 is broken on these cards
        if any(x in props.name for x in nvidia_16_series):
            return False

        return True
    except:
        return False


def should_use_bf16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    """Enhanced BF16 detection with advanced compatibility checks."""
    if device is not None:
        if hasattr(device, 'type') and device.type == 'cpu':
            return False

    if device is not None:
        if hasattr(device, 'type') and device.type == 'mps':
            return True

    if cpu_state == CPUState.MPS:
        return True

    if cpu_state == CPUState.CPU:
        return False

    try:
        if device is None:
            device = torch.device("cuda")

        props = torch.cuda.get_device_properties(device)
        if props.major >= 8:
            return True

        bf16_works = torch.cuda.is_bf16_supported()

        if bf16_works or manual_cast:
            free_model_memory = (get_free_memory(device) * 0.9 - current_inference_memory)
            if (not prioritize_performance) or model_params * 4 > free_model_memory:
                return True

        return False
    except:
        return False


def get_computation_dtype(inference_device, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
    """Get optimal computation dtype for device."""
    for candidate in supported_dtypes:
        if candidate == torch.float16:
            if should_use_fp16(inference_device, prioritize_performance=False):
                return candidate
        if candidate == torch.bfloat16:
            if should_use_bf16(inference_device):
                return candidate
    return torch.float32


def set_inference_memory_estimate(width: int, height: int, steps: int, batch_size: int = 1):
    """Update inference memory estimate."""
    global current_inference_memory
    current_inference_memory = estimate_inference_memory(width, height, steps, batch_size)


def get_vram_state():
    """Get current VRAM state."""
    return vram_state


def set_vram_state(state: VRAMState):
    """Set VRAM state."""
    global vram_state
    vram_state = state
    info(f"VRAM State set to: {vram_state.name}")


def get_cpu_state():
    """Get current CPU state."""
    return cpu_state


def is_device_cpu(device):
    """Check if device is CPU."""
    return hasattr(device, 'type') and device.type == 'cpu'


def is_device_mps(device):
    """Check if device is MPS."""
    return hasattr(device, 'type') and device.type == 'mps'


def is_device_cuda(device):
    """Check if device is CUDA."""
    return hasattr(device, 'type') and device.type == 'cuda'


def can_install_bnb():
    """Check if bitsandbytes can be installed."""
    try:
        import torch
        import torch.version
        if not torch.cuda.is_available():
            return False

        # Check CUDA version for bitsandbytes compatibility
        # bitsandbytes requires CUDA 11.7+
        try:
            # Try to get CUDA version string
            cuda_version_str = torch.version.cuda
            if cuda_version_str:
                cuda_version = tuple(int(x) for x in cuda_version_str.split('.'))
                return cuda_version >= (11, 7)
        except (AttributeError, ValueError):
            pass

        # Fallback: check device capability
        try:
            device_capability = torch.cuda.get_device_capability(0)
            # Ampere (8.x) and newer generally support bitsandbytes
            return device_capability[0] >= 8
        except:
            pass

        return False
    except:
        return False


# Keep existing functions for backward compatibility
def get_torch_device() -> torch.device:
    """Get the optimal PyTorch device based on availability and settings."""
    runtime_settings = SETTINGS.get("runtime", {})
    device_setting = runtime_settings.get("device", "auto")
    
    if device_setting == "cpu":
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        return torch.device(f"cuda:{current_device}")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    
    return torch.device("cpu")


def get_optimal_dtype(device: Optional[torch.device] = None, model_params: int = 0) -> torch.dtype:
    """Get the optimal dtype based on device capabilities and settings."""
    if device is None:
        device = get_torch_device()
    
    runtime_settings = SETTINGS.get("runtime", {})
    precision = runtime_settings.get("precision", "auto")
    
    # Explicit precision override
    if precision == "fp32":
        return torch.float32
    elif precision == "fp16":
        return torch.float16 if _supports_fp16(device, model_params) else torch.float32
    elif precision == "bf16":
        return torch.bfloat16 if _supports_bf16(device, model_params) else torch.float32
    
    # Auto-detection for optimal precision
    if device.type == "cpu":
        return torch.float32
    elif device.type == "mps":
        return torch.float16
    elif device.type == "cuda":
        # Prefer bf16 > fp16 > fp32 based on capability
        if _supports_bf16(device, model_params):
            return torch.bfloat16
        elif _supports_fp16(device, model_params):
            return torch.float16
        else:
            return torch.float32
    
    return torch.float32


def _supports_fp16(device: torch.device, model_params: int = 0) -> bool:
    """Check if device supports FP16 efficiently."""
    if device.type == "cpu":
        return False
    elif device.type == "mps":
        return True
    elif device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        # Tensor cores from compute capability 7.0+
        if props.major >= 8:  # Ampere and newer
            return True
        elif props.major == 7:  # Turing
            # Check for specific problematic cards (GTX 16xx series)
            problematic_cards = ["1660", "1650", "1630"]
            for card in problematic_cards:
                if card in props.name:
                    return False
            return True
        elif props.major == 6:  # Pascal (1080, etc.)
            # Only use FP16 if memory constrained
            free_memory = get_free_memory(device)
            required_memory = model_params * 4  # Assume FP32 size
            return required_memory > free_memory * 0.9
        else:
            return False
    
    return False


def _supports_bf16(device: torch.device, model_params: int = 0) -> bool:
    """Check if device supports BF16 efficiently."""
    if device.type == "cpu":
        return False  # BF16 on CPU is extremely slow
    elif device.type == "mps":
        return True
    elif device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            return False
        props = torch.cuda.get_device_properties(device)
        return props.major >= 8  # Ampere and newer
    
    return False


def get_free_memory(device: Optional[torch.device] = None) -> int:
    """Get free memory in bytes for the given device."""
    if device is None:
        device = get_torch_device()
    
    if device.type == "cpu" or device.type == "mps":
        return psutil.virtual_memory().available
    elif device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        return free
    else:
        # Fallback
        return psutil.virtual_memory().available


def get_total_memory(device: Optional[torch.device] = None) -> int:
    """Get total memory in bytes for the given device."""
    if device is None:
        device = get_torch_device()
    
    if device.type == "cpu" or device.type == "mps":
        return psutil.virtual_memory().total
    elif device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        return total
    else:
        return psutil.virtual_memory().total


def module_size(module: torch.nn.Module, exclude_device: Optional[torch.device] = None) -> int:
    """Calculate memory size of a PyTorch module in bytes."""
    module_mem = 0
    for param in module.parameters():
        if exclude_device is not None and param.device == exclude_device:
            continue
        module_mem += param.numel() * param.element_size()
    
    # Add buffers
    for buffer in module.buffers():
        if exclude_device is not None and buffer.device == exclude_device:
            continue
        module_mem += buffer.numel() * buffer.element_size()
    
    return module_mem


def estimate_inference_memory(width: int = 1024, height: int = 1024, steps: int = 20, 
                            batch_size: int = 1) -> int:
    """Estimate memory required for inference based on image parameters."""
    # Base memory estimation (very rough)
    pixels = width * height * batch_size
    
    # Memory scales roughly with: pixels, steps (for intermediate tensors), and model complexity
    base_memory = pixels * 16  # Rough estimate: 16 bytes per pixel for intermediate tensors
    step_memory = pixels * 8 * steps  # Memory for denoising steps
    latent_memory = pixels * 4  # Latent representation memory
    
    total = base_memory + step_memory + latent_memory
    
    # Add safety margin
    return int(total * 1.3)


def configure_cuda_allocator(allocator_type: str = "") -> bool:
    """Configure CUDA allocator programmatically when possible.

    Args:
        allocator_type: Type of allocator ("cudaMallocAsync", "native", "malloc", "")

    Returns:
        bool: True if configuration was applied, False otherwise
    """
    if not torch.cuda.is_available():
        return False

    if not allocator_type:
        return False

    try:
        # Note: PyTorch doesn't provide a stable public API for changing allocator
        # after initialization. The allocator is set via PYTORCH_CUDA_ALLOC_CONF
        # environment variable before torch is imported.
        #
        # This function serves as documentation and a placeholder for future
        # allocator configuration methods.

        info(f"CUDA allocator configured to: {allocator_type}")
        return True

    except Exception as e:
        warning(f"Failed to configure CUDA allocator: {str(e)}")
        return False


def optimize_cuda_performance():
    """Apply CUDA-specific performance optimizations."""
    if not torch.cuda.is_available():
        return

    performance = SETTINGS.get("generation", {})

    # Note: CUDA allocator configuration via PYTORCH_CUDA_ALLOC_CONF is handled
    # in run.py due to environment variable restrictions. Here we handle other optimizations.

    # Enable TF32 for faster training/inference on Ampere+
    if performance.get("enable_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        info("TF32 enabled for faster performance")

    # Set matmul precision
    precision = performance.get("matmul_precision", "high")
    if precision == "high":
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        info("Matmul precision set to HIGH")

    # Enable cuDNN benchmark for consistent input sizes
    if performance.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True
        info("cuDNN benchmark enabled")

    # Configure SDPA backends
    sdpa = performance.get("sdpa", {})
    if sdpa.get("enable_flash_sdp", True):
        torch.backends.cuda.enable_flash_sdp(True)
    if sdpa.get("enable_mem_efficient_sdp", True):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if not sdpa.get("enable_math_sdp", True):
        torch.backends.cuda.enable_math_sdp(False)
    info("SDPA optimizations applied")
def soft_empty_cache(force: bool = False):
    """Intelligently clear GPU cache."""
    device = get_torch_device()
    
    if device.type == "cuda":
        if force:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        else:
            # Only clear if we have significant cached memory
            free, total = torch.cuda.mem_get_info(device)
            cache_threshold = (SETTINGS or {}).get("memory", {}).get("cache_clear_threshold_mb", 512)
            if (total - free) > cache_threshold * 1024 * 1024:  # Configurable threshold
                torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    
    # CPU garbage collection
    if force:
        gc.collect()


def apply_unified_memory_optimizations(pipeline: Any, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    """Apply comprehensive memory optimizations to any Diffusers pipeline.

    Returns a dict of applied optimizations for tracking.
    """
    applied = {}
    perf = SETTINGS.get("generation", {})

    # VAE optimizations
    if bool(perf.get("vae_slicing", True)) and hasattr(pipeline, "enable_vae_slicing"):
        try:
            pipeline.enable_vae_slicing()
            applied["vae_slicing"] = True
        except Exception:
            applied["vae_slicing"] = False

    if bool(perf.get("vae_tiling", False)) and hasattr(pipeline, "enable_vae_tiling"):
        try:
            pipeline.enable_vae_tiling()
            applied["vae_tiling"] = True
        except Exception:
            applied["vae_tiling"] = False

    # Attention slicing for memory constrained systems
    if bool(perf.get("attention_slicing", False)) and hasattr(pipeline, "enable_attention_slicing"):
        try:
            pipeline.enable_attention_slicing()
            applied["attention_slicing"] = True
        except Exception:
            applied["attention_slicing"] = False

    # Disable progress bars
    try:
        if hasattr(pipeline, "set_progress_bar_config"):
            pipeline.set_progress_bar_config(disable=True)
        applied["progress_bar_disabled"] = True
    except Exception:
        applied["progress_bar_disabled"] = False

    # Apply channels_last memory format for potential conv speedup
    if bool(perf.get("channels_last", False)) and device.type == "cuda":
        try:
            for component_name in ["unet", "vae", "text_encoder", "text_encoder_2", "transformer"]:
                component = getattr(pipeline, component_name, None)
                if component is not None:
                    try:
                        component.to(memory_format=torch.channels_last)
                    except Exception:
                        pass
            applied["channels_last"] = True
        except Exception:
            applied["channels_last"] = False

    return applied


class AdaptiveMemoryManager:
    """Advanced adaptive memory management for dynamic GPU/CPU offloading."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.cpu_device = torch.device("cpu")
        self.loaded_models: List[Dict[str, Any]] = []
        self.inference_memory_estimate = 1024 * 1024 * 1024  # 1GB default
        self._last_width = 1024
        self._last_height = 1024
        self._last_steps = 20
        self._last_batch_size = 1
    
    def estimate_model_memory(self, model: torch.nn.Module) -> int:
        """Estimate memory required for a model."""
        return module_size(model)
    
    def get_available_memory(self) -> Tuple[int, int]:
        """Get available memory for both GPU and CPU."""
        gpu_free = get_free_memory(self.device)
        cpu_free = get_free_memory(self.cpu_device)
        return gpu_free, cpu_free
    
    def should_use_cpu_swap(self, model_memory: int) -> Tuple[bool, int]:
        """Determine if we should use CPU swap and how much GPU memory to allocate."""
        gpu_free, cpu_free = self.get_available_memory()

        # The raw heuristic estimate is extremely optimistic for SDXL/large resolutions.
        # Use a conservative floor based on resolution, plus a fraction of model size.
        pixels = int(self._last_width) * int(self._last_height) * max(1, int(self._last_batch_size))
        if pixels >= 1024 * 1024:
            floor_bytes = 3 * 1024 * 1024 * 1024
        elif pixels >= 768 * 768:
            floor_bytes = 2 * 1024 * 1024 * 1024
        else:
            floor_bytes = int(1.25 * 1024 * 1024 * 1024)

        effective_inference = max(self.inference_memory_estimate, floor_bytes, int(model_memory * 0.6))
        total_required = model_memory + effective_inference

        info(
            "GPU Free: %.2fMB | Model Needs: %.2fMB | Inference Needs: %.2fMB (effective %.2fMB)"
            % (
                gpu_free / (1024**2),
                model_memory / (1024**2),
                self.inference_memory_estimate / (1024**2),
                effective_inference / (1024**2),
            )
        )
        
        if total_required <= gpu_free:
            # Enough GPU memory for everything
            return False, -1
        
        if model_memory <= gpu_free - effective_inference:
            # Model fits in GPU with inference memory
            return False, -1
        
        # Need CPU swap - calculate how much to keep on GPU
        available_for_model = max(0, gpu_free - effective_inference)
        
        # Use adaptive factor: keep more on GPU for larger available memory
        k_factor = float(self.inference_memory_estimate / (1024**3))  # Convert to GB
        k_factor = max(0.0, min(1.0, k_factor))
        adaptive_factor = 1.0 - 0.23 * k_factor
        
        gpu_allocation = int(available_for_model * adaptive_factor)
        
        warning(f"Insufficient GPU memory. Allocating {gpu_allocation / (1024**2):.2f}MB to GPU, rest to CPU.")
        
        return True, gpu_allocation
    
    def apply_cpu_swap(self, pipeline: Any, gpu_memory_limit: int):
        """Apply CPU swap using diffusers offloading mechanisms."""
        # If we have very low VRAM (e.g. < 2GB allocated for model), use sequential offload
        # This is slower but safer for low VRAM
        if gpu_memory_limit < 2 * 1024 * 1024 * 1024:
            if hasattr(pipeline, "enable_sequential_cpu_offload"):
                info("Low VRAM detected: Enabling sequential CPU offload")
                pipeline.enable_sequential_cpu_offload(device=self.device)
                return

        # Otherwise use model offload (faster, moves whole models)
        if hasattr(pipeline, "enable_model_cpu_offload"):
            info("Enabling model CPU offload")
            pipeline.enable_model_cpu_offload(device=self.device)
            return
            
        # Fallback if neither is available (shouldn't happen with modern diffusers)
        warning("Could not enable CPU offload. Moving pipeline to CPU.")
        pipeline.to(self.cpu_device)
    
    def set_inference_memory_estimate(self, width: int, height: int, steps: int, batch_size: int = 1):
        """Update inference memory estimate based on generation parameters."""
        self._last_width = int(width)
        self._last_height = int(height)
        self._last_steps = int(steps)
        self._last_batch_size = int(batch_size)
        self.inference_memory_estimate = estimate_inference_memory(width, height, steps, batch_size)


class MemoryManager:
    """Main memory management class with advanced optimizations."""
    
    def __init__(self):
        self.device = get_torch_device()
        self.dtype = get_optimal_dtype(self.device)
        self.adaptive_manager = AdaptiveMemoryManager(self.device)
        
        # Initialize CUDA optimizations
        optimize_cuda_performance()
        
        info(f"Memory Manager initialized on {self.device} ({self.dtype})")
    
    def prepare_pipeline_memory(self, pipeline, **generation_kwargs) -> torch.nn.Module:
        """Prepare pipeline for memory-efficient execution."""
        # Extract generation parameters for memory estimation
        width = generation_kwargs.get("width", 1024)
        height = generation_kwargs.get("height", 1024) 
        num_inference_steps = generation_kwargs.get("num_inference_steps", 20)
        
        self.adaptive_manager.set_inference_memory_estimate(width, height, num_inference_steps)
        
        # Get UNet for memory analysis
        unet = getattr(pipeline, "unet", None)
        if unet is None:
            # Try transformer for Flux/SD3
            unet = getattr(pipeline, "transformer", None)
            
        if unet is None:
            warning("Could not find UNet/Transformer for adaptive memory management. Skipping optimizations.")
            return pipeline
        
        model_memory = self.adaptive_manager.estimate_model_memory(unet)
        should_swap, gpu_allocation = self.adaptive_manager.should_use_cpu_swap(model_memory)
        
        if should_swap:
            # We pass the whole pipeline to apply_cpu_swap now
            self.adaptive_manager.apply_cpu_swap(pipeline, gpu_allocation)
        else:
            # If no swap needed, ensure everything is on GPU
            pipeline.to(self.device)
        
        return pipeline
    
    def cleanup_after_generation(self):
        """Clean up memory after generation."""
        soft_empty_cache()


def suggest_optimizations(vram_gb: float) -> Dict[str, Any]:
    """
    Suggest optimizations based on available VRAM.
    """
    suggestions = {
        "optimizations": [],
        "warnings": [],
        "settings": {
            "low_vram": False,
            "cpu_offload": False,
            "attention_slicing": False,
            "vae_slicing": False
        }
    }
    
    if vram_gb == 0:
        suggestions["warnings"].append("No GPU detected or CPU mode forced.")
        suggestions["settings"]["low_vram"] = True
        return suggestions

    # VRAM thresholds (in GB)
    LOW_VRAM_THRESHOLD = 6.0
    MEDIUM_VRAM_THRESHOLD = 10.0
    
    if vram_gb < LOW_VRAM_THRESHOLD:
        suggestions["optimizations"].append("Enable Low VRAM mode")
        suggestions["optimizations"].append("Use CPU offloading")
        suggestions["optimizations"].append("Enable Attention Slicing")
        suggestions["optimizations"].append("Enable VAE Slicing")
        suggestions["settings"]["low_vram"] = True
        suggestions["settings"]["cpu_offload"] = True
        suggestions["settings"]["attention_slicing"] = True
        suggestions["settings"]["vae_slicing"] = True
        
    elif vram_gb < MEDIUM_VRAM_THRESHOLD:
        suggestions["optimizations"].append("Enable Balanced mode")
        suggestions["optimizations"].append("Use CPU offloading for larger models")
        suggestions["settings"]["low_vram"] = False
        suggestions["settings"]["cpu_offload"] = True
        
    else:
        suggestions["optimizations"].append("High Performance mode available")
        suggestions["settings"]["low_vram"] = False
        suggestions["settings"]["cpu_offload"] = False
        
    if platform.system() == "Linux":
        suggestions["optimizations"].append("Linux detected: Ensure CUDA drivers are up to date")
        
    return suggestions
