import threading
import time
import traceback
from typing import Optional, Dict, Any
from .schema import TrainingConfig
from .engine_sdxl import train_sdxl
from .engine_sd15 import train_sd15
from .engine_flux import train_flux
from .engine_sd3 import train_sd3
from .engine_utils import flush

class TrainingManager:
    def __init__(self):
        self.is_training = False
        self.status = "idle"
        self.progress = 0.0
        self.current_step = 0
        self.total_steps = 0
        self.loss = 0.0
        self.epoch = 0
        self.logs = []
        self.error = None
        self._thread = None
        self._stop_event = threading.Event()

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_training": self.is_training,
            "status": self.status,
            "progress": self.progress,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "loss": self.loss,
            "epoch": self.epoch,
            "logs": self.logs[-50:], # Return last 50 logs
            "error": self.error
        }

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        print(f"[TRAIN] {message}")

    def start_training(self, config: TrainingConfig):
        if self.is_training:
            raise RuntimeError("Training is already in progress")
        
        self.is_training = True
        self.status = "starting"
        self.progress = 0.0
        self.current_step = 0
        self.logs = []
        self.error = None
        self._stop_event.clear()
        
        self._thread = threading.Thread(target=self._training_loop, args=(config,))
        self._thread.start()

    def stop_training(self):
        if self.is_training:
            self.log("Stopping training...")
            self._stop_event.set()

    def _training_loop(self, config: TrainingConfig):
        try:
            from system.hardware import get_system_info
            sys_info = get_system_info()
            if sys_info["gpu"]["has_cuda"]:
                gpu = sys_info["gpu"]["gpus"][0]
                self.log(f"Detected GPU: {gpu['name']} with {gpu['total_memory']:.1f}GB VRAM")
            else:
                self.log("No CUDA GPU detected, training on CPU (will be very slow!)")

            self.log(f"Starting training for {config.output_name} ({config.model_type})")
            
            def progress_callback(step, total, loss, epoch, msg=None, phase: Optional[str] = None):
                if self._stop_event.is_set():
                    raise InterruptedError("Training stopped by user")
                
                self.current_step = step
                self.total_steps = total
                self.loss = loss
                self.epoch = epoch
                self.progress = (step / total) * 100 if total > 0 else 0
                if phase:
                    self.status = phase
                else:
                    self.status = "training"
                if msg:
                    self.log(msg)
            
            # Dispatch to correct engine
            if config.model_type == "sdxl":
                train_sdxl(config, progress_callback)
            elif config.model_type == "sd_legacy":
                train_sd15(config, progress_callback)
            elif config.model_type in ["flux1", "flux2"]:
                train_flux(config, progress_callback)
            elif config.model_type in ["sd3", "sd3.5"]:
                train_sd3(config, progress_callback)
            else:
                raise ValueError(f"Unknown model type: {config.model_type}")
            
            self.status = "completed"
            self.progress = 100.0
            self.log("Training completed successfully!")
            
        except InterruptedError:
            self.status = "stopped"
            self.log("Training stopped.")
        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            self.log(f"Training failed: {e}")
            traceback.print_exc()
        finally:
            self.is_training = False
            try:
                self.log("Unloading models and clearing VRAM...")
                flush()
                self.log("Memory cleared.")
            except Exception as e:
                print(f"Error clearing memory: {e}")
