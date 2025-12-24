from system.diffusion.manager import ModelManager
from system.dataset.manager import DatasetManager
from system.training.manager import TrainingManager
from pathlib import Path

# Global runtime context
MODEL_MANAGER = ModelManager()
# Initialized with current dir, but entry.py updates it
DATASET_MANAGER = DatasetManager(Path(".")) 
TRAINING_MANAGER = TrainingManager()
