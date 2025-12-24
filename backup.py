import os
import zipfile
import time
from pathlib import Path

from system.log import info, warning, error, success


def create_backup():
    project_root = Path(__file__).parent
    parent_folder_name = project_root.name.lower()
    
    # Ask user for backup type
    info("Select backup type:")
    info("1. Working directory backup")
    info("2. Checkpoint backup (default)")
    info("3. Custom backup with comment")
    info("")
    
    user_input = input("Enter your choice (1-3) or press Enter for default: ").strip()
    
    if user_input == "1":
        backup_type = "working"
        comment = ""
    elif user_input == "3":
        backup_type = "custom"
        comment = input("Enter backup comment: ").strip()
        # Make comment filename-safe
        comment = comment.replace(" ", "_").replace("/", "_").replace("\\", "_")[:50]  # Limit length
        if not comment:
            comment = "no_comment"
    else:
        backup_type = "checkpoint"  # Default for empty input or "2"
        comment = ""
    
    info(f"Creating {backup_type} backup...")
    
    # Define backup locations with project-specific subdirectories
    backup_base_locations = [Path("b:/backup"), Path("d:/backup"), Path("c:/AI/DEV/.backup")]
    backup_locations = [base_dir / parent_folder_name for base_dir in backup_base_locations]
    
    # Create backup directories if they don't exist
    for backup_dir in backup_locations:
        backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    if comment:
        backup_filename = f"{parent_folder_name}_{timestamp}_{backup_type}_{comment}.zip"
    else:
        backup_filename = f"{parent_folder_name}_{timestamp}_{backup_type}.zip"
    
    excluded_dirs = {"venv", "projects", ".backup", "__pycache__", ".git", "inspiration","guides_comfyui","guides_diffusers","guides_forge","old_system","cache","models","logs","temp","project"}
    excluded_files = {".gitignore", "*.pyc", "*.pyo", "*.pyd"}

    # Create backup at each location
    for backup_dir in backup_locations:
        backup_path = backup_dir / backup_filename
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add comment file if provided
            if comment:
                original_comment = comment.replace("_", " ")  # Restore spaces for readability
                zipf.writestr("backup_comment.txt", f"Backup Comment: {original_comment}\nCreated: {time.ctime(timestamp)}\nType: {backup_type}")
            
            for root, dirs, files in os.walk(project_root):
                root_path = Path(root)
                relative_root = root_path.relative_to(project_root)
                
                if any(excluded in relative_root.parts for excluded in excluded_dirs):
                    continue
                
                dirs[:] = [d for d in dirs if d not in excluded_dirs]
                
                for file in files:
                    file_path = root_path / file
                    
                    if file.endswith(('.pyc', '.pyo', '.pyd')):
                        continue
                    
                    relative_file_path = file_path.relative_to(project_root)
                    zipf.write(file_path, relative_file_path)
        
        success(f"Backup created successfully: {backup_path}")
        success(f"Backup size: {backup_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    create_backup()
