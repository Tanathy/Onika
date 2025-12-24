import sys
from pathlib import Path
import platform

# Add project root to sys.path to allow imports from 'system'
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Dummy triton for Windows to avoid ModuleNotFoundError
if platform.system() == "Windows":
    try:
        import triton
    except ImportError:
        from types import ModuleType
        t = ModuleType("triton")
        sys.modules["triton"] = t
        # Add some common triton attributes if needed
        t.jit = lambda x: x
        t.Config = lambda *args, **kwargs: None
        sys.modules["triton.compiler"] = ModuleType("triton.compiler")
        sys.modules["triton.language"] = ModuleType("triton.language")
        sys.modules["triton.ops"] = ModuleType("triton.ops")

import uvicorn
import threading
import webview
from system.config_manager import load_config
from system import coordinator_settings as cs
from system.log import info, error
from system.api import init_app

def start_server(app, host, port):
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)

def main():
    # STRICT: no command-line arguments. Use the project root.
    root_path = ROOT_DIR
    info(f"Starting Onika system in {root_path}")

    # Load configuration
    load_config(root_path)

    # Update context with real root path
    from system.context import DATASET_MANAGER
    DATASET_MANAGER.root_path = root_path

    # Initialize API
    app = init_app(root_path)

    # Get host and port from settings
    host = cs.SETTINGS.get("host", "127.0.0.1")
    port = cs.SETTINGS.get("port", 7860)
    url = f"http://{host}:{port}"

    info(f"Starting server at {url}")
    
    # Start server in thread
    t = threading.Thread(target=start_server, args=(app, host, port), daemon=True)
    t.start()
    
    # Start webview
    try:
        icon_path = (ROOT_DIR / "ui" / "favicon.ico").resolve()
        
        # Create the main window
        webview.create_window(
            "Onika Trainer", 
            url, 
            width=1200, 
            height=720, 
            min_size=(800, 450)
        )
        
        # Start the application, applying the icon if it exists
        if icon_path.exists():
            try:
                webview.start(icon=str(icon_path))
            except Exception as icon_exc:
                error(f"Failed to start webview with icon: {icon_exc}")
                webview.start()
        else:
            webview.start()
            
    except Exception as e:
        error(f"Failed to start webview: {e}")
        # Keep main thread alive if webview fails
        t.join()

if __name__ == "__main__":
    main()
