"""
Marbl Meat QC Pipeline — Server Launcher.

Usage:
    python start_server.py

This script guarantees:
  1. The backend directory is added to sys.path regardless of where the command is run.
  2. The server binds cleanly to http://127.0.0.1:8000 without firewall blocks.
  3. Auto-reload is enabled for development.
"""

import os
import sys
from pathlib import Path

# Add backend directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

try:
    import uvicorn
except ImportError:
    print("[ERROR] 'uvicorn' is not installed in this Python environment.")
    print("Install it with: pip install uvicorn[standard]")
    sys.exit(1)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🥩 Marbl Meat Quality Control Pipeline")
    print("  Server URL: http://localhost:8000")
    print("  Module 1: Meat Intake & Predictive Shelf-Life Assessment")
    print("  Module 2: AI Visual Quality & Sizing Conveyor Inspection")
    print("  Module 3: AI Visible Fat Coverage Segmentation")
    print("=" * 60 + "\n")

    try:
        uvicorn.run(
            "app:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            app_dir=str(BACKEND_DIR),
        )
    except Exception as exc:
        print(f"\n[ERROR] Server launch failed: {exc}")
        print("Tip: If port 8000 is in use, check task manager or change port.\n")
        sys.exit(1)

