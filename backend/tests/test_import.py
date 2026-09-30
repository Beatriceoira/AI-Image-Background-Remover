import sys
sys.path.insert(0, '/home/beatriceoira/Projects/background-remover')

try:
    from backend.app.main import app
    print("SUCCESS: Imported FastAPI app from backend.app.main")
except Exception as e:
    print(f"ERROR: Failed to import app: {e}")
    sys.exit(1)