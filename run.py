import subprocess
import time
import threading

def run_backend():
    subprocess.run(["uvicorn", "app.main:app", "--reload"])

def run_frontend():
    time.sleep(3)
    subprocess.run(["python", "frontend/frontend.py"])

if __name__ == "__main__":
    threading.Thread(target=run_backend, daemon=True).start()
    run_frontend()