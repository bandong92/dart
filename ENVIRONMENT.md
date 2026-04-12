# DART Environment

This project was developed and tested with:

- Python 3.11.9
- Windows
- PyQt5 5.15.11
- PyTorch 2.11.0 CPU
- torchvision 0.26.0
- ONNX 1.21.0
- ONNX Runtime 1.24.4

Create and install the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Run the application:

```powershell
.\.venv\Scripts\python.exe app.py
```

The `.venv` directory itself is intentionally not committed because it is large and machine-specific.
