import os

if os.getenv("CI") and os.name == "nt":
    location = r"C:\msys64\mingw64\bin"
    os.environ["PATH"] = location + os.pathsep + os.getenv("PATH")
    import ctypes

    ctypes.CDLL(r"C:\msys64\mingw64\bin\OPENGL32.dll")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(location)
