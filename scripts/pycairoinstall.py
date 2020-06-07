import platform
import os
import sys
import urllib.request

if 'Windows' in platform.system():
    #In case the python version is 3.6 and the system is 32-bit, try pycairo‑1.19.1‑cp37‑cp37m‑win32.whl version of cairo
    if sys.version[:3]=='3.6' and platform.machine()=='x86':
        urllib.request.urlretrieve("https://download.lfd.uci.edu/pythonlibs/w3jqiv8s/pycairo-1.19.1-cp36-cp36m-win32.whl", "pycairo-1.19.1-cp36-cp36m-win32.whl")
        os.system("pip install pycairo-1.19.1-cp36-cp36m-win32.whl")
        os.remove("pycairo-1.19.1-cp37-cp37m-win32.whl")

    #In case the python version is 3.6 and the system is 64-bit, try pycairo‑1.19.1‑cp37‑cp37m‑win32.whl version of cairo
    elif sys.version[:3]=='3.6' and platform.machine()=='AMD64':
        urllib.request.urlretrieve("https://download.lfd.uci.edu/pythonlibs/w3jqiv8s/pycairo-1.19.1-cp36-cp36m-win_amd64.whl", "pycairo-1.19.1-cp36-cp36m-win_amd64.whl")
        print("Sucessfully downloaded Cairo for your system")
        print("Installing Cairo")
        os.system("pip install pycairo-1.19.1-cp36-cp36m-win_amd64.whl")
        os.remove("pycairo-1.19.1-cp36-cp36m-win_amd64.whl")
    
    #In case the python version is 3.7 and the system is 32-bit, try pycairo‑1.19.1‑cp37‑cp37m‑win32.whl version of cairo
    elif sys.version[:3]=='3.7' and platform.machine()=='x86':
        urllib.request.urlretrieve("https://download.lfd.uci.edu/pythonlibs/w3jqiv8s/pycairo-1.19.1-cp37-cp37m-win32.whl", "pycairo-1.19.1-cp37-cp37m-win32.whl")
        print("Sucessfully downloaded Cairo for your system")
        print("Installing Cairo")
        os.system("pip install pycairo-1.19.1-cp37-cp37m-win32.whl")
        os.remove("pycairo-1.19.1-cp37-cp37m-win32.whl")

    #In case the python version is 3.7 and the system is AMD64, try pycairo-1.19.1-cp37-cp37m-win_amd64.whl version of cairo
    elif sys.version[:3]=='3.7' and platform.machine()=='AMD64':
        urllib.request.urlretrieve("https://download.lfd.uci.edu/pythonlibs/w3jqiv8s/pycairo-1.19.1-cp37-cp37m-win_amd64.whl", "pycairo-1.19.1-cp37-cp37m-win_amd64.whl")
        print("Sucessfully downloaded Cairo for your system")
        print("Installing Cairo")
        os.system("pip install pycairo-1.19.1-cp37-cp37m-win_amd64.whl")
        os.remove("pycairo-1.19.1-cp37-cp37m-win_amd64.whl")
        
    #In case the python version is 3.8 and the system is 32-bit, try pycairo-1.19.1-cp38-cp38-win32.whl version of cairo
    elif sys.version[:3]=='3.8' and platform.machine()=='x86':
        urllib.request.urlretrieve("https://download.lfd.uci.edu/pythonlibs/w3jqiv8s/pycairo-1.19.1-cp38-cp38-win32.whl", "pycairo-1.19.1-cp38-cp38-win32.whl")
        print("Sucessfully downloaded Cairo for your system")
        print("Installing Cairo")
        os.system("pip install pycairo-1.19.1-cp38-cp38-win32.whl")
        os.remove("pycairo-1.19.1-cp38-cp38-win32.whl")
        
    #In case the python version is 3.8 and the system is AMD64, try pycairo-1.19.1-cp38-cp38-win_amd64.whl version of cairo
    elif sys.version[:3]=='3.8' and platform.machine()=='AMD64':
        urllib.request.urlretrieve("https://download.lfd.uci.edu/pythonlibs/w3jqiv8s/pycairo-1.19.1-cp38-cp38-win_amd64.whl", "pycairo-1.19.1-cp38-cp38-win_amd64.whl")
        print("Sucessfully downloaded Cairo for your system")
        print("Installing Cairo")
        os.system("pip install pycairo-1.19.1-cp38-cp38-win_amd64.whl")
        os.remove("pycairo-1.19.1-cp38-cp38-win_amd64.whl")   
