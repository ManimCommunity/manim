from shutil import *
from os import *

rmtree(".travis")
rmtree(".github")
rmtree("logo")
rmtree("readme-assets")
rmtree("tests")
remove(".travis.yml")
remove("setup.py")