from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("jmaBackup.pyx")
)

# python3 setup.py build_ext --inplace