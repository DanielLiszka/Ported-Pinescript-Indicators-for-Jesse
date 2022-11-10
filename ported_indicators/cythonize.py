from setuptools import setup
from Cython.Build import cythonize
from Cython.Compiler import Options
import multiprocessing
#python3 cythonize.py build_ext --inplace
#jesse backtest  '2021-01-01' '2021-03-01' 
# Options.annotate = True
Options.convert_range = True

setup(
    ext_modules=cythonize( "*.pyx",nthreads=6,compiler_directives={'language_level':3,'infer_types':True,'optimize.use_switch':True,'optimize.unpack_method_calls':True,'cdivision':True,'wraparound':False,'boundscheck':False,})
)