from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
  name = 'Cython rtlsdrutils',
  ext_modules = cythonize("rtl433/crtl433.pyx"),
    include_dirs=[numpy.get_include()]
)
