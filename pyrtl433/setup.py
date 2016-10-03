from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
  name = 'Cython rtlsdrutils',
  ext_modules = cythonize("crtlsdrutils.pyx"),
    include_dirs=[numpy.get_include()]
)
