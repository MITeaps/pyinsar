# from distutils.core import setup
from setuptools import setup
from setuptools import find_packages

package_name = 'pyinsar'

package_list = find_packages()

setup(name     = package_name,
      version  = '0.0.4',
      packages = package_list,

      install_requires = [
          'numpy',
          'pandas',
          'scipy',
          'python-opencv',
          'numba',
          'statsmodels',
          'geodesy',
          'GDAL',
          'matplotlib',
          'ipywidgets',
          'atomicwrites',
          'requests',
          'setuptools'
      ],

      description = 'Package of InSAR utilities',
      author = 'MITHAGI',
      author_email='skdaccess@mit.edu',
      classifiers=[
          'Topic :: Scientific/Engineering',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3 :: Only'
          ],

      python_requires='>=3.4',
)
