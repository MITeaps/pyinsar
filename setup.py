# from distutils.core import setup
from setuptools import setup
from setuptools import find_packages

package_name = 'pyinsar'

package_list = find_packages()

with open("README.md", 'r', encoding='utf-8') as rfile:
    readme = rfile.read()

setup(name     = package_name,
      version  = '0.0.5',
      packages = package_list,

      install_requires = [
          'numpy',
          'scikit-dataaccess',
          'scikit-discovery',
          'pandas',
          'scipy',
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

      package_data={'pyinsar': ['license/LICENSE',
                                'docs/pyinsar_doxygen.pdf']},


      python_requires='>=3.4',

      long_description = readme,
      long_description_content_type='text/markdown'
)
