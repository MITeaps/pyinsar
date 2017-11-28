# from distutils.core import setup
from setuptools import setup
from setuptools import find_packages

package_name = 'pyinsar'

package_list = find_packages()

setup(name     = package_name,
      version  = '0.0.2',
      packages = package_list,

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
