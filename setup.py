#!/usr/bin/env python

from distutils.core import setup
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop

import stable


class install(_install):
    def run(self):
        _install.run(self)


class develop(_develop):
    def run(self):
        _develop.run(self)


setup(    name = "pyStable",
    version = stable.__version__,
    author = "Francisco J. Martinez-Murcia",
    author_email = "fjesusmartinez@ugr.es",
    url = "https://github.com/pakitochus/pyStable",
    license = "GPL",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
         'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    keywords = 'pdf, cdf, distribution, stable, levy, alpha-stable',
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    include_package_data = True,
    package_dir={'stable': 'stable'},
    package_data={
        'stable': ['data/*.npz'],
    },
    description="Library based on pyLevy for working with Alpha-Stable distributions.",
    long_description=stable.stable.__doc__,
    packages=["stable"],
)
