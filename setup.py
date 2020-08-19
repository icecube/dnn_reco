#!/usr/bin/env python

from setuptools import setup, find_packages

# get version number
exec(compile(open('dnn_reco/__init__.py', "rb").read(),
             'dnn_reco/__init__.py',
             'exec'))

setup(
    name='dnn_reco',
    version=__version__,
    description='DNN reconstruction for IceCube',
    long_description='',
    author='Mirco Huennefeld',
    author_email='mirco.huennefeld@tu-dortmund.de',
    url='https://icecube.wisc.edu/~mhuennefeld/docs/dnn_reco/html/index.html',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'click', 'ruamel.yaml',
                      'gitpython', 'tfscripts', 'h5py',
                      ],
    dependency_links=[
      'git+https://github.com/IceCubeOpenSource/TFScripts.git',
    ],
    )
