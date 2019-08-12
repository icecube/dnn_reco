#!/usr/bin/env python

from setuptools import setup

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
    packages=['dnn_reco'],
    install_requires=['numpy', 'pandas', 'click', 'ruamel.yaml', 'tables',
                      'gitpython', 'tfscripts', 'h5py',
                      ],
    dependency_links=[
      'git+https://github.com/mhuen/TFScripts.git@master#egg=tfscripts-0.0.1',
    ],
    )
