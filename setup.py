#!/usr/bin/env python

from distutils.core import setup
exec(compile(open('version.py', "rb").read(),
             'version.py',
             'exec'))

setup(name='dnn_reco',
      version=__version__,
      description='DNN reconstruction for IceCube',
      author='Mirco Huennefeld',
      author_email='mirco.huennefeld@tu-dortmund.de',
      url='https://github.com/mhuen/dnn_reco',
      packages=['dnn_reco'],
      install_requires=['numpy', 'pandas', 'click', 'ruamel.yaml', 'tables',
                        'gitpython', 'tfscripts',
                        ],
      dependency_links=[
       'git+https://github.com/mhuen/TFScripts.git@master#egg=tfscripts-0.0.1',
      ],
      )
