from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 2 - Pre-Alpha',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows',
  'License :: OSI Approved :: Apache Software License',
  'Programming Language :: Python :: 3',
  'Topic :: Scientific/Engineering :: Artificial Intelligence',
  'Topic :: Scientific/Engineering :: Artificial Life',
  'Topic :: Scientific/Engineering :: Visualization',
  'Natural Language :: English'
]
 
setup(
  name='matrixforge',
  version='23.0.1',
  description='MatrixForge for neural networks is a set of tools and programming libraries that allow you to quickly and easily create, train and evaluate the effectiveness of ML models.',
  long_description=open('README.md').read(),
  url='',  
  author='Kacper Popek',
  author_email='popeqkacper@gmail.com',
  license='Apache 2.0', 
  classifiers=classifiers,
  keywords='machinelearning', 
  packages=find_packages(),
  install_requires=['numpy', 'dataclasses']
)
