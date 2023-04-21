from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='MatrixForge',
  version='23.0.1',
  description='MatrixForge for neural networks is a set of tools and programming libraries that allow you to quickly and easily create, train and evaluate the effectiveness of ML models.',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='',  
  author='Kacper Popek',
  author_email='popeqkacper@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords='machinelearning', 
  packages=find_packages(),
  install_requires=['numpy', 'dataclasses']
)