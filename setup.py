from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='nanopq',
    version='0.1.5',
    description='Pure python implementation of product quantization for nearest neighbor search ',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Yusuke Matsui',
    author_email='matsui528@gmail.com',
    url='https://github.com/matsui528/nanopq',
    license='MIT',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['numpy', 'scipy'],
)
