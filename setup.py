from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='nanopq',
    version='0.1.0',
    description='Product quantization for nearest neighbor search in a single python file',
    long_description=readme,
    author='Yusuke Matsui',
    author_email='matsui528@gmail.com',
    url='https://github.com/matsui528/nanopq',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

