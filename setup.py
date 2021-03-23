import re

from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

with open("nanopq/__init__.py") as f:
    version = re.search(r"__version__ = \"(.*?)\"", f.read()).group(1)

setup(
    name="nanopq",
    version=version,
    description="Pure python implementation of product quantization for nearest neighbor search ",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Yusuke Matsui",
    author_email="matsui528@gmail.com",
    url="https://github.com/matsui528/nanopq",
    license="MIT",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=["numpy", "scipy"],
)
