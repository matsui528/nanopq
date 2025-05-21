.PHONY: test clean build format deploy test_deploy mypy

test: #mypy
	pytest

mypy:
	mypy nanopq --ignore-missing-imports

clean:
	rm -rf build dist *.egg-info

build:
	python setup.py sdist bdist_wheel

# To run format, install pysen by 'pip install "pysen[lint]"'
format:
	pysen run format

deploy: clean build
	twine upload dist/*

test_deploy: clean build
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*	
