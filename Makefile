.PHONY: test clean build deploy test_deploy

test:
	pytest

clean:
	rm -rf build dist *.egg-info

build:
	python setup.py sdist bdist_wheel

deploy: clean build
	twine upload dist/*

test_deploy: clean build
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*	
