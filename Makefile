.PHONY: init test clean build deploy test_deploy

init:
	pip install -r requirements.txt

test:
	python -m unittest tests/*.py

clean:
	rm -rf build dist *.egg-info

build:
	python setup.py sdist bdist_wheel

deploy: clean build
	twine upload dist/*

test_deploy: clean build
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*	