.PHONY: init test clean upload

init:
	pip install -r requirements.txt

test:
	python -m unittest tests/*.py

clean:
	rm -rf build dist *.egg-info

upload:
	python setup.py sdist bdist_wheel
	twine upload dist/*
