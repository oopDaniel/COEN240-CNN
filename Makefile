remove:
	rm -rf ./tmp

install:
	@echo Creating virtual environment...
	python3 -m venv tmp
	@echo Installing pkg...
	./tmp/bin/pip3 install numpy tensorflow sklearn keras

start:
	./tmp/bin/python3 main.py

all:
	make remove install start

.PHONY: remove install start all