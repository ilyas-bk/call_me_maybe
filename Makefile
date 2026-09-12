SHELL := /bin/bash

all: install run

install:
	uv pip install pydantic && uv pip install numpy

run:
	HF_HOME=/media/ibouelba/elay-drive/hf-cache UV_CACHE_DIR=/media/ibouelba/elay-drive/uv-cache uv run python -m src

clean:
	rm -rf src/__pycache__

