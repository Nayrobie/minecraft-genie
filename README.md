conda create -n mc-genie "python=3.12"
conda activate mc-genie

pip install poetry

poetry install --no-root --with dev

poetry run ruff check . --fix; poetry run ruff format .