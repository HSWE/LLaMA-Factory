.PHONY: build commit license quality style test run_lmf

check_dirs := scripts src tests tests_v1 setup.py

build:
	pip3 install build && python3 -m build

commit:
	pre-commit install
	pre-commit run --all-files

license:
	python3 tests/check_license.py $(check_dirs)

quality:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)

style:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)

test:
	WANDB_DISABLED=true pytest -vv --import-mode=importlib tests/ tests_v1/

run_lmf:
	nohup env GRADIO_SERVER_PORT=7860 python run_lmf.py webui > logs/webui_7860.log 2>&1 &
	nohup env GRADIO_SERVER_PORT=7861 python run_lmf.py webui > logs/webui_7861.log 2>&1 &
	nohup env GRADIO_SERVER_PORT=7862 python run_lmf.py webui > logs/webui_7862.log 2>&1 &
	nohup env GRADIO_SERVER_PORT=7863 python run_lmf.py webui > logs/webui_7863.log 2>&1 &
	nohup env GRADIO_SERVER_PORT=7864 python run_lmf.py webui > logs/webui_7864.log 2>&1 &

