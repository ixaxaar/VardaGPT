.PHONY: help setup train evaluate inference precommit format clean

.DEFAULT_GOAL := help

help:
	@echo "[35mVardaGPT[0m - Memory-enhanced GPT-2 model powered by Hugging Face Transformers and FAISS"
	@echo "[1mUsage:[0m"
	@echo "  make [35m<command>[0m"
	@echo ""
	@echo "[1mCommands:[0m"
	@echo "  [35mhelp[0m        Display this help message"
	@echo "  [35msetup[0m       Set up the project by creating a virtual environment and installing dependencies"
	@echo "  [35mtrain[0m       Train the VardaGPT model"
	@echo "  [35mevaluate[0m    Evaluate the trained model on validation and testing sets"
	@echo "  [35minference[0m   Generate text using the memory-enhanced GPT-2 model"
	@echo "  [35mprecommit[0m   Run pre-commit hooks manually on all files"
	@echo "  [35mformat[0m      Format code using black, flake8, mypy, and prettier"
	@echo "  [35mclean[0m       Clean up the project directory by removing virtual environment and temporary files"

setup:
	python -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt

train:
	source venv/bin/activate
	python src/train.py

evaluate:
	source venv/bin/activate
	python src/evaluate.py

inference:
	source venv/bin/activate
	python src/inference.py --prompt "Your prompt text here"

precommit:
	pre-commit run --all-files

format:
	pre-commit run --all-files

clean:
	rm -rf venv/
	find . -type f -name "*.pyc" -exec rm -f {} \;
	find . -type d -name "__pycache__" -exec rm -rf {} \;
