# Throughput Bench Makefile
# Usage: make setup && make benchmark
#
# Override GPU_ID to use a different GPU:
#   make benchmark GPU_ID=2

GPU_ID ?= 0
PYTHON ?= python

.PHONY: setup setup-pip benchmark lint format help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup:  ## Create conda environment and install dependencies
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo ""
	@echo "✅ Environment ready. Activate with:"
	@echo "   conda activate throughput-bench"

setup-pip:  ## Install dependencies with pip (alternative to conda)
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

benchmark:  ## Run the full benchmark on a single GPU (override GPU_ID, pass extra flags via ARGS)
	$(PYTHON) benchmark.py --gpu-id $(GPU_ID) $(ARGS)

lint:  ## Run ruff linter
	ruff check .

format:  ## Format code with ruff
	ruff format .
