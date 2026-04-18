# ThroughputBencher Makefile
# Usage: make setup && make benchmark && make visualize
#
# Override GPU_ID to use a different GPU:
#   make benchmark GPU_ID=2

GPU_ID ?= 0
PYTHON ?= python

.PHONY: setup benchmark benchmark-quick benchmark-compiled geobenchmark visualize lint format clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup:  ## Create conda environment and install dependencies
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo ""
	@echo "✅ Environment ready. Activate with:"
	@echo "   conda activate throughput-bencher"

setup-pip:  ## Install dependencies with pip (alternative to conda)
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

benchmark:  ## Run full benchmark (auto batch size, 30s/config, all models)
	$(PYTHON) benchmark.py --gpu-id $(GPU_ID)
	@echo ""
	@echo "Now run 'make visualize' to generate charts"

benchmark-compiled:  ## Run benchmark with torch.compile (default + max-autotune)
	$(PYTHON) benchmark.py --gpu-id $(GPU_ID) --compile-modes default max-autotune

benchmark-quick:  ## Quick benchmark (4 models, 10s/config)
	$(PYTHON) benchmark.py --gpu-id $(GPU_ID) \
		--models resnet18 resnet50 efficientnet_b0 vit_base_patch16_224 \
		--timed-seconds 10

geobenchmark:  ## Geo foundation model comparison (geo models + timm baselines at matching inputs)
	$(PYTHON) benchmark.py --gpu-id $(GPU_ID) --geo-compare

visualize:  ## Generate charts from results/ CSVs
	$(PYTHON) visualize.py

lint:  ## Run ruff linter
	ruff check .

format:  ## Format code with ruff
	ruff format .

clean:  ## Remove generated figures
	rm -f figures/*.png figures/*.svg
	@echo "✅ Cleaned figures"
