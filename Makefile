SHELL := /bin/bash

.DEFAULT_GOAL := help

.PHONY: help train validate test unit-test eval generate

PYTHON ?= python3
TRAIN_SCRIPT ?= train.py
EVAL_SCRIPT ?= evaluate/eval.py
GENERATE_SCRIPT ?= generate.py

TRAINER_CONFIG ?= config/trainer.yaml
MODEL_CONFIG ?= config/model.yaml
DATA_CONFIG ?= config/data.yaml

# Convenience selector: `make train EXPERIMENT=experiment_alt`
# resolves to `config/experiment_alt.yaml`.
EXPERIMENT ?=
ifneq ($(strip $(EXPERIMENT)),)
EXPERIMENT_CONFIG ?= config/$(EXPERIMENT).yaml
else
EXPERIMENT_CONFIG ?= config/experiment.yaml
endif

CKPT ?=
PRETRAINED ?=
PRETRAINED_USE_EMA ?= false
PRETRAINED_STRICT ?= false

# Defaults aligned with config/experiment.yaml
WANDB_PROJECT ?= prism_tts
WANDB_NAME ?= mask-dyn
WANDB_SAVE_DIR ?= logs
WANDB_OFFLINE ?= false
WANDB_LOG_MODEL ?= false
WANDB_ENTITY ?=
WANDB_GROUP ?=
WANDB_TAGS ?=
PYTORCH_CUDA_ALLOC_CONF ?= backend:cudaMallocAsync
# Adaptive length-estimation parallelism defaults.
PRISM_TTS_ADAPTIVE_LENGTH_WORKERS ?= 16
PRISM_TTS_ADAPTIVE_LENGTH_CHUNK_SIZE ?= 8192
PRISM_TTS_ADAPTIVE_LENGTH_MIN_PARALLEL_SAMPLES ?= 20000
# Distributed debug defaults (override at runtime if needed).
TORCH_DISTRIBUTED_DEBUG ?= DETAIL
NCCL_DEBUG ?= INFO
# Optional rendezvous overrides (needed for multi-node runs).
MASTER_ADDR ?=
MASTER_PORT ?=

TRAIN_ARGS ?=
VALIDATE_ARGS ?=
TEST_ARGS ?=
PYTEST_ARGS ?=
EVAL_ARGS ?=
GENERATE_ARGS ?=

COMMON_TRAIN_ARGS = \
	--trainer-config $(TRAINER_CONFIG) \
	--model-config $(MODEL_CONFIG) \
	--data-config $(DATA_CONFIG) \
	--experiment-config $(EXPERIMENT_CONFIG)

WANDB_ARGS :=
ifneq ($(strip $(WANDB_PROJECT)),)
WANDB_ARGS += --wandb-project "$(WANDB_PROJECT)"
endif
ifneq ($(strip $(WANDB_NAME)),)
WANDB_ARGS += --wandb-name "$(WANDB_NAME)"
endif
ifneq ($(strip $(WANDB_SAVE_DIR)),)
WANDB_ARGS += --wandb-save-dir "$(WANDB_SAVE_DIR)"
endif
ifneq ($(strip $(WANDB_OFFLINE)),)
WANDB_ARGS += --wandb-offline "$(WANDB_OFFLINE)"
endif
ifneq ($(strip $(WANDB_LOG_MODEL)),)
WANDB_ARGS += --wandb-log-model "$(WANDB_LOG_MODEL)"
endif
ifneq ($(strip $(WANDB_ENTITY)),)
WANDB_ARGS += --wandb-entity "$(WANDB_ENTITY)"
endif
ifneq ($(strip $(WANDB_GROUP)),)
WANDB_ARGS += --wandb-group "$(WANDB_GROUP)"
endif
ifneq ($(strip $(WANDB_TAGS)),)
WANDB_ARGS += --wandb-tags "$(WANDB_TAGS)"
endif

ifneq ($(strip $(CKPT)),)
CKPT_ARG := --ckpt-path $(CKPT)
else
CKPT_ARG :=
endif

ifneq ($(strip $(PRETRAINED)),)
PRETRAINED_ARG := --pretrained-path $(PRETRAINED)
ifeq ($(strip $(PRETRAINED_USE_EMA)),true)
PRETRAINED_ARG += --pretrained-use-ema
else
PRETRAINED_ARG += --pretrained-no-ema
endif
ifeq ($(strip $(PRETRAINED_STRICT)),true)
PRETRAINED_ARG += --pretrained-strict
else
PRETRAINED_ARG += --pretrained-non-strict
endif
else
PRETRAINED_ARG :=
endif

help:
	@echo "Prism-TTS workflow automation"
	@echo ""
	@echo "Targets:"
	@echo "  make train      - Train model"
	@echo "  make validate   - Run validation only"
	@echo "  make test       - Train/Resume then run test loop (--test-after-fit)"
	@echo "  make unit-test  - Run pytest suite in ./test"
	@echo "  make eval       - Evaluate outputs (TBD until evaluate/eval.py is implemented)"
	@echo "  make generate   - Generate samples (TBD until generate.py is implemented)"
	@echo ""
	@echo "Common overrides:"
	@echo "  CKPT=<path>            Add --ckpt-path (resume Lightning trainer state)"
	@echo "  PRETRAINED=<path>      Add --pretrained-path (model weights only)"
	@echo "  PRETRAINED_USE_EMA=... Toggle EMA when PRETRAINED is set (default: false)"
	@echo "  PRETRAINED_STRICT=...  Toggle strict pretrained loading (default: false)"
	@echo "  EXPERIMENT=<name>      Use config/<name>.yaml as experiment config"
	@echo "  EXPERIMENT_CONFIG=...  Override experiment config"
	@echo "  TRAINER_CONFIG=...     Override trainer config"
	@echo "  MODEL_CONFIG=...       Override model config"
	@echo "  DATA_CONFIG=...        Override data config"
	@echo "  WANDB_PROJECT=...      WandB project (default: prism_tts)"
	@echo "  WANDB_NAME=...         WandB run name (default: baseline_local)"
	@echo "  WANDB_SAVE_DIR=...     WandB save dir (default: logs)"
	@echo "  WANDB_OFFLINE=true     WandB offline mode (default: false)"
	@echo "  WANDB_LOG_MODEL=...    WandB log_model (default: false)"
	@echo "  WANDB_ENTITY=...       Override WandB entity"
	@echo "  WANDB_GROUP=...        Override WandB group"
	@echo "  WANDB_TAGS=a,b,c       Override WandB tags"
	@echo "  PYTORCH_CUDA_ALLOC_CONF=...  CUDA allocator config (default: backend:cudaMallocAsync)"
	@echo "  PRISM_TTS_ADAPTIVE_LENGTH_WORKERS=...            Adaptive length workers (default: 16)"
	@echo "  PRISM_TTS_ADAPTIVE_LENGTH_CHUNK_SIZE=...         Adaptive length chunk size (default: 8192)"
	@echo "  PRISM_TTS_ADAPTIVE_LENGTH_MIN_PARALLEL_SAMPLES=...  Adaptive length parallel threshold (default: 20000)"
	@echo "  TORCH_DISTRIBUTED_DEBUG=...  DDP debug mode (default: DETAIL)"
	@echo "  NCCL_DEBUG=...         NCCL debug mode (default: INFO)"
	@echo "  MASTER_ADDR=...        Optional DDP rendezvous host (multi-node)"
	@echo "  MASTER_PORT=...        Optional DDP rendezvous port (multi-node)"
	@echo ""
	@echo "Extra args:"
	@echo "  TRAIN_ARGS='...'"
	@echo "  VALIDATE_ARGS='...'"
	@echo "  TEST_ARGS='...'"
	@echo "  PYTEST_ARGS='...'"
	@echo "  EVAL_ARGS='...'"
	@echo "  GENERATE_ARGS='...'"

train:
	@unset LOCAL_RANK RANK WORLD_SIZE NODE_RANK; \
	if [ -z "$(strip $(MASTER_ADDR))" ]; then unset MASTER_ADDR; else export MASTER_ADDR="$(MASTER_ADDR)"; fi; \
	if [ -z "$(strip $(MASTER_PORT))" ]; then unset MASTER_PORT; else export MASTER_PORT="$(MASTER_PORT)"; fi; \
	export TORCH_DISTRIBUTED_DEBUG="$(TORCH_DISTRIBUTED_DEBUG)"; \
	export NCCL_DEBUG="$(NCCL_DEBUG)"; \
	export PRISM_TTS_ADAPTIVE_LENGTH_WORKERS="$(PRISM_TTS_ADAPTIVE_LENGTH_WORKERS)"; \
	export PRISM_TTS_ADAPTIVE_LENGTH_CHUNK_SIZE="$(PRISM_TTS_ADAPTIVE_LENGTH_CHUNK_SIZE)"; \
	export PRISM_TTS_ADAPTIVE_LENGTH_MIN_PARALLEL_SAMPLES="$(PRISM_TTS_ADAPTIVE_LENGTH_MIN_PARALLEL_SAMPLES)"; \
	PYTORCH_CUDA_ALLOC_CONF="$(PYTORCH_CUDA_ALLOC_CONF)" \
	$(PYTHON) $(TRAIN_SCRIPT) $(COMMON_TRAIN_ARGS) $(WANDB_ARGS) $(PRETRAINED_ARG) $(CKPT_ARG) $(TRAIN_ARGS)

validate:
	@unset LOCAL_RANK RANK WORLD_SIZE NODE_RANK; \
	if [ -z "$(strip $(MASTER_ADDR))" ]; then unset MASTER_ADDR; else export MASTER_ADDR="$(MASTER_ADDR)"; fi; \
	if [ -z "$(strip $(MASTER_PORT))" ]; then unset MASTER_PORT; else export MASTER_PORT="$(MASTER_PORT)"; fi; \
	export TORCH_DISTRIBUTED_DEBUG="$(TORCH_DISTRIBUTED_DEBUG)"; \
	export NCCL_DEBUG="$(NCCL_DEBUG)"; \
	export PRISM_TTS_ADAPTIVE_LENGTH_WORKERS="$(PRISM_TTS_ADAPTIVE_LENGTH_WORKERS)"; \
	export PRISM_TTS_ADAPTIVE_LENGTH_CHUNK_SIZE="$(PRISM_TTS_ADAPTIVE_LENGTH_CHUNK_SIZE)"; \
	export PRISM_TTS_ADAPTIVE_LENGTH_MIN_PARALLEL_SAMPLES="$(PRISM_TTS_ADAPTIVE_LENGTH_MIN_PARALLEL_SAMPLES)"; \
	PYTORCH_CUDA_ALLOC_CONF="$(PYTORCH_CUDA_ALLOC_CONF)" \
	$(PYTHON) $(TRAIN_SCRIPT) $(COMMON_TRAIN_ARGS) $(WANDB_ARGS) --validate-only $(PRETRAINED_ARG) $(CKPT_ARG) $(VALIDATE_ARGS)

test:
	@if [ -z "$(CKPT)" ]; then \
		echo "[make test] CKPT is empty: this will run fit before test."; \
	fi
	@unset LOCAL_RANK RANK WORLD_SIZE NODE_RANK; \
	if [ -z "$(strip $(MASTER_ADDR))" ]; then unset MASTER_ADDR; else export MASTER_ADDR="$(MASTER_ADDR)"; fi; \
	if [ -z "$(strip $(MASTER_PORT))" ]; then unset MASTER_PORT; else export MASTER_PORT="$(MASTER_PORT)"; fi; \
	export TORCH_DISTRIBUTED_DEBUG="$(TORCH_DISTRIBUTED_DEBUG)"; \
	export NCCL_DEBUG="$(NCCL_DEBUG)"; \
	export PRISM_TTS_ADAPTIVE_LENGTH_WORKERS="$(PRISM_TTS_ADAPTIVE_LENGTH_WORKERS)"; \
	export PRISM_TTS_ADAPTIVE_LENGTH_CHUNK_SIZE="$(PRISM_TTS_ADAPTIVE_LENGTH_CHUNK_SIZE)"; \
	export PRISM_TTS_ADAPTIVE_LENGTH_MIN_PARALLEL_SAMPLES="$(PRISM_TTS_ADAPTIVE_LENGTH_MIN_PARALLEL_SAMPLES)"; \
	PYTORCH_CUDA_ALLOC_CONF="$(PYTORCH_CUDA_ALLOC_CONF)" \
	$(PYTHON) $(TRAIN_SCRIPT) $(COMMON_TRAIN_ARGS) $(WANDB_ARGS) --test-after-fit $(PRETRAINED_ARG) $(CKPT_ARG) $(TEST_ARGS)

unit-test:
	$(PYTHON) -m pytest test $(PYTEST_ARGS)

eval:
	@if [ ! -s "$(EVAL_SCRIPT)" ]; then \
		echo "[make eval] TBD: $(EVAL_SCRIPT) is empty or missing."; \
		echo "Implement the evaluation entrypoint, then use EVAL_ARGS for runtime flags."; \
	else \
		$(PYTHON) $(EVAL_SCRIPT) $(EVAL_ARGS); \
	fi

generate:
	@if [ ! -s "$(GENERATE_SCRIPT)" ]; then \
		echo "[make generate] TBD: $(GENERATE_SCRIPT) is empty or missing."; \
		echo "Implement the generation entrypoint, then use GENERATE_ARGS for runtime flags."; \
	else \
		$(PYTHON) $(GENERATE_SCRIPT) $(GENERATE_ARGS); \
	fi
