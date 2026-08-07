SHELL := /bin/bash

.DEFAULT_GOAL := help

.PHONY: help train validate test unit-test eval generate generate-metadata

PYTHON ?= python3
TRAIN_SCRIPT ?= train.py
EVAL_SCRIPT ?= evaluate/eval.py
GENERATE_SCRIPT ?= generate.py
GENERATE_METADATA_SCRIPT ?= scripts/generate_from_metadata.py

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
PRETRAINED_WEIGHTS ?=

# Batch inference defaults for the LibriSpeech-style four-column metadata
# format: target_id|target_text|prompt_id|prompt_text.
METADATA ?= /astro/nghiahuynh/data/librispeech/LibriSpeech/metadata-test-clean.txt
METADATA_AUDIO_ROOT ?= /astro/nghiahuynh/data/librispeech/LibriSpeech
# Leave empty to infer the split from metadata-test-clean.txt.
METADATA_SPLIT ?=
METADATA_AUDIO_EXTENSION ?= .flac
METADATA_OUTPUT_DIR ?= synth/librispeech-test-clean
METADATA_REPORT ?= $(METADATA_OUTPUT_DIR)/report.jsonl

# Defaults aligned with config/experiment.yaml
WANDB_PROJECT ?= prism_tts
WANDB_NAME ?= ar-2048
WANDB_SAVE_DIR ?= logs
WANDB_OFFLINE ?= false
WANDB_LOG_MODEL ?= false
WANDB_ENTITY ?=
WANDB_GROUP ?=
WANDB_TAGS ?=
PYTORCH_CUDA_ALLOC_CONF ?= expandable_segments:True,garbage_collection_threshold:0.8

TRAIN_ARGS ?=
VALIDATE_ARGS ?=
TEST_ARGS ?=
PYTEST_ARGS ?=
EVAL_ARGS ?=
GENERATE_ARGS ?=
METADATA_ARGS ?= --skip-existing

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

ifneq ($(strip $(PRETRAINED_WEIGHTS)),)
PRETRAINED_WEIGHTS_ARG := --pretrained-weights $(PRETRAINED_WEIGHTS)
else
PRETRAINED_WEIGHTS_ARG :=
endif

ifneq ($(strip $(METADATA_SPLIT)),)
METADATA_SPLIT_ARG := --split "$(METADATA_SPLIT)"
else
METADATA_SPLIT_ARG :=
endif

ifneq ($(strip $(METADATA_REPORT)),)
METADATA_REPORT_ARG := --report "$(METADATA_REPORT)"
else
METADATA_REPORT_ARG :=
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
	@echo "  make generate            - Generate one sample with generate.py"
	@echo "  make generate-metadata   - Batch-generate from four-column LibriSpeech metadata"
	@echo ""
	@echo "Common overrides:"
	@echo "  CKPT=<path>            Add --ckpt-path (or --checkpoint for generate-metadata)"
	@echo "  PRETRAINED_WEIGHTS=<path>  Initialize model weights without resuming Trainer state"
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
	@echo "  PYTORCH_CUDA_ALLOC_CONF=...  CUDA allocator config (default: expandable segments)"
	@echo "  METADATA=...            Four-column target/prompt metadata file"
	@echo "  METADATA_AUDIO_ROOT=... LibriSpeech root containing split directories"
	@echo "  METADATA_SPLIT=...      Override inferred split (e.g. test-clean)"
	@echo "  METADATA_OUTPUT_DIR=... Batch WAV output directory"
	@echo "  METADATA_REPORT=...     JSONL batch report path (empty to disable)"
	@echo ""
	@echo "Extra args:"
	@echo "  TRAIN_ARGS='...'"
	@echo "  VALIDATE_ARGS='...'"
	@echo "  TEST_ARGS='...'"
	@echo "  PYTEST_ARGS='...'"
	@echo "  EVAL_ARGS='...'"
	@echo "  GENERATE_ARGS='...'"
	@echo "  METADATA_ARGS='...'"

train:
	PYTORCH_CUDA_ALLOC_CONF="$(PYTORCH_CUDA_ALLOC_CONF)" \
	$(PYTHON) $(TRAIN_SCRIPT) $(COMMON_TRAIN_ARGS) $(WANDB_ARGS) $(CKPT_ARG) $(PRETRAINED_WEIGHTS_ARG) $(TRAIN_ARGS)

validate:
	PYTORCH_CUDA_ALLOC_CONF="$(PYTORCH_CUDA_ALLOC_CONF)" \
	$(PYTHON) $(TRAIN_SCRIPT) $(COMMON_TRAIN_ARGS) $(WANDB_ARGS) --validate-only $(CKPT_ARG) $(PRETRAINED_WEIGHTS_ARG) $(VALIDATE_ARGS)

test:
	@if [ -z "$(CKPT)" ]; then \
		echo "[make test] CKPT is empty: this will run fit before test."; \
	fi
	PYTORCH_CUDA_ALLOC_CONF="$(PYTORCH_CUDA_ALLOC_CONF)" \
	$(PYTHON) $(TRAIN_SCRIPT) $(COMMON_TRAIN_ARGS) $(WANDB_ARGS) --test-after-fit $(CKPT_ARG) $(PRETRAINED_WEIGHTS_ARG) $(TEST_ARGS)

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

generate-metadata:
	@if [ -z "$(CKPT)" ]; then \
		echo "[make generate-metadata] CKPT is required." >&2; \
		exit 2; \
	fi
	@if [ ! -s "$(GENERATE_METADATA_SCRIPT)" ]; then \
		echo "[make generate-metadata] Missing script: $(GENERATE_METADATA_SCRIPT)" >&2; \
		exit 2; \
	fi
	PYTORCH_CUDA_ALLOC_CONF="$(PYTORCH_CUDA_ALLOC_CONF)" \
	$(PYTHON) $(GENERATE_METADATA_SCRIPT) \
		--checkpoint "$(CKPT)" \
		--metadata "$(METADATA)" \
		--audio-root "$(METADATA_AUDIO_ROOT)" \
		$(METADATA_SPLIT_ARG) \
		--audio-extension "$(METADATA_AUDIO_EXTENSION)" \
		--output-dir "$(METADATA_OUTPUT_DIR)" \
		$(METADATA_REPORT_ARG) \
		$(METADATA_ARGS)
