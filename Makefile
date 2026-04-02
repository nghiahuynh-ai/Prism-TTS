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

# Defaults aligned with config/experiment.yaml
WANDB_PROJECT ?= prism_tts
WANDB_NAME ?= baseline
WANDB_SAVE_DIR ?= logs
WANDB_OFFLINE ?= false
WANDB_LOG_MODEL ?= false
WANDB_ENTITY ?=
WANDB_GROUP ?=
WANDB_TAGS ?=

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
	@echo "  CKPT=<path>            Add --ckpt-path"
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
	@echo ""
	@echo "Extra args:"
	@echo "  TRAIN_ARGS='...'"
	@echo "  VALIDATE_ARGS='...'"
	@echo "  TEST_ARGS='...'"
	@echo "  PYTEST_ARGS='...'"
	@echo "  EVAL_ARGS='...'"
	@echo "  GENERATE_ARGS='...'"

train:
	$(PYTHON) $(TRAIN_SCRIPT) $(COMMON_TRAIN_ARGS) $(WANDB_ARGS) $(CKPT_ARG) $(TRAIN_ARGS)

validate:
	$(PYTHON) $(TRAIN_SCRIPT) $(COMMON_TRAIN_ARGS) $(WANDB_ARGS) --validate-only $(CKPT_ARG) $(VALIDATE_ARGS)

test:
	@if [ -z "$(CKPT)" ]; then \
		echo "[make test] CKPT is empty: this will run fit before test."; \
	fi
	$(PYTHON) $(TRAIN_SCRIPT) $(COMMON_TRAIN_ARGS) $(WANDB_ARGS) --test-after-fit $(CKPT_ARG) $(TEST_ARGS)

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
