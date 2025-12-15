.PHONY: smoke preprocess embed split coarsen train eval report ui clean venv deps ffmpeg

CONFIG ?= configs/smoke.json
RUN_NAME ?= $(shell python3 -c "import json;print(json.load(open('$(CONFIG)','r',encoding='utf-8'))['run_name'])")
PY ?= .venv/bin/python

export TMPDIR := $(CURDIR)/.cache/tmp
export PATH := $(CURDIR)/.cache/ffmpeg/bin:$(PATH)

venv:
	@test -x "$(PY)" || python3 -m venv .venv

deps: venv
	@mkdir -p "$(TMPDIR)" .cache/modelscope
	@$(PY) -m pip install -U pip
	@$(PY) -m pip install -r requirements.txt

ffmpeg:
	@mkdir -p .cache/ffmpeg
	@python3 bootstrap_ffmpeg.py

preprocess: deps ffmpeg
	@mkdir -p "$(TMPDIR)"
	@$(PY) -m dialectsense.cli preprocess --config $(CONFIG)

embed: deps ffmpeg
	@mkdir -p "$(TMPDIR)"
	@$(PY) -m dialectsense.cli embed --config $(CONFIG)

split: deps ffmpeg
	@mkdir -p "$(TMPDIR)"
	@$(PY) -m dialectsense.cli split --config $(CONFIG)

coarsen: deps ffmpeg
	@mkdir -p "$(TMPDIR)"
	@$(PY) -m dialectsense.cli coarsen --config $(CONFIG)

train: deps ffmpeg
	@mkdir -p "$(TMPDIR)"
	@$(PY) -m dialectsense.cli train --config $(CONFIG)

eval: deps ffmpeg
	@mkdir -p "$(TMPDIR)"
	@$(PY) -m dialectsense.cli eval --config $(CONFIG)

report: deps ffmpeg
	@mkdir -p "$(TMPDIR)"
	@$(PY) -m dialectsense.cli report --config $(CONFIG)

ui: deps ffmpeg
	@mkdir -p "$(TMPDIR)"
	@$(PY) -m dialectsense.cli ui --config $(CONFIG)

smoke: CONFIG := configs/smoke.json
smoke: preprocess embed split coarsen train eval report

clean:
	@rm -rf "artifacts/$(RUN_NAME)"
