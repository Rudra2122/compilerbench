# CompilerBench v2. `make all` = everything that runs on the laptop. `make aihub` needs an AI Hub token.
PY ?= python3

.PHONY: all data export quant executorch aihub report smoke clean

all: data export quant executorch report

data:        ; $(PY) data_prep.py
export:      ; $(PY) stage2a_onnx.py
quant:       ; $(PY) stage3_quantize.py
executorch:  ; $(PY) stage4a_executorch.py
aihub:       ; $(PY) stage5_aihub.py && $(PY) report.py
report:      ; $(PY) report.py

# Whole pipeline on random weights + random images: no downloads, ~3 min, checks plumbing only.
smoke:
	CB_SMOKE=1 $(PY) stage2a_onnx.py
	CB_SMOKE=1 $(PY) stage3_quantize.py --iters 30
	CB_SMOKE=1 $(PY) stage4a_executorch.py --iters 20 --portable-eval-limit 4
	CB_SMOKE=1 $(PY) report.py

clean:
	rm -rf artifacts results/smoke_* data/smoke_*
