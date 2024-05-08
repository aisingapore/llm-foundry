#!/bin/bash

set -euo pipefail

cp -r /workspace/llm_foundry/scripts/eval/local_data $HF_DATASETS_CACHE

composer /workspace/llm_foundry/scripts/eval/eval.py $@
