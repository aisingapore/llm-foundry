#!/bin/bash

set -euo pipefail

cp -r eval/local_data $HF_DATASETS_CACHE

composer eval/eval.py $@
