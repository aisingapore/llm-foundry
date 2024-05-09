#!/bin/bash

set -euo pipefail

cp -r /workspace/llm_foundry/scripts/eval/local_data $HF_DATASETS_CACHE

args=$(getopt -o c:s: -l cli:,script: -n 'm3exam' -- "$@")
if [ $? != 0 ]; then
    echo 'Terminating...' >&2
    exit 1
fi
eval set -- "${args}"

cli_args=
script_args=
while true; do
    case "${1}" in
        -c | --cli )
            cli_args=$(echo ${2} | sed 's/,--/ --/g')
            shift 2
            ;;
        -s | --script )
            script_args=$(echo ${2} | sed 's/,--/ --/g')
            shift 2
            ;;
        -- )
            shift
            break
            ;;
        * )
            break
            ;;
    esac
done

composer ${cli_args} /workspace/llm_foundry/scripts/eval/eval.py ${script_args}
