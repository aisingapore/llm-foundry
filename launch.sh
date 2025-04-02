#!bin/sh

gpus_per_node=${1:-$(nvidia-smi -L | wc -l)}
world_size=${2:-$gpus_per_node}
node_rank=${3:-0}
master_addr=${4:-'127.0.0.1'}
master_port=${5:-$((10000 + $RANDOM % 9000))}

set -euo pipefail

source .venv/bin/activate

yaml_path=scripts/train/yamls/pretrain/mpt-13b.yaml
script=scripts/train/train.py



# check both files exist
if [ ! -f "$yaml_path" ]; then
    echo "YAML file not found: $yaml_path"
    exit 1
fi

if [ ! -f "$script" ]; then
    echo "Script file not found: $script"
    exit 1
fi


composer_args=(
    composer
    --nproc $gpus_per_node
    --world_size $world_size
    --node_rank $node_rank
    --master_addr $master_addr
    --master_port $master_port
    --verbose
    $script
    $yaml_path
)

composer_args=${composer_args[@]}

echo "Running command: $composer_args"

[ -d $TRITON_CACHE_DIR ] || mkdir -p $TRITON_CACHE_DIR

# check if log_dir is present

log_dir=${log_dir:-logs}

mkdir -p $log_dir

$composer_args 2>&1 | tee -a $log_dir/${SLURM_PROCID}_python.log