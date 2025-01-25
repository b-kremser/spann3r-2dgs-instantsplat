#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

# Change the absolute path first!
DATA_ROOT_DIR="<Absolute_Path>/InstantSplat/assets"
OUTPUT_DIR="output_infer"
DATASETS=(
    scannetpp
)

SCENES=(
    a980334473_00
)

N_VIEWS=(
    3
)

gs_train_iter=(
  1000
)

# Can be 3DGS and / or 2DGS
gs_type=(
  3DGS
)

# Can be spann3r and / or dust3r
pc_initializer=(
  spann3r
)

# Function to get the id of an available GPU
get_available_gpu() {
    local mem_threshold=1000
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
    $2 < threshold { print $1; exit }
    '
}

# Function: Run task on specified GPU
run_on_gpu() {
    local GPU_ID=$1
    local DATASET=$2
    local SCENE=$3
    local N_VIEW=$4
    local gs_train_iter=$5
    local gs_type=$6
    local pc_initializer=$7
    SOURCE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/
    IMAGE_PATH=${SOURCE_PATH}images
    MODEL_PATH=./${OUTPUT_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views/${gs_type}_${pc_initializer}

    # Create necessary directories
    mkdir -p ${MODEL_PATH}

    echo "======================================================="
    echo "Starting process: ${DATASET}/${SCENE} (${N_VIEW} views/${gs_type}_${pc_initializer}/${gs_train_iter} iters) on GPU ${GPU_ID}"
    echo "======================================================="

    # (1) Co-visible Global Geometry Initialization
    # Choose the correct Python script based on pc_initializer
    if [ "$pc_initializer" = "spann3r" ]; then
        INIT_GEO_SCRIPT="./init_geo_spann3r.py"
    elif [ "$pc_initializer" = "dust3r" ]; then
        INIT_GEO_SCRIPT="./init_geo_dust3r.py"
    else
        echo "Invalid pc_initializer value: $pc_initializer. Must be 'spann3r' or 'dust3r'. Exiting."
        exit 1
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Co-visible Global Geometry Initialization..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ${INIT_GEO_SCRIPT} \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    --n_views ${N_VIEW} \
    --infer_video \
    > ${MODEL_PATH}/01_init_geo.log 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Co-visible Global Geometry Initialization completed. Log saved in ${MODEL_PATH}/01_init_geo.log"

    # (2) Train: jointly optimize pose
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training..."
    if [ "$gs_type" = "2DGS" ]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID} python 2DGS/train.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH} \
        -r 1 \
        --n_views ${N_VIEW} \
        --iterations ${gs_train_iter} \
        --optim_pose \
        --depth_ratio 0 \
        --lambda_dist 50 \
        --lambda_normal 0 \
        --pp_optimizer \
        > ${MODEL_PATH}/02_train.log 2>&1
    elif [ "$gs_type" = "3DGS" ]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID} python 3DGS/train.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH} \
        -r 1 \
        --n_views ${N_VIEW} \
        --iterations ${gs_train_iter} \
        --pp_optimizer \
        --optim_pose \
        > ${MODEL_PATH}/02_train.log 2>&1
    else
        echo "Invalid gs_type value: $gs_type. Must be '2DGS' or '3DGS'. Exiting."
        exit 1
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed. Log saved in ${MODEL_PATH}/02_train.log"

    # (3) Render-Training_View
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting rendering training views..."
    if [ "$gs_type" = "2DGS" ]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID} python 2DGS/render.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH} \
        -r 1 \
        --n_views ${N_VIEW} \
        --iterations ${gs_train_iter} \
        --depth_ratio 0 \
        --num_cluster 50 \
        --mesh_res 2048 \
        --depth_trunc 2.0 \
        > ${MODEL_PATH}/03_render_train.log 2>&1
    elif [ "$gs_type" = "3DGS" ]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID} python 3DGS/render.py \
        -s ${SOURCE_PATH} \
        -m ${MODEL_PATH} \
        -r 1 \
        --n_views ${N_VIEW} \
        --iterations ${gs_train_iter} \
        --skip_mesh \
        --infer_video \
        > ${MODEL_PATH}/03_render.log 2>&1
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering completed. Log saved in ${MODEL_PATH}/03_render_train.log"



    echo "======================================================="
    echo "Task completed: ${DATASET}/${SCENE} (${N_VIEW} views/${gs_type}_${pc_initializer}/${gs_train_iter} iters) on GPU ${GPU_ID}"
    echo "======================================================="
}

# Main loop
total_tasks=$((${#DATASETS[@]} * ${#SCENES[@]} * ${#N_VIEWS[@]} * ${#gs_train_iter[@]}))
current_task=0

for DATASET in "${DATASETS[@]}"; do
    for SCENE in "${SCENES[@]}"; do
        for N_VIEW in "${N_VIEWS[@]}"; do
            for gs_train_iter in "${gs_train_iter[@]}"; do
                for gs_type in "${gs_type[@]}"; do
                    for pc_initializer in "${pc_initializer[@]}"; do
                        current_task=$((current_task + 1))
                        echo "Processing task $current_task / $total_tasks"

                        # Get available GPU
                        GPU_ID=$(get_available_gpu)

                        # If no GPU is available, wait for a while and retry
                        while [ -z "$GPU_ID" ]; do
                            echo "[$(date '+%Y-%m-%d %H:%M:%S')] No GPU available, waiting 60 seconds before retrying..."
                            sleep 60
                            GPU_ID=$(get_available_gpu)
                        done

                        # Run the task in the background
                        (run_on_gpu $GPU_ID "$DATASET" "$SCENE" "$N_VIEW" "$gs_train_iter" "$gs_type" "$pc_initializer") &

                        # Wait for 20 seconds before trying to start the next task
                        sleep 10
                    done
                done
            done
        done
    done
done

# Wait for all background tasks to complete
wait

echo "======================================================="
echo "All tasks completed! Processed $total_tasks tasks in total."
echo "======================================================="
