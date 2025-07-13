#!/bin/bash

set -e  # Exit immediately if any command exits with a non-zero status

# Usage check
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <model_type> <segment_epochs> <classify_epochs <rounds> <name> <python_command>"
    echo "Example: $0 11m 20 5 5 attempt1 python3"
    exit 1
fi

# Parse command line arguments
model_type=$1        # e.g., 11m
segment_epochs=$2    # e.g., 20
classify_epochs=$3   # e.g., 5
rounds=$4            # e.g., 5
name=$5              # e.g., attempt1
python_command=$6    # e.g., python3

# Derived variables
project_name="${name}_${model_type}"
last_segment_model="yolo${model_type}-seg.pt" #pretrained on COCO
last_classify_model="yolo${model_type}-cls.pt" #pretrained on ImageNet

train_yaml="yolo_dataset/cvs_only.yaml" #change this to cvs_endo.yaml if use_endoscapes
endoscapes_arg="" #change this to --use_endoscapes if use_endoscapes

if [ -d "$project_name" ]; then
    echo "Error: Directory '$project_name' already exists. Aborting to prevent overwrite."
    exit 1
fi

# Loop
for round in $(seq 1 ${rounds}); do
    echo "=== Round $round ==="

    # Train segmentation model
    yolo task=segment mode=train model=${last_segment_model} data=yolo_dataset/cvs_only.yaml \
         epochs=${segment_epochs} imgsz=640 batch=16 project=${project_name} name=segment_${round}

    last_segment_model="${project_name}/segment_${round}/weights/best.pt"
    new_classify_model="${project_name}/segment_${round}/weights/classify.pt"

    ${python_command} train_code/train.py --model_name ${last_classify_model} --backbone_model ${last_segment_model} --num_epochs ${classify_epochs} --mlc_batch_size 32 --output_file ${new_classify_model}
    last_classify_model=${new_classify_model}
    new_segment_model="${project_name}/segment_${round}/weights/segment_classify_shared.pt"

    if [ "$round" -lt "$rounds" ]; then
        # Copy backbone back to segmentation model
        ${python_command} train_code/copy_backbone.py ${last_segment_model} ${last_classify_model} ${new_segment_model}
        last_segment_model=${new_segment_model}
    fi
done

echo "Final segmentation model: ${last_segment_model}"
echo "Final classification model: ${last_classify_model}"
