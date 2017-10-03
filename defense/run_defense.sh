#!/bin/bash
#
# run_defense.sh is a script which executes the defense
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_defense.sh INPUT_DIR OUTPUT_FILE
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_FILE - file to store classification labels
#

INPUT_DIR=$1
OUTPUT_FILE=$2

python defense.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="${OUTPUT_FILE}" \
  --checkpoint_path="model_ckpts" \
  --vote_models=7,8,9,10 \
  --thresh_models=54,55,69,70,71 \
  --back_off_thresh=0.7 \
  --num_wrong=1 \
  --wt_pow=10 \
  --wt_mod=-1 \
  --hard_models=7,8,9,10,43,44,45,46

