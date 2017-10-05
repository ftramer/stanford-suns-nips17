#!/bin/bash
#
# run_attack.sh is a script which executes the attack
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_attack.sh INPUT_DIR OUTPUT_DIR MAX_EPSILON
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_DIR - directory where adversarial images should be written
#   MAX_EPSILON - maximum allowed L_{\infty} norm of adversarial perturbation
#

INPUT_DIR=$1
OUTPUT_DIR=$2
MAX_EPSILON=$3

# --whitebox_train="1,2,3,4,6,7,8,9" \
python attack.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --blackbox_train="10" \#not used
  --whitebox_train="1,2,4,7" \#InceptionV3, InceptionV4, ResNetV2, AdvInceptionV3 
  --test="" \
  --iternum=25 \
  --checkpoint_path=model_ckpts
