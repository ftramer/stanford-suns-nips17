#!/bin/bash
set -xe

NAME=$1

TMP_DIR="/scr/yis/submit_final/tmp"
DATA_DIR="/scr/yis/data_for_submit"
OUT_FILE="/scr/yis/submit_final/submit_${NAME}.zip"

rm -rf "${TMP_DIR}"
mkdir "${TMP_DIR}"
cp metadata.json "${TMP_DIR}"/metadata.json
cp run_defense.sh "${TMP_DIR}"
cp *.py "${TMP_DIR}"
cp -R nets "${TMP_DIR}"
cp -R preprocessing "${TMP_DIR}"
cp -R "${DATA_DIR}" "${TMP_DIR}"/model_ckpts

cd "${TMP_DIR}"
zip "${OUT_FILE}" *.py metadata.json model_ckpts/* nets/* preprocessing/* run_defense.sh
