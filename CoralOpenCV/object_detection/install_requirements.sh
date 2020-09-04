#!/bin/bash
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get TF Lite model and labels
MODEL_DIR="${SCRIPTPATH}/models"
TEST_DATA_URL=https://github.com/google-coral/edgetpu/raw/master/test_data
mkdir -p "${MODEL_DIR}"
(cd "${MODEL_DIR}"
curl -OL "${TEST_DATA_URL}/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite" \
     -OL "${TEST_DATA_URL}/ssd_mobilenet_v2_coco_quant_postprocess.tflite" \
     -OL "${TEST_DATA_URL}/coco_labels.txt")

