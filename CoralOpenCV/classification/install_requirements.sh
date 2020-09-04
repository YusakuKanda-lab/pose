#!/bin/bash
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get TF Lite model and labels
MODEL_DIR="${SCRIPTPATH}/models"
TEST_DATA_URL=https://github.com/google-coral/edgetpu/raw/master/test_data
mkdir -p "${MODEL_DIR}"
(cd "${MODEL_DIR}"
curl -OL "${TEST_DATA_URL}/mobilenet_v1_1.0_224_quant.tflite" \
     -OL "${TEST_DATA_URL}/mobilenet_v1_1.0_224_quant_edgetpu.tflite"
curl -O https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip
unzip mobilenet_v1_1.0_224_quant_and_labels.zip labels_mobilenet_quant_v1_224.txt)
