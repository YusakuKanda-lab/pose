# Object detection and classification example on Coral and OpenCV with camera

Tested with 

- Raspberry Pi (Raspbian Buster)
- OpenCV3 (python3-opencv)
- TensorFlow Lite 1.14.0 (TensorFlow should be uninstalled)
- Coral USB Accelerator
- Raspberry Pi Camera or Web camera (Logicool)


##  Object detection

```
cd object_detection

bash install_requirements.sh

python3 detect_opencv.py --model models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite --labels models/coco_labels.txt 
```

## Classification

```
cd classification

bash install_requirements.sh

python3 classify_opencv.py --model models/mobilenet_v1_1.0_224_quant_edgetpu.tflite --labels models/labels_mobilenet_quant_v1_224.txt
```
