
import argparse
from functools import partial
import re
import time
import os

import numpy as np
from PIL import Image
import cv2
# import svgwrite
# import gstreamer

from myline import LINENotifyBot
import threading
import subprocess
import json

from pose_engine import PoseEngine

EDGES = (
    ('nose', 'left eye'),
    ('nose', 'right eye'),
    ('nose', 'left ear'),
    ('nose', 'right ear'),
    ('left ear', 'left eye'),
    ('right ear', 'right eye'),
    ('left eye', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right elbow'),
    ('right shoulder', 'right hip'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip'),
    ('left hip', 'left knee'),
    ('right hip', 'right knee'),
    ('left knee', 'left ankle'),
    ('right knee', 'right ankle'),
)


    
    
def face_direct(xys):
    color = (0,255,0)
    if   'nose'in xys and 'left eye' in xys and 'right eye' in xys:
        color = (255,0,255)
    return color

       
def draw_pose(img, pose, threshold=0.2):
    xys = {}
    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold: continue
        xys[label] = (int(keypoint.yx[1]), int(keypoint.yx[0]))
        img = cv2.circle(img, (int(keypoint.yx[1]), int(keypoint.yx[0])), 5, (0, 255, 0), -1)

    #print("_--------",xys.get('left hip'))
    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        ax, ay = xys[a]
        bx, by = xys[b]
        color = (0,255,0)

        color = face_direct(xys)

        img = cv2.line(img, (ax, ay), (bx, by), color, 2)




def posecamera(cap):
    
    model = 'models/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite'

    engine = PoseEngine(model)
    # engine = PoseEngine(model)

    width, height = src_size

    isVideoFile = False
    frameCount = 0
    maxFrames = 0

    # VideoCapture init

    print("Open Video Source")
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        pose_ary = {}
        flag = False
        while (True
            ):
            ret, frame = cap.read()
            if ret == False:
                print('can\'t read video source')
                break;

           # frame = cv2.imread('./images/output'+str(i)+'.jpg')
            rgb = frame[:,:,::-1]

#             nonlocal n, sum_fps, sum_process_time, sum_inference_time, last_time
            # image = Image.fromarray(rgb)
            outputs, inference_time = engine.DetectPosesInImage(rgb)
     #      cv2.imshow('TEST PoseNet - OpenCV', frame)

#            print(text_line)

            # crop image
            imgDisp = frame[0:appsink_size[1], 0:appsink_size[0]].copy()
##            cv2.imshow('TEST2 PoseNet - OpenCV', frame)

            if args.mirror == True:
                imgDisp = cv2.flip(imgDisp, 1)

            for pose in outputs:
                draw_pose(imgDisp, pose)

            cv2.imshow('PoseNet - OpenCV', imgDisp)

    except Exception as ex:
        raise ex
    finally:
        cv2.destroyAllWindows()
        cap.release()


def lineNtf(e):

    """
    myline = LINENotifyBot(access_token = token)
    azure_url = "https://hst-face.cognitiveservices.azure.com"
    azure_pg = "hst-test"
    """

    while(True):
        e.wait()
        """
        try:
            output_js = subprocess.check_output(["/usr/local/bin/dotnet" ,"/home/pi/netcoreapp3.1/AzureFaceAuthorizer.dll" ,azure_key, azure_url ,"1", azure_pg,"test.jpg"]).decode()
            output_dic = json.loads(output_js)
            message = output_dic['error']
            if message == None:
                message = output_dic['name']
        except subprocess.CalledProcessError as e:
            message = 'Azure Failed'
        message = 'detected'
        myline.send(message=message,image='test.jpg')
        """
        print('message sent')

def sound(e):
    while(True):
        e.wait()
        subprocess.run(['aplay','file_20131208_Cursor3.wav','-q'])
        print('sound played')






