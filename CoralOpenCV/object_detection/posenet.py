
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




def shadow_text(img, x, y, text, font_size=0.5):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,255), 1, cv2.LINE_AA)
    
    
def face_direct(xys):
    color = (0,255,0)
    if   'nose'in xys and 'left eye' in xys and 'right eye' in xys:
        color = (255,0,255)
    return color

def rightarm_detect(pose):
    rw =pose.keypoints['right wrist']
    rs = pose.keypoints['right shoulder']
  #  ns = pose.keypoints['nose']

    THR = 0.5
    if( rw.score > THR and rs.score > THR):
        return(rw.yx[0] < rs.yx[0])
       
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




def posecamera(e):

    e.clear()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--res', help='Resolution', default='640x480',
                        choices=['480x360', '640x480', '1280x720'])
    parser.add_argument('--videosrc', help='Which video source to use (WebCam number or video file path)', default='0')
    # parser.add_argument('--h264', help='Use video/x-h264 input', action='store_true')
    args = parser.parse_args()

    default_model = 'models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite'
    if args.res == '480x360':
        src_size = (640, 480)
        appsink_size = (480, 360)
        model = args.model or default_model % (353, 481)
    elif args.res == '640x480':
        src_size = (640, 480)
        appsink_size = (640, 480)
        model = args.model or default_model % (481, 641)
    elif args.res == '1280x720':
        src_size = (1280, 720)
        appsink_size = (1280, 720)
        model = args.model or default_model % (721, 1281)

    print('Loading model: ', model)
    engine = PoseEngine(model, mirror=args.mirror)
    # engine = PoseEngine(model)


    last_time = time.monotonic()
    n = 0
    sum_fps = 0
    sum_process_time = 0
    sum_inference_time = 0

    width, height = src_size

    isVideoFile = False
    frameCount = 0
    maxFrames = 0

    # VideoCapture init
    videosrc = args.videosrc
    if videosrc.isdigit():
        videosrc = int(videosrc)
    else:
        isVideoFile = os.path.exists(videosrc)

    print("Start VideoCapture")
    cap = cv2.VideoCapture(videosrc)
        
    
    
    if cap.isOpened() == False:
        print('can\'t open video source \"%s\"' % str(videosrc))
        return;

    print("Open Video Source")
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if isVideoFile:
        maxFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    try:
        i=0
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
            start_time = time.monotonic()
            # image = Image.fromarray(rgb)
            outputs, inference_time = engine.DetectPosesInImage(rgb)
            end_time = time.monotonic()
     #      cv2.imshow('TEST PoseNet - OpenCV', frame)

            n += 1
            sum_fps += 1.0 / (end_time - last_time)
            sum_process_time += 1000 * (end_time - start_time) - inference_time
            sum_inference_time += inference_time
            last_time = end_time
            text_line = 'PoseNet: %.1fms Frame IO: %.2fms TrueFPS: %.2f Nposes %d' % (
                sum_inference_time / n, sum_process_time / n, sum_fps / n, len(outputs)
            )
#            print(text_line)

            # crop image
            imgDisp = frame[0:appsink_size[1], 0:appsink_size[0]].copy()
##            cv2.imshow('TEST2 PoseNet - OpenCV', frame)

            if args.mirror == True:
                imgDisp = cv2.flip(imgDisp, 1)

            shadow_text(imgDisp, 10, 20, text_line)
            for pose in outputs:
                draw_pose(imgDisp, pose)
                if rightarm_detect(pose) and i==0:
                    print("Right DETECTED")
                    cv2.imwrite('testR.jpg',imgDisp)
                    i=1
                    flag = True
                    e.set()

            if i >= 40:
                i = 0
                flag = False
            if flag: i+=1
#            print(i)
            e.clear()
            cv2.imshow('PoseNet - OpenCV', imgDisp)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if isVideoFile:
                frameCount += 1
                # check frame count
                if frameCount >= maxFrames:
                    # rewind video file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frameCount = 0


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

if __name__ == '__main__':
    e = threading.Event()
    e.clear()
    thread_posecamera = threading.Thread(target=posecamera,args=(e,))
#    thread_lineNtf = threading.Thread(target=lineNtf, args=(e,))
    thread_sound = threading.Thread(target=sound, args=(e,))


    thread_posecamera.start()
#    thread_lineNtf.start()
    thread_sound.start()

    print("ALL THREAD STARTED")



