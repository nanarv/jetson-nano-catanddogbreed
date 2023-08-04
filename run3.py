import torch
import cv2
import numpy as np
from PIL import Image
import json
import time
#import line


print("avail:",torch.cuda.is_available())
model = torch.hub.load('ultralytics/yolov5', 'custom', path='bestcatanddogbreed.pt')
model.to("cuda")
vid = cv2.VideoCapture("rtsp://admin:Otmadmin1234@10.10.9.115:554/cam/realmonitor?channel=1&subtype=0")


# https://gist.github.com/syanyong/5fd83ff7d006d4566e115f9dbf203905
def plot_boxes(result_dict, frame):
    for ob in result_dict:
        rec_start = (int(ob['xmin']), int(ob['ymin']))
        rec_end = (int(ob['xmax']), int(ob['ymax']))
        color = (255, 0, 0)
        thickness = 3

        if ob['name'] == 'cat_dragon_li':
            color = (0, 0, 255)
            #cv2.imwrite("./output/output.jpg", frame)
            #line.sendImage("alert", "./output.jpg")
        cv2.rectangle(frame, rec_start, rec_end, color, thickness)
        cv2.putText(frame, "%s %0.4f" % (ob['name'], ob['confidence']), rec_start, cv2.FONT_HERSHEY_DUPLEX, 2, color, thickness)
    return frame

count=0
previous_runtime=0
new_runtime=0
totalfps=0

# read each videoframe from cv2 VideoCapture
while(True):
    ret, frame = vid.read()
    
    frame2= frame[...,::-1] # convert OpenCV's default BGR format to RGB for the model input's veracity to reality
    results = model(frame2)
    
    result_jsons = results.pandas().xyxy[0].to_json(orient="records")
    result_dict = json.loads(result_jsons)
    frame2 = plot_boxes(result_dict, frame)
  
    cv2.imshow('YOLO', frame)
    
    # calculate fps
    new_runtime = time.time()
  
    if count !=0:  # disregard first fps calculation
        totalfps+=fps
    count+=1
  
    fps = float(1/(new_runtime-previous_runtime))
    print('Frames per second : ', fps,'FPS')
  
    previous_runtime = new_runtime

  
    # enable quit command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("avg fps:", totalfps/(count-1))
        break

