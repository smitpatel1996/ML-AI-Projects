import tensorflow as tf
from tensorflow import keras
from model import saveYOLOv3Model
from infer import inferObjects
from datetime import datetime

class Yolov3Net():
    def __init__(self):
        self.weightsFile = "yolov3.weights"
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
        
    def create(self, filename):
        self.model = saveYOLOv3Model.perform(None, self.weightsFile)
        self.model.save(filename)
    
    def test(self, model, test):
        self.model = model
        self.model.summary()
        now = datetime.now().strftime("%H:%M:%S")
        print("Current Time =", now)
        inferObjects.perform(None, self.model, self.class_names, test, 0.7, 0.3)
        now = datetime.now().strftime("%H:%M:%S")
        print("Current Time =", now)
    
yoloNet = Yolov3Net()
yoloNet.test(keras.models.load_model("yoloCOCO.h5"), "sample.jpg")