import os
import skimage.io
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import visualize

class DetectionConfig(Config):
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 2048
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 128
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50

class MaskRCNNnet():
    def __init__(self):
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
        self.modelDir = "logs"
        self.config = DetectionConfig()

    def create(self, init_with="coco"):
        self.model = modellib.MaskRCNN(mode="inference", config=self.config, model_dir=self.modelDir)
        if init_with == "imagenet":
            self.model.load_weights(model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            self.model.load_weights("mask_rcnn_coco.h5", by_name=True)
    
    def test(self, test):
        image = skimage.io.imread(test)
        results = self.model.detect([image], verbose=1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores'])
    
    
maskRCNNnet = MaskRCNNnet()
maskRCNNnet.create()
maskRCNNnet.test("sample.jpg")