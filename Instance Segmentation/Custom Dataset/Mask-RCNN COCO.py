import os, json
import skimage.io
import numpy as np
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import visualize, utils

class TrainingConfig(Config):
    NAME = "balloonIS"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 100
    VALIDATION_STES = 5
    IMAGE_MIN_DIM = 32
    IMAGE_MAX_DIM = 2048
    RPN_ANCHOR_SCALES = (64, 128, 256, 512, 1024)
    DETECTION_MIN_CONFIDENCE = 0.9

class BuildDataset(utils.Dataset):
    def __init__(self):
        super().__init__()
        self.modelDir = "logs"
        self.name = "balloonIS"
        self.class_names = ["balloon"]
    
    def load_dataset(self, dataset_dir, subset, annotationFile):
        
        index=1
        for i in self.class_names:
            self.add_class(self.name, index, i)
            index = index+1

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }

        annotations = json.load(open(os.path.join(dataset_dir, annotationFile)))
        annotations = list(annotations.values())
        
        # Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(self.name, image_id=a['filename'], path=image_path, width=width, height=height, polygons=polygons)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != self.name:
            return super(self.__class__, self).load_mask(image_id)
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == self.name:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

class DetectionConfig(TrainingConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class MaskRCNNnet():
    def __init__(self):
        self.class_names = ["balloon"]
        self.modelDir = "logs"
        self.config = TrainingConfig()

    def create(self, dataset_dir, init_with="coco"):
        self.dataset_train = BuildDataset()
        self.dataset_train.load_dataset(dataset_dir, "train",  "via_region_data.json")
        self.dataset_train.prepare()
        
        self.dataset_val = BuildDataset()
        self.dataset_val.load_dataset(dataset_dir, "val",  "via_region_data.json")
        self.dataset_val.prepare()
        
        self.model = modellib.MaskRCNN(mode="training", config=self.config, model_dir=self.modelDir)
        if init_with == "imagenet":
            self.model.load_weights(self.model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            self.model.load_weights("mask_rcnn_coco.h5", by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])    
        elif init_with == "last":
            self.model.load_weights(self.model.find_last(), by_name=True)
    
    def train(self, epHeads, epAll):
        self.model.train(self.dataset_train, self.dataset_val, learning_rate=self.config.LEARNING_RATE, epochs=epHeads, layers='heads')
        self.model.train(self.dataset_train, self.dataset_val, learning_rate=self.config.LEARNING_RATE/10, epochs=epAll, layers='all')
        model_path = os.path.join(self.modelDir, "final_maskRCNN_model.h5")
        self.model.keras_model.save_weights(model_path)
    
    def testBuild(self):
        self.config = DetectionConfig()
        self.model = modellib.MaskRCNN(mode="inference", config=self.config, model_dir=self.modelDir)
        model_path = os.path.join(self.modelDir, "final_maskRCNN_model.h5")
        print("Loading weights from ", model_path)
        self.model.load_weights(model_path, by_name=True)
    
    def test(self, test):
        image = skimage.io.imread(test)
        results = self.model.detect([image], verbose=1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores'])
    
    
maskRCNNnet = MaskRCNNnet()
maskRCNNnet.create("balloon")
maskRCNNnet.train(1,2)
maskRCNNnet.testBuild()
maskRCNNnet.test("sample.jpg")