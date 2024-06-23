#!/usr/bin/env python3
""" Object Detection """

from tensorflow import keras as K


class Yolo:
    """ Use the Yolo v3 algorithm to perform object detection """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ Use the Yolo v3 algorithm to perform object detection """
        self.model = K.models.load_model(filepath=model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [l.split("\n")[0] for l in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
