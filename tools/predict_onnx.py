import io
import os
import sys
import cv2
import time
import torch
import pathlib
import imutils
import onnxruntime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.crop_util import predict, load_model

import warnings
warnings.filterwarnings('ignore')

__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))
from post_processing import get_post_processing


def preprocess_input(img_ori, short_size):
    img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    img_resize = resize_image(img, short_size)

    img = img_resize.astype(np.float32) / 255
    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img.astype(np.float32), 0), img_resize


def resize_image(img, short_size):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


class Cropper:
    def __init__(self, model_path):
        assert os.path.exists(model_path)
        self.model = load_model(model_path)

    def infer(self, img_path, height=400.0):
        orig = cv2.imread(img_path)

        start = time.time()
        mask = predict(self.model, orig).convert("L")
        print('Cropping time:', time.time() - start)

        mask = mask.resize(tuple(reversed(orig.shape[:2])), Image.LANCZOS)
        gray = np.array(mask)
        ratio = orig.shape[0] / height

        img = imutils.resize(gray, height=int(height))
        _, threshold = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        outline = None
        for c in cnts:
            perimeter = cv2.arcLength(c, True)
            polygon = cv2.approxPolyDP(c, 0.02 * perimeter, True)

            if len(polygon) == 4:
                outline = polygon.reshape(4, 2).astype(int)

        if outline is None:
            result = orig
        else:
            result = four_point_transform(orig, outline * ratio)

        return result


class ONNXDetector:
    def __init__(self, model_path, post_p_thre=0.7, gpu_id=None):
        assert os.path.exists(model_path)
        ort_session = onnxruntime.InferenceSession(model_path)
        self.gpu_id = gpu_id
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")

        self.model = ort_session
        config = {'type': 'SegDetectorRepresenter',
                  'args': {'thresh': 0.3, 'box_thresh': 0.7, 'max_candidates': 1000,
                           'unclip_ratio': 2.5}}
        self.post_process = get_post_processing(config)
        self.post_process.box_thresh = post_p_thre

    def infer(self, img_ori, short_size: int = 512):
        h, w = img_ori.shape[:2]
        batch = {'shape': [(h, w)]}

        input_predict, img_resize = preprocess_input(img_ori, short_size)

        ort_inputs = {self.model.get_inputs()[0].name: input_predict}
        ort_outs = self.model.run(None, ort_inputs)[0]
        tensor_out = torch.from_numpy(ort_outs)
        boxes_list, score_list = self.post_process(batch, tensor_out)
        boxes_list, score_list = boxes_list[0], score_list[0]
        if len(boxes_list) > 0:
            idx = boxes_list.reshape(boxes_list.shape[0], -1).sum(axis=1) > 0
            boxes_list, score_list = boxes_list[idx], score_list[idx]
        else:
            boxes_list, score_list = [], []

        return boxes_list, score_list

    @staticmethod
    def show(input_img, boxes_list, height=800):
        from utils.util import draw_bbox
        img = draw_bbox(input_img[:, :, ::-1], boxes_list)
        img_show = imutils.resize(img, height=height)

        cv2.imshow('Img', img_show)
        cv2.waitKey(0)
        # plt.figure(figsize=(8, 6), dpi=150)
        # plt.imshow(img)


if __name__ == '__main__':
    onnx_path = 'D:/OCR/Localization/text_localization/models/detector.onnx'  # path to detection model
    cropper_path = 'C:/Users/ADMIN/Desktop/u2net.pth'  # path to cropping model
    img_path = 'D:/OCR/real_img/Images/IMG_20210927_092834.jpg'  # path to receipt image

    onnx_detector = ONNXDetector(onnx_path, post_p_thre=0.7, gpu_id=0)
    cropper = Cropper(cropper_path)

    IMG = cropper.infer(img_path)

    start = time.time()
    boxes_LST, score_LST = onnx_detector.infer(IMG)
    print('Detecting time:', time.time() - start)

    # Show the result
    onnx_detector.show(IMG, boxes_LST)
