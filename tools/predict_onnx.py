import os
import sys
import cv2
import time
import torch
import imutils
import pathlib
import warnings
import onnxruntime


warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))


class Cropper:
    def __init__(self, model_path, mode='torch'):
        from utils.crop_util import load_cropper
        assert os.path.exists(model_path)
        self.mode = mode
        self.model = load_cropper(model_path, self.mode)

    def infer(self, path, folder_des, height=400.0, is_save=False):
        from utils.crop_util import crop, cropping_post_process
        orig = cv2.imread(path)

        start = time.time()
        mask = crop(self.model, orig, mode=self.mode)
        print('Cropping time:', time.time() - start)

        mask = mask.convert("L")
        result = cropping_post_process(orig, mask, height,
                                       folder_des, path.split('/')[-1],
                                       is_save=is_save)
        return result


class Detector:
    def __init__(self, path, mode, post_p_thre=0.7, unclip_ratio=2.5):
        from utils.util import load_detector
        from post_processing import get_post_processing

        assert os.path.exists(path)

        self.model, post_process_config = load_detector(path, device='cpu', mode='onnx')
        post_process_config['args']['unclip_ratio'] = unclip_ratio

        self.mode = mode
        self.post_process = get_post_processing(post_process_config)
        self.post_process.box_thresh = post_p_thre

    def infer(self, img_ori, short_size: int = 512):
        from utils.util import detect

        start = time.time()
        boxes_list, score_list = detect(net=self.model,
                                        item=img_ori,
                                        short_size=short_size,
                                        device='cpu',
                                        post_processor=self.post_process,
                                        mode=self.mode)
        print('Detecting time:', time.time() - start)

        return boxes_list, score_list

    @staticmethod
    def show(input_img, boxes_list, height=800, is_vis=True):
        from utils.util import draw_bbox
        img = draw_bbox(input_img[:, :, ::-1], boxes_list, is_sorted=True)
        img_show = imutils.resize(img, height=height)
        if is_vis:
            cv2.imshow('Img', img_show)
            cv2.waitKey(0)

        return img


if __name__ == '__main__':
    MODE = 'onnx'
    detector_path = 'D:/OCR/Localization/text_localization/models/detector.onnx'  # path to detecting model
    cropper_path = '../cropper.onnx'  # path to cropping model
    # img_path = 'D:/OCR/real_img/Images/IMG_20210927_092834.jpg'  # path to receipt image

    onnx_detector = Detector(detector_path, mode=MODE, post_p_thre=0.7, unclip_ratio=2.5)
    onnx_cropper = Cropper(cropper_path, mode=MODE)

    org_folder = 'D:/OCR/real_img/Images/'
    save_folder = 'D:/OCR/predict_result2/'
    for i, file in enumerate(os.listdir(org_folder)[27:28]):
        img_path = org_folder + file
        # img_path = 'D:/OCR/real_img/Images/IMG_7592.jpg'
        print(i + 1, end=' ')

        IMG = onnx_cropper.infer(img_path, save_folder, is_save=False)
        boxes_LST, score_LST = onnx_detector.infer(IMG)

        # Show the result
        RS = onnx_detector.show(IMG, boxes_LST, is_vis=True)

        # save_path = save_folder + file
        # cv2.imwrite(save_path, RS)

# vertices err: 18, 28, 88
# pp err: 23, 29, 31, 52, 95, 104
