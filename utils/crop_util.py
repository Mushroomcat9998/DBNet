import cv2
import torch
import imutils
import numpy as np
import onnxruntime
from PIL import Image
from imutils import perspective
from torchvision import transforms

from models import cropper
from data_loader import crop_loader


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


def draw_bbox(input_img, result, color=(0, 0, 255), thickness=2):
    input_img = input_img.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(input_img, [point], True, color, thickness)
    return input_img


def load_cropper(path, mode='torch'):
    net = None
    if mode == 'torch':
        net = cropper.U2NET()
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(path))
            net.to(torch.device("cuda"))
        else:
            net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
    elif mode == 'onnx':
        net = onnxruntime.InferenceSession(path)

    return net


def norm_crop_pred(d):
    mi, ma = 0, 0
    if type(d) == torch.Tensor:
        ma = torch.max(d)
        mi = torch.min(d)
    elif type(d) == np.ndarray:
        ma = np.max(d)
        mi = np.min(d)

    dn = (d - mi) / (ma - mi)
    return dn


def cropping_pre_process(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if 3 == len(label_3.shape):
        label = label_3[:, :, 0]
    elif 2 == len(label_3.shape):
        label = label_3

    if 3 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    elif 2 == len(image.shape) and 2 == len(label.shape):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose(
        [crop_loader.RescaleT(320), crop_loader.ToTensorLab(flag=0)]
    )
    sample = transform({"imidx": np.array([0]), "image": image, "label": label})
    return sample


def get_miss_points(polygon, width):
    miss_left_point = [A[0] for A in polygon if A[0][0] == 0]
    miss_right_point = [A[0] for A in polygon if A[0][0] == width - 1]
    return miss_left_point, miss_right_point


def cropping_post_process(orig, mask, height, folder_path, file, is_save=False):
    from utils.util import intersection_of_2lines

    mask = mask.resize(tuple(reversed(orig.shape[:2])), Image.LANCZOS)
    gray = np.array(mask)
    ratio = orig.shape[0] / height

    img = imutils.resize(gray, height=int(height))
    _, threshold = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    outline = None
    c = contours[0]
    perimeter = cv2.arcLength(c, True)
    polygon_check = cv2.approxPolyDP(c, 0.008 * perimeter, True)
    temp1 = imutils.resize(orig, height=400)
    temp2 = temp1.copy()

    left, right = get_miss_points(polygon_check, img.shape[1])

    if (len(left) == 2 or len(right) == 2) and len(polygon_check) == 5:
        cv2.drawContours(temp2, [polygon_check], -1, (0, 255, 0), 2)
        cv2.imshow('0.009', temp2)

        if len(left) == 2:
            pseudo_points_lst = left
            pseudo_vertical_point, pseudo_horizontal_point = sorted(pseudo_points_lst, key=lambda x: x[1])

            true_points_lst = [A[0] for A in polygon_check if A[0][0] != 0]
            true_points_lst = np.array(true_points_lst)
            top_left = sorted(true_points_lst, key=lambda x: sum(x))[0]
            bot_right = sorted(true_points_lst, key=lambda x: x[1])[-1]
            top_right = [i for i in true_points_lst
                         if not np.array_equal(i, top_left) and not np.array_equal(i, bot_right)][0]

            line1 = np.concatenate([top_left, pseudo_vertical_point])
            line2 = np.concatenate([bot_right, pseudo_horizontal_point])
            miss_point = intersection_of_2lines(line1, line2)
            outline = np.array([top_left, top_right, miss_point, bot_right]).astype(int)

        elif len(right) == 2:
            pseudo_points_lst = right
            pseudo_vertical_point, pseudo_horizontal_point = sorted(pseudo_points_lst, key=lambda x: x[1])
            true_points_lst = [A[0] for A in polygon_check if A[0][0] != img.shape[1] - 1]
            true_points_lst = np.array(true_points_lst)
            top_left = sorted(true_points_lst, key=lambda x: sum(x))[0]
            bot_left = sorted(true_points_lst, key=lambda x: x[1])[-1]
            top_right = [i for i in true_points_lst
                         if not np.array_equal(i, top_left) and not np.array_equal(i, bot_left)][0]

            line1 = np.concatenate([top_right, pseudo_vertical_point])
            line2 = np.concatenate([bot_left, pseudo_horizontal_point])
            miss_point = intersection_of_2lines(line1, line2)
            outline = np.array([top_left, top_right, miss_point, bot_left]).astype(int)

    else:
        polygon = cv2.approxPolyDP(c, 0.035 * perimeter, True)
        cv2.drawContours(temp1, [polygon], -1, (0, 255, 0), 2)
        cv2.imshow('0.02', temp1)
        cv2.drawContours(temp2, [polygon_check], -1, (0, 255, 0), 2)
        cv2.imshow('0.009', temp2)

        if len(polygon) == 4:
            outline = polygon.reshape(4, 2).astype(int)
        else:
            polygon = cv2.approxPolyDP(c, 0.035 * perimeter, True)
            if len(polygon) == 4:
                outline = polygon.reshape(4, 2).astype(int)

    if outline is None:
        result = orig
    else:
        result = perspective.four_point_transform(orig, outline * ratio)

    if is_save:
        from utils.util import save_image
        save_image(folder_path, file, gray)

    return result


def crop(net, item, mode='onnx'):
    sample = cropping_pre_process(item)
    predict = item
    if mode == 'torch':
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs_test = torch.cuda.FloatTensor(
                    sample["image"].unsqueeze(0).cuda().float()
                )
            else:
                inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())

            # d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)
            torch_out = net(inputs_test)[0]
            pred = torch_out[:, 0, :, :]
            predict = norm_crop_pred(pred)

            predict = predict.squeeze()
            predict = predict.cpu().detach().numpy()

        # del d1, d2, d3, d4, d5, d6, d7, pred, predict, predict_np, inputs_test, sample

    elif mode == 'onnx':
        input_predict = np.array(sample["image"].unsqueeze(0).float())
        ort_inputs = {net.get_inputs()[0].name: input_predict}
        ort_outs = net.run(None, ort_inputs)[0]

        pred = ort_outs[:, 0, :, :]
        predict = norm_crop_pred(pred)[0]

    img = Image.fromarray(predict * 255).convert("RGB")
    return img
