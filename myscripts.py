import json
import os
import gc

def write_txt(mode='train'):
    folder_img = os.path.join('/home/administrator/OCR/DBNet.pytorch/datasets', mode, 'img')
    folder_gt = os.path.join('/home/administrator/OCR/DBNet.pytorch/datasets', mode, 'gt')
    text_file = mode + '.txt'
    text_path = os.path.join('./datasets', text_file)
    file_lst = os.listdir(folder_img)
    file_lst.sort()

    with open(text_path, 'w+') as f:
        for img_file in file_lst:
            if img_file.lower().endswith('.jpg'):
                img_path = os.path.join(folder_img, img_file)
                gt_file = img_file[:-4] + '.txt'
                gt_path = os.path.join(folder_gt, gt_file)
                line = img_path + '\t' + gt_path + '\n'
                f.write(line)


def json2txt(dataset_path):
    org_path = '/home/administrator/Downloads/receipt/dbnet_rs_json'
    folder_lst = [0, 2, 3]
    folder = os.path.join(org_path, str(folder_lst[2]))
    file_lst = os.listdir(folder)
    file_lst.sort()

    for file in file_lst[:40]:
        txt_file = file[:-4] + 'txt'
        json_path = os.path.join(folder, file)

        content = ''

        with open(json_path) as jf:
            obj = json.load(jf)

            for shape in obj['shapes']:
                points = sum(shape['points'], [])  # flatten a list
                annotation = shape['label']

                # label for training
                for coor in points:
                    content += str(coor) + ', '
                content += annotation + '\n'

        gt_path = os.path.join(dataset_path, txt_file)
        with open(gt_path, 'w') as tf:
            tf.write(content)


if __name__ == '__main__':
    write_txt(mode='train')
    write_txt(mode='test')
    # gc.collect()
    # json2txt('./datasets/train/gt')

    # with open('/home/administrator/OCR/DBNet.pytorch/datasets/train/gt/000.txt', encoding='utf-8', mode='r') as f:
    #     i = 0
    #     for i, line in enumerate(f.readlines()):
    #         print(line.strip())
    #         print(i)

    # org_path = '/home/administrator/Downloads/receipt/dbnet_rs_json'
    # folder = os.path.join(org_path, str(0))
    # txt_file = '000.txt'
    # json_path = os.path.join(folder, '000.json')
    #
    # content = ''
    #
    # with open(json_path) as jf:
    #     obj = json.load(jf)
    #
    #     for shape in obj['shapes']:
    #         points = sum(shape['points'], [])  # flatten a list
    #         annotation = shape['label']
    #
    #         # label for training
    #         for coor in points:
    #             content += str(coor) + ', '
    #         content += annotation
    #
    # gt_path = os.path.join(dataset_path, txt_file)
    # with open(gt_path, 'w') as tf:
    #     tf.writeline(content)