import os 
import scipy.io
import glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import csv

import joblib
import argparse

from pathlib import Path
from tqdm import tqdm

from fisheye_trans import FisheyeTrans


def arg_parser():
    parser = argparse.ArgumentParser(description='Get some arguments')
#     parser.add_argument('img_dir', type=str, default='.', help='input directory contains raw images')
#     parser.add_argument('yolo_dir', type=str, default='.', help='input yolo directory contains raw images')
#     parser.add_argument('ano_dir', type=str, default='.', help='input ano directory contains raw images')
#     parser.add_argument('out_dir', type=str, help='Output directory')
    parser.add_argument('--image-extensions', required=False, default='jpg,png,jpeg',
                        metavar='"jpg,png,jpeg"',
                        help='Comma-separated list of file types that will be anonymized')
    
    args = parser.parse_args()
#     if not (args.img_dir.endswith('/')):
#         args.img_dir += '/'
#     if not (args.yolo_dir.endswith('/')):
#         args.yolo_dir += '/'
#     if not (args.ano_dir.endswith('/')):
#         args.ano_dir += '/'
#     if not (args.out_dir.endswith('/')):
#         args.out_dir += '/'
    return args

def load_txt_data(txt_path, w_original, h_original):
    lb = []
    for line in open(txt_path).read().strip().split('\n'):
        if not line:
            continue
        info = line.split()
        label_idx = int(float(info[0]))
        xc, yc, width, height, conf= map(float, info[1:])

        if len(lb) == 0:
            lb = [[label_idx, xc, yc, width, height, conf]]
        else:
            lb.append([label_idx, xc, yc, width, height, conf])

    return lb

def fisheye_transform(input_image_path, normal_relative_path, fisheye_img_path, fisheye_pseudo_relative_path, focal_length):
    # init Converter object
    ft = FisheyeTrans(focal_length)
    
    # load image
    img = cv2.imread(str(input_image_path))
    dh, dw, _ = img.shape
    
    dstImg = ft.convert2(img)
    cv2.imwrite("out_test/test.png", dstImg)

    img = cv2.imread(str(input_image_path))
    
    
    lb_d = load_txt_data(yolo_text_path, dw, dh, True)
    lb_s = load_txt_data(ano_text_path, dw, dh)
    
    check_lb_d = [True] * len(lb_d)
    check_lb_s = [True] * len(lb_s)
    
    # output
    lb_new = []
    num_of_lpd = 0
    num_of_f = 0
    
    for i in range(len(lb_s)):
        lb_s_item = lb_s[i]

        bb2_x1 = int((lb_s_item[1] - lb_s_item[3]/2) * dw)
        bb2_y1 = int((lb_s_item[2] - lb_s_item[4]/2) * dh)
        bb2_x2 = int((lb_s_item[1] + lb_s_item[3]/2) * dw)
        bb2_y2 = int((lb_s_item[2] + lb_s_item[4]/2) * dh)

        # if license plate box of yolo
        if lb_s_item[0] > 0:
            if len(lb_new) == 0:
                lb_new = [lb_s_item]
            else:
                lb_new.append(lb_s_item)
            num_of_lpd = num_of_lpd + 1
            
            check_lb_s[i] = False
            continue

        for j in range(len(lb_d)):
            lb_d_item = lb_d[j]
            
            if check_lb_d[j]:
                num_of_f = num_of_f + 1
                if len(lb_new) == 0:
                    lb_new = [lb_d_item]
                else:
                    lb_new.append(lb_d_item)
                check_lb_d[j] = False


            bb1_x1 = int((lb_d_item[1] - lb_d_item[3]) * dw)
            bb1_y1 = int((lb_d_item[2] - lb_d_item[4]) * dh)
            bb1_x2 = int((lb_d_item[1] + lb_d_item[3]) * dw)
            bb1_y2 = int((lb_d_item[2] + lb_d_item[4]) * dh)


            xA = max(bb1_x1, bb2_x1)
            yA = max(bb1_y1, bb2_y1)
            xB = min(bb1_x2, bb2_x2)
            yB = min(bb1_y2, bb2_y2)

            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (bb1_x2 - bb1_x1 + 1) * (bb1_y2 - bb1_y1 + 1)
            boxBArea = (bb2_x2 - bb2_x1 + 1) * (bb2_y2 - bb2_y1 + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou = interArea / float(boxAArea + boxBArea - interArea)

            if iou > 0.3:
                check_lb_s[i] = False

        if check_lb_s[i]:
            # attempt 14: don't add other class during training
#             if len(lb_new) == 0:
#                 lb_new = [[2, lb_s_item[1], lb_s_item[2], lb_s_item[3], lb_s_item[4], lb_s_item[5]]]
#             else:
#                 lb_new.append([2, lb_s_item[1], lb_s_item[2], lb_s_item[3], lb_s_item[4], lb_s_item[5]])
            check_lb_s[i] = False
#     print(len(lb_new), len(lb_s), len(lb_d))
#     print("hello",len(lb_new))
#     print(lb_new)
#     print(lb_new[0][1], lb_new[0][2], lb_new[0][3], lb_new[0][4], lb_new[0][5])
    with open(output_text_path, 'w') as f:
        for i in range(len(lb_new)):
            f.write(str(lb_new[i][0])) 
            f.write(f' {lb_new[i][1]:.6f} {lb_new[i][2]:.6f} {lb_new[i][3]:.6f} {lb_new[i][4]:.6f} {lb_new[i][5]:.6f}\n')
            
    return num_of_lpd, num_of_f
        


def process(img_dir, output_dir, focal_length, file_types):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    assert Path(output_dir).is_dir(), 'Output path must be a directory'

    files = []
    for file_type in file_types:
        files.extend(list(Path(img_dir).glob(f'**/*.{file_type}')))

    for input_image_path in tqdm(files):
        relative_path = input_image_path.relative_to(img_dir)
        (Path(output_dir) / relative_path.parent).mkdir(exist_ok=True, parents=True)

        normal_relative_path = str(relative_path).replace("images", "pseudo_labels")
        normal_text_path = (Path(output_dir) / normal_relative_path).with_suffix('.txt')
        
        fisheye_relative_path = str(relative_path).replace("images", "fisheye_img")
        fisheye_img_path = Path(output_dir) / fisheye_relative_path
        Path(fisheye_img_path).parent.mkdir(parents=True, exist_ok=True)
        
        fisheye_pseudo_relative_path = str(relative_path).replace("images", "fisheye_txt")
        fisheye_text_path = (Path(output_dir) / fisheye_pseudo_relative_path).with_suffix('.txt')
        Path(fisheye_text_path).parent.mkdir(parents=True, exist_ok=True)
        
        Path(output_text_path).parent.mkdir(parents=True, exist_ok=True)
        
        fisheye_transform(input_image_path, normal_relative_path, fisheye_img_path, fisheye_pseudo_relative_path, focal_length)

if __name__ == "__main__":
    option = arg_parser()
    print(option)
    img_dir = "/raid/qc_perception/datasets/anonymizer_annotations/attempt14/raw_img/"
    out_dir = "/raid/qc_perception/datasets/anonymizer_annotations/attempt14/"
    img_extensions = option.image_extensions.split(',')
    focal_length = 100
    process(img_dir, out_dir, focal_length, img_extensions)