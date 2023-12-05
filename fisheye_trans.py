import numpy as np
import cv2
import os
import math

class FisheyeTrans():
    """
    rf = f*atan(rc/f)
    theta = atan(rc/f)
    """
    def __init__(self, fl=90):
        self.focal_length = fl
        
    def convert_tangential(self, img, q):
        h, w = img.shape[0], img.shape[1]
        dstImg = np.zeros((h+w, w+w, img.shape[2]), np.uint8)

        ux, uy = w/2, h/2 # principle point
        
        for i in range(h):
            for j in range(w):
                dx, dy = j - ux, i - uy
                dx = dx / (w + h)
                dy = dy / (w + h)
                
                x_d = dx / 2 / (q * dy**2 + dx**2 * q) * (1 - math.sqrt(1 - 4 * q * dy**2 - 4 * dx**2 * q))
                y_d = dy / 2 / (q * dy**2 + dx**2 * q) * (1 - math.sqrt(1 - 4 * q * dy**2 - 4 * dx**2 * q))
                
                x_d *= w + h
                y_d *= w + h
                x_d += ux
                y_d += uy
  
                
                dstImg[int(y_d)][int(x_d)] = img[i][j]

        return dstImg
        
    def convert_radial(self, img, k_1, k_2):
        h, w = img.shape[0], img.shape[1]
        dstImg = np.zeros((h + w, h + w, img.shape[2]), np.uint8)

        ux, uy = w/2, h/2 # principle point
        
        for i in range(h):
            for j in range(w):
                dx, dy = j - ux, i - uy
                dx = dx / (h + w)
                dy = dy / (h + w)
                
                radius = np.sqrt(dx**2 + dy**2)
                m_r = 1 + k_1 * radius + k_2*radius**2 # + k_3*radius**6 # radial distortion model
                
                # apply the model 
                x = dx * m_r 
                y = dy * m_r
                
                # reset all the shifting
                x = int(x * (h + w) + ux)
                y = int(y * (h + w) + uy)
                
                dstImg[y][x] = img[i][j]

        return dstImg
        
    def convert_cv2(self, img):
        h, w = img.shape[0], img.shape[1]
        dstImg = np.zeros(img.shape, np.uint8)
        print("dstImg.shape=", dstImg.shape)
        ux, uy = w/2, h/2 # principle point
        
        for i in range(h):
            for j in range(w):
                dx, dy = j - ux, i - uy
                rc = np.sqrt((dx**2 + dy**2))
                theta = np.arctan2(rc, self.focal_length)
                gamma = np.arctan2(dx, dy)
                rf = self.focal_length * theta
                xf = rf * np.sin(gamma)
                yf = rf * np.cos(gamma)
                
                x = int(xf + ux)
                y = int(yf + uy)
                
                dstImg[y][x] = img[i][j]

        return dstImg
    
    def convert_point_cv2(self,  in_x, in_y, w, h):
        ux, uy = w/2, h/2 # principle point
        
 
        dx, dy = in_x - ux, in_y - uy
        rc = np.sqrt((dx**2 + dy**2))
        theta = np.arctan2(rc, self.focal_length)
        gamma = np.arctan2(dx, dy)
        rf = self.focal_length * theta
        xf = rf * np.sin(gamma)
        yf = rf * np.cos(gamma)

        x = int(xf + ux)
        y = int(yf + uy)
                

        return x, y
    
    def convert2(self, img):
        h, w = img.shape[0], img.shape[1]
        dstImg = np.zeros(img.shape, np.uint8)
        ux, uy = w/2, h/2 # principle point
        
        for i in range(h):
            for j in range(w):
                dx, dy = j - ux, i - uy
                
                # normalized (+-1, +-1)
                dx = dx / (w/2)
                dy = dy / (h/2)
                
                # transformation
                x_ = dx * np.sqrt(1 - dy**2 / 2)
                y_ = dy * np.sqrt(1 - dx**2 / 2)
                
                rc = np.sqrt((x_**2 + y_**2))
                
                x__ = x_ * np.exp(-1*rc**2 / 4)
                y__ = y_ * np.exp(-1*rc**2 / 4)
                
                # invert
                x = int(x__ * (w/2) + ux)
                y = int(y__ * (h/2) + uy)
                
                
                dstImg[y][x] = img[i][j]
                
        return dstImg
    
    def convert_point(self, in_x, in_y, w, h):
        ux, uy = w/2, h/2 # principle point
        

        dx, dy = in_x - ux, in_y - uy

        # normalized (+-1, +-1)
        dx = dx / (w/2)
        dy = dy / (h/2)

        # transformation
        x_ = dx * np.sqrt(1 - dy**2 / 2)
        y_ = dy * np.sqrt(1 - dx**2 / 2)

        rc = np.sqrt((x_**2 + y_**2))

        x__ = x_ * np.exp(-1*rc**2 / 4)
        y__ = y_ * np.exp(-1*rc**2 / 4)

        # invert
        x = int(x__ * (w/2) + ux)
        y = int(y__ * (h/2) + uy)
                
        return x, y
    
    def iou_cal(self, bb1, bb2):
        bb1_x1 = bb1[0]
        bb1_y1 = bb1[1]
        bb1_x2 = bb1[2]
        bb1_y2 = bb1[3]
        
        bb2_x1 = bb2[0]
        bb2_y1 = bb2[1]
        bb2_x2 = bb2[2]
        bb2_y2 = bb2[3]
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
        
        return iou
    

if __name__ == "__main__":
    ft = FisheyeTrans()
    
#     in_img_path = "/raid/qc_perception/datasets/anonymizer_annotations/attempt13/WiderFace/val/images/0_Parade_marchingband_1_1004.jpg"
    in_img_path = "out_test/cropped.jpg"
    cut_img = "Wider-360/Val/0--Parade/0_Parade_marchingband_1_1004_fisheye_1.jpg"
    original = "/raid/qc_perception/anonymizer_annotations/attempt13/WiderFace/val/0_Parade_marchingband_1_1004.jpg"
    out_path = "out1.jpg"
    img = cv2.imread(in_img_path)
    print(img.shape)
    
    dstImg = ft.convert2(img)
    cv2.imwrite(out_path, dstImg)
        