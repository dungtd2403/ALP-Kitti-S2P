import pandas as pd
import numpy as np
import shutil

import os
seg_dir = "/home/dung/KITTI/ALP/yolov7_fake/yolov7/seg"
file_data = "/home/dung/KITTI/ALP/yolov7_fake/yolov7/predict_with_position/"
seg_dir_filtered = "/home/dung/KITTI/ALP/yolov7_fake/yolov7/seg_filtered"

X_list = []
Y_list = []
Z_list = []
Score_list  = []
Distance_list = []
image_list = []


for files in os.listdir(file_data):
    image_name = str(files)
    label_path = file_data + image_name
    gt_df = pd.read_csv(label_path, sep=" ",header= None)
    gt_df['bbox_Xtl'] = gt_df[1]
    gt_df['bbox_Ytl'] = gt_df[2]
    gt_df['bbox_Xbr'] = gt_df[3]
    gt_df['bbox_Ybr'] = gt_df[4]
    gt_df['location_x'] = gt_df[5]
    gt_df['location_y'] = gt_df[6]
    gt_df['location_z'] = gt_df[7]
    gt_df['class'] = gt_df[0]
    gt_df['distance'] = gt_df[8]
    gt_df['name_file'] = gt_df[9]
    print(label_path)

    for i in range(gt_df.shape[0]):        
        class_name = gt_df.loc[i,'class']       
        x_l1 = gt_df.loc[i,'bbox_Xtl']
        y_l1 = gt_df.loc[i,'bbox_Ytl']
        x_l2 = gt_df.loc[i,'bbox_Xbr']
        y_l2 = gt_df.loc[i,'bbox_Ybr']
        x_gt =  gt_df.loc[i,'location_x']
        y_gt =  gt_df.loc[i,'location_y']
        z_gt =  gt_df.loc[i,'location_z']
        distance = gt_df.loc[i,'distance']
        name_txt =  gt_df.loc[i,'name_file']
        for img in os.listdir(seg_dir):
            img_name = img[3:9] + "_ins" + img[19]
            # print(img[3:9])
            # print(image_name[:6])
            if img_name == name_txt:
                # print('save')
                source = f"{seg_dir}/{img}"
                dest = f"{seg_dir_filtered}/{name_txt}.png"
                shutil.copyfile(source, dest)
                X_list.append(x_gt)
                Y_list.append(y_gt)
                Z_list.append(z_gt)
                score = img[27:32]
                Score_list.append(score)
                image_list.append(name_txt)
                Distance_list.append(distance)
# print(max(X_list))
# print(max(Y_list))
# print(max(Z_list))

  
    # score = image_name[27:32]

    # name_txt = image_name[3:9] + "_ins" + image_name[19]
    # print(name_txt)

    # file_predict = ''
    # txt = pd.read_csv(file_predict + name_txt + '.txt', sep= " ", header=None)
    # txt[5] = s

    
df = pd.DataFrame()
df['x'] = X_list
df['y'] = Y_list
df['z'] = Z_list 
df['score'] = Score_list
df['distance'] = Distance_list
df['image_name'] = image_list

print(df)

df.to_csv('kitti_label.csv')

