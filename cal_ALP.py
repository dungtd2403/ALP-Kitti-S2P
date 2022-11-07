import io as sysio
import time
import numba
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
# from Preprocess import clean_data, _prepare_data


def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    # sort form low -> high
    scores.sort()
    # rearrange from high -> low
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    # print(len(thresholds), len(scores), num_gt)
    return thresholds


def clean_data(label_dir,difficulty):
    df = pd.read_csv(label_dir, delimiter=',', header= 0)
    num_valid_det = 0
    score_list = []
    num_valid_gt = 0
    for ind in df.index:
        instance_name = df['image_name'][ind]
        grount_truth_dis = df['distance'][ind]
        error = df['error'][ind]
        score = df['score'][ind]
        num_valid_gt += 1
        score_list.append(score)
        if error < difficulty:
            num_valid_det += 1
            
            
    return score_list,num_valid_gt, num_valid_det
    
# def get_ALP(recall_threshold, num_valid_det):
#     sums = 0
#     for i in range(0, recall_threshold, 4):
#         sums = sums + prec[..., i]
#     return sums / 11 * 100

def cal_ALP(recall_threshold, label_dir,difficulty):
    df = pd.read_csv(label_dir, delimiter=',', header= 0)
    
    LP_sum = 0
    LP_list = []
    loop_count = 1
    for i in range(0, len(recall_threshold),4):
        tp = 0
        fp = 0
        print(i)
        # print('index_recall',i)
        # print(recall_threshold[i+4])
        if i > 36:
            break
        print(recall_threshold[i+4])
        for ind in df.index:
            instance_name = df['image_name'][ind]
            grount_truth_dis = df['distance'][ind]
            error = df['error'][ind]
            score = df['score'][ind]
            # print("score",score)
            
            
            if i > 36:
                break
            # if recall_threshold[i] <= score < recall_threshold[i+4]:
            #     # print("r1",i)
            #     # print("r2",i+4)
            #     if error <= difficulty:
            #         tp += 1
            #     if error > difficulty:
            #         fn += 1
            # print(i + 1)
            
            if  score >= recall_threshold[i+4]:
                # print("r1",i)
                # print("r2",i+4)
                if error <= difficulty:
                    tp += 1
                if error > difficulty:
                    fp += 1
        LP = tp/(tp + fp)
        LP_sum = LP_sum + LP
        print(f'localization precison = {LP}')
        print('loop_number',loop_count)
        loop_count +=1
        
        
        
    # print('loop_number',loop_count)
    # LP_list.append(LP)
    # LP_sum = sum(LP_list)
    ALP = (LP_sum / 11) * 100
    return ALP
        



if __name__ == "__main__":
    # det_path = "/home/dung/KITTI/ALP/yolov7_fake/yolov7/predict_with_position"
    # gt_path = "/home/dung/KITTI/ALP/ALP-Kitti-S2P/gt_after_change/"
    label_dir = "/home/dung/KITTI/ALP/IVSR_S2P/ivsr-s2p/kitti_evaluation.csv"
    # label_dir = "/home/dung/KITTI/ALP/IVSR_S2P/ivsr-s2p/kitti_evaluation_cp9.csv"
    difficulty = [0.5,1,2,5]
    
    current_class = 2
    # difficultys=0 


    # dt_annos = kitti.get_label_annos(det_path)
    # gt_annos = kitti.get_label_annos(gt_path)
    # scores = [1,3,2,5,2,3,1,3,6,8,2,7,9]
    # scores = np.random.rand(7481,1)
    # print(scores.shape)
    
    for dif in difficulty:
        score_list,num_valid_gt, num_valod_det = clean_data(label_dir,dif)
        score = np.array(score_list)
        score.sort()
        scores = score[::-1]
        # print(scores)
        thresholds = get_thresholds(scores=scores, num_gt= num_valid_gt)
        # print(len(thresholds))
        # print(thresholds)
        # threshold = thresholds[::-1]
        print('threshold',thresholds)
        ALP = cal_ALP(thresholds, label_dir, dif)
        print(f'ALP with error < {dif} = {ALP}')
