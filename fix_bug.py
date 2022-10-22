import os 


path_dt = "/home/huynhmink/Desktop/DUNG/kitti-object-eval-python/detect/"
path_gt = "/home/huynhmink/Desktop/DUNG/kitti-object-eval-python/gt_after_change/"


for files in os.listdir(path_gt):
  file_gt_path = path_gt + str(files)
  file_dt_path = path_dt + str(files)

  if not os.path.exists(file_dt_path):
    f = open(file_dt_path, 'a+')
    f.close() 