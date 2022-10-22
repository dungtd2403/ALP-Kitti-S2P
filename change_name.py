import pandas as pd 
import numpy as np
import os 

file_gt = '/home/huynhmink/Desktop/DUNG/kitti-object-eval-python/data_object_label_2/training/label_2/'
path_txt =  '/home/huynhmink/Desktop/DUNG/kitti-object-eval-python/gt_after_change/'

df = pd.DataFrame()

for files in os.listdir(file_gt):

  txt_path = file_gt + str(files)
  txt = pd.read_csv(txt_path, sep=" ", header=None)
  
  txt['name_file'] = ['Giangsama']*txt.shape[0]

  for i in range(txt.shape[0]):
    name_by_index = files + "_ins" + str(i)
    txt.loc[i, "name_file"] = name_by_index

  df = pd.concat([df, txt], axis=0)

print(df)

df = df.rename(columns={0 : 'Class'})

print(df['Class'].value_counts())
print(df['Class'].unique())
print(df.loc[df.Class == "Pedestrian"].shape[0])
print(df.loc[df.Class == "Cyclist"].shape[0])
print(df.loc[df.Class == "Person_sitting"].shape[0])

df.loc[df.Class == "Pedestrian", "Class"] = "Person"
df.loc[df.Class == 'Cyclist', "Class"] = "Person"
df.loc[df.Class == 'Person_sitting', "Class"] = "Person"

print(df.loc[df.Class == "Person"].shape[0])

for files in df['name_file']:

  path_txt_gt = path_txt + str(files[:10])

  Cla = str(np.array(df.loc[df.name_file == files]['Class'])[0])
  truncated = float(df.loc[df.name_file == files][1])
  occluded = float(df.loc[df.name_file == files][2])
  x_min = float(df.loc[df.name_file == files][4])
  y_min = float(df.loc[df.name_file == files][5])
  x_max = float(df.loc[df.name_file == files][6])
  y_max = float(df.loc[df.name_file == files][7])
  x = float(df.loc[df.name_file == files][8])
  y = float(df.loc[df.name_file == files][9])
  z = float(df.loc[df.name_file == files][10])

  f = open(path_txt_gt, "a+")
  f.write("{} {} {} {} {} {} {} {} {} {}\n".format(Cla, truncated, occluded, x_min, y_min, x_max, y_max, x, y, z))
  f.close()
print(df)