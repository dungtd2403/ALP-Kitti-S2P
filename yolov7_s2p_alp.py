import pandas as pd 
import os

folder_predict = '/home/huynhmink/Desktop/DUNG/yolov7_fake/yolov7/predict/'
folder_s2p = '/home/huynhmink/Desktop/DUNG/IVSR_S2P/ivsr-s2p/output_s2p.csv'
folder_detect = '/home/huynhmink/Desktop/DUNG/kitti-object-eval-python/detect/'

df = pd.DataFrame()
df_s2p = pd.read_csv(folder_s2p)

for files in os.listdir(folder_predict):

  data_txt = pd.read_csv(folder_predict + str(files), sep=" ", header=None)

  df = pd.concat([df, data_txt], axis=0)

print(df)
print(df_s2p)

df = df.rename(columns={5 : 'image_name'})
df = df.merge(df_s2p, how='left', on='image_name')

df = df.drop(columns=[6])
df = df[[0, 1, 2, 3, 4, 'x', 'y', 'z', 'score', 'image_name']]

for name in df['image_name']:

  txt = str(name[:6]) + ".txt"
  path_txt = folder_detect + txt

  f = open(path_txt, "a+")
  x_min = float((df.loc[df.image_name == name][1]))
  y_min = float(df.loc[df.image_name == name][2])
  x_max = float(df.loc[df.image_name == name][3])
  y_max = float(df.loc[df.image_name == name][4])
  x = float(df.loc[df.image_name == name]['x'])
  y = float(df.loc[df.image_name == name]['y'])
  z = float(df.loc[df.image_name == name]['z'])
  score = float(df.loc[df.image_name == name]['score'])

  f.write("Person {} {} {} {} {} {} {} {}\n".format(x_min, y_min, x_max, y_max, x, y, z, score))



print(df)