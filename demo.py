import kitti_common as kitti
from eval_alp import get_official_eval_result
# from eval import get_official_eval_result
def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]
det_path = "/home/huynhmink/Desktop/DUNG/kitti-object-eval-python/detect/"
dt_annos = kitti.get_label_annos(det_path)
gt_path = "/home/huynhmink/Desktop/DUNG/kitti-object-eval-python/gt_after_change/"
# gt_split_file = "/path/to/val.txt" # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
# val_image_ids = _read_imageset_file(gt_split_file)
gt_annos = kitti.get_label_annos(gt_path)
print(get_official_eval_result(gt_annos, dt_annos, 2)) # 6s in my computer
# # print(get_coco_eval_result(gt_annos, dt_annos, 0)) # 18s in my computer