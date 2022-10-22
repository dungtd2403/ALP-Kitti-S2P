#####################
# Based on https://github.com/hongzhenwang/RRPN-revise
# Licensed under The MIT License
# Author: yanyan, scrin@foxmail.com
#####################
import math

import numba
import numpy as np
from numba import cuda


@numba.jit(nopython=True)
def div_up(m, n):
    return m // n + (m % n > 0)


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def devRotateIoUEval(rbox1, rbox2):
    dis1 = (rbox1[0]**2 + rbox1[1]**2 + rbox1[2]**2)**(1/2)
    dis2 = (rbox2[0]**2 + rbox2[1]**2 + rbox2[2]**2)**(1/2)
    delta_dis = abs(dis1 - dis2)

    return delta_dis

@cuda.jit('(int64, int64, float32[:], float32[:], float32[:], int32)', fastmath=False)
def rotate_iou_kernel_eval(N, K, dev_boxes, dev_query_boxes, dev_iou, criterion=-1):

    threadsPerBlock = 8 * 8

    x, y = cuda.grid(2)


    row_start = cuda.blockIdx.x
    col_start = cuda.blockIdx.y
    tx = cuda.threadIdx.x

    row_size = min(N - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(K - col_start * threadsPerBlock, threadsPerBlock)

    block_boxes = cuda.shared.array(shape=(64 * 3, ), dtype=numba.float32)
    block_qboxes = cuda.shared.array(shape=(64 * 3, ), dtype=numba.float32)

    dev_query_box_idx = threadsPerBlock * col_start + tx
    dev_box_idx = threadsPerBlock * row_start + tx


    if (tx < col_size):
        block_qboxes[tx * 3 + 0] = dev_query_boxes[dev_query_box_idx * 3 + 0]
        block_qboxes[tx * 3 + 1] = dev_query_boxes[dev_query_box_idx * 3 + 1]
        block_qboxes[tx * 3 + 2] = dev_query_boxes[dev_query_box_idx * 3 + 2]
        
    if (tx < row_size):
        block_boxes[tx * 3 + 0] = dev_boxes[dev_box_idx * 3 + 0]
        block_boxes[tx * 3 + 1] = dev_boxes[dev_box_idx * 3 + 1]
        block_boxes[tx * 3 + 2] = dev_boxes[dev_box_idx * 3 + 2]

    cuda.syncthreads()
    if tx < row_size:
        for i in range(col_size):
            offset = row_start * threadsPerBlock * K + col_start * threadsPerBlock + tx * K + i
            dev_iou[offset] = devRotateIoUEval(block_qboxes[i * 3:i * 3 + 3],
                                           block_boxes[tx * 3:tx * 3 + 3])


def rotate_iou_gpu_eval(boxes, query_boxes, device_id=0):
    """rotated box iou running in gpu. 500x faster than cpu version
    (take 5ms in one example with numba.cuda code).
    convert from [this project](
        https://github.com/hongzhenwang/RRPN-revise/tree/master/lib/rotation).
    
    Args:
        boxes (float tensor: [N, 5]): rbboxes. format: centers, dims, 
            angles(clockwise when positive)
        query_boxes (float tensor: [K, 5]): [description]
        device_id (int, optional): Defaults to 0. [description]
    
    Returns:
        [type]: [description]
    """
    box_dtype = boxes.dtype
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    threadsPerBlock = 8 * 8
    cuda.select_device(device_id)
    blockspergrid = (div_up(N, threadsPerBlock), div_up(K, threadsPerBlock))
    
    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes.reshape([-1]), stream)
        query_boxes_dev = cuda.to_device(query_boxes.reshape([-1]), stream)
        iou_dev = cuda.to_device(iou.reshape([-1]), stream)
        rotate_iou_kernel_eval[blockspergrid, threadsPerBlock, stream](
            N, K, boxes_dev, query_boxes_dev, iou_dev)
        iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    return iou.astype(boxes.dtype)