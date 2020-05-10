# pip install opencv-contrib-python
# pip install flow_vis
import cv2
import glob
import os
from tqdm import tqdm
import numpy as np
import flow_vis



def save_flow(input_path, ref_path, flow_path):
    inputs = glob.glob(os.path.join(input_path, '*.png'))

    for i, input in enumerate(tqdm(inputs)):
        img_i = cv2.imread(input)
        img_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2GRAY)
        img_r = cv2.imread(os.path.join(ref_path, input.split('/')[-1]))
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        tmp_flow = compute_flow(img_i, img_r)
        # tmp_flow = ToImg(tmp_flow)

        if not os.path.exists(os.path.join(flow_path, 'color')):
            # os.makedirs os.mkdir
            os.makedirs(os.path.join(flow_path, 'color'))
        tmp_flow_color = flow_vis.flow_to_color(tmp_flow, convert_to_bgr=False)
        cv2.imwrite(os.path.join(flow_path, 'color', input.split('/')[-1]), tmp_flow_color)

        if not os.path.exists(os.path.join(flow_path, 'u')):
            # os.makedirs os.mkdir
            os.makedirs(os.path.join(flow_path, 'u'))
        cv2.imwrite(os.path.join(flow_path, 'u', input.split('/')[-1]), tmp_flow[:, :, 0])

        if not os.path.exists(os.path.join(flow_path, 'v')):
            # os.makedirs os.mkdir
            os.makedirs(os.path.join(flow_path, 'v'))
        cv2.imwrite(os.path.join(flow_path, 'v', input.split('/')[-1]), tmp_flow[:, :, 1])
    print('complete:' + flow_path)
    return

def compute_flow(input, ref):
    # Compute the Farneback optical flowwFarne
    # flow = cv2.calcOpticalFloback(ref, input, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute the TV-L1 optical flow
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    raw_flow = TVL1.calc(input, ref, None)

    assert raw_flow.dtype == np.float32

    return raw_flow

def ToImg(raw_flow, bound=15):
    '''
        this function scale the input pixels to 0-255 with bi-bound

        :param raw_flow: input raw pixel value (not in 0-255)
        :param bound: upper and lower bound (-bound, bound)
        :return: pixel value scale from 0 to 255
    '''
    flow = raw_flow
    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0


    return flow


if __name__ == '__main__':
    input_path = './input_'
    ref_path = './ref_'
    flow_path = './flow'
    save_flow(input_path, ref_path, flow_path)

# https://blog.csdn.net/xwmwanjy666/article/details/102731645
