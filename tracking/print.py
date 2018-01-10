import numpy as np
import os
import sys
import time
import argparse
import json
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

sys.path.insert(0,'../modules')
from sample_generator import *
from data_prov import *
from model import *
from bbreg import *
from options import *
from gen_config import *
from utils import *


def display_bboxes(savefig_dir, image, bboxes, gt, i, display, colors=None):
    savefig = savefig_dir != ''
    if savefig:
        dpi = 80.0
        figsize = (image.size[0] / dpi, image.size[1] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image)
            
        for k, bbox in enumerate(bboxes):
            if colors is None or len(colors)<=k:
                edgecolor = "#ff0000"
            else:
                edgecolor = colors[k]    
            rect = plt.Rectangle(tuple(bbox[:2]), bbox[2], bbox[3],
                             linewidth=3, edgecolor=edgecolor, zorder=1, fill=False)
            ax.add_patch(rect)
            
        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[:2]), gt[2], gt[3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)
        
        if display:
            plt.pause(.01)
            plt.draw()
            
        if savefig:
            fig.savefig(os.path.join(savefig_dir, '%04d.jpg' % (i)), dpi=dpi)
            

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seq', default='', help='input seq')
parser.add_argument('-j', '--json', default='', help='input json')
parser.add_argument('-f', '--savefig', action='store_true')
parser.add_argument('-d', '--display', action='store_true')
parser.add_argument('--base', nargs='?', type=bool, default=False,
                    help='Use ROI Pooling Layer')

args = parser.parse_args()

# Generate sequence config
img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)
print(result_path)

path1 = '../result/mdnet_opt_dragon.json'
path2 = '../result/mdnet_dragon.json'
path3 = '../result/fastmdnet_dragon.json'
path4 = '../result/fastmdnet_align_dragon.json'

        
for i in range(len(img_list)):
    image = Image.open(img_list[i]).convert('RGB')
    
    boxes = []
    for path in [path1, path2, path3, path4]:
        with open(path) as data_file:    
            data = json.load(data_file)
            # print(data['res'][i])
            boxes.append(data['res'][i])
    print(i)
    display_bboxes('./print', image, boxes, gt[i], i, display, ["#ff0000", "#0000ff", "#ff00ff", "#00ffff"])
            

    # Run tracker
# result, result_bb, fps, spf, tun_time, len_img  = run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display, args=args)

# Save result
# res = {}
# res['res'] = result_bb.round().tolist()
# res['type'] = 'rect'
# res['fps'] = fps
# print(result_path)

# json.dump(res, open(result_path, 'w'), indent=2)
# return spf, tun_time, len_img