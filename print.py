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



parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seq', default='', help='input seq')
parser.add_argument('-j', '--json', default='', help='input json')
parser.add_argument('-f', '--savefig', action='store_true')
parser.add_argument('-d', '--display', action='store_true')
parser.add_argument('--base', nargs='?', type=bool, default=False,
                    help='Use ROI Pooling Layer')

args = parser.parse_args()
print(args) 

# Generate sequence config
# img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)

# Run tracker
# result, result_bb, fps, spf, tun_time, len_img  = run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display, args=args)

# Save result
res = {}
res['res'] = result_bb.round().tolist()
res['type'] = 'rect'
res['fps'] = fps
print(result_path)

# json.dump(res, open(result_path, 'w'), indent=2)
# return spf, tun_time, len_img