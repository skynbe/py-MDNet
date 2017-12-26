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

np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)

def nchw_image(image):
    # image: PIL.Image
    return np.array([np.asarray(image, dtype=np.float32)]).transpose(0, 3, 1, 2) - 128 # NCHW

def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image, samples, opts['img_size'], opts['padding'], opts['batch_test'])
    for i, regions in enumerate(extractor):
        
        regions = Variable(regions)
        if opts['use_gpu']:
            regions = regions.cuda()
        feat = model(regions, out_layer=out_layer)
        if i==0:
            feats = feat.data.clone()
        else:
            feats = torch.cat((feats,feat.data.clone()),0)
    return feats

def forward_roi_samples(model, images, samples, out_layer='conv3', spat=True):
    model.eval()
    # images = nchw_image(image)
    images = Variable(torch.from_numpy(images).float()).cuda()
    whole_feat = model(images, out_layer=out_layer, spat=spat)
    whole_feat = whole_feat.data.cpu().numpy()
    whole_feat = whole_feat.transpose(0, 2, 3, 1)[0]
    height, width,  _ = whole_feat.shape
    samples = samples * [width, height, width, height]

    pool = np.array([crop_image(whole_feat, sample, img_size=3, valid=True, max_pooling=False) for sample in samples]).transpose(0, 3, 1, 2)
    pool = pool.reshape(pool.shape[0], -1)
    feats = torch.from_numpy(pool).cuda()
    return feats


def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.iteritems():
        lr = lr_base
        for l, m in lr_mult.iteritems():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()
    
    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while(len(pos_idx) < batch_pos*maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand*maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):

        # select pos idx
        pos_next = pos_pointer+batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer+batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx)).cuda()
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx)).cuda()

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0,batch_neg_cand,batch_test):
                end = min(start+batch_test,batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.data[:,1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:,1].clone()),0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            model.train()
        
        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)
        
        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
        optimizer.step()

        #print "Iter %d, Loss %.4f" % (iter, loss.data[0])


def run_mdnet(img_list, init_bbox, gt=None, savefig_dir='', display=False, args=None):

    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list),4))
    result_bb = np.zeros((len(img_list),4))
    result[0] = target_bbox
    result_bb[0] = target_bbox
    result_accuracy = np.zeros(len(img_list))

    # Init model
    model = MDNet(opts['model_path'])
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])
    
    # Init criterion and optimizer 
    criterion = BinaryLoss()
    init_optimizer = set_optimizer(model, opts['lr_init'])
    update_optimizer = set_optimizer(model, opts['lr_update'])

    tic = time.time()
    # Load first image
    image = Image.open(img_list[0]).convert('RGB')
    init_width, init_height = image.size
    
    # Train bbox regressor
    print("Start bbox regressing")
    bbreg_examples = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
                                 target_bbox, 
                                 opts['n_bbreg'], 
                                 opts['overlap_bbreg'], opts['scale_bbreg'])
    
    if args.base:
        bbreg_feats = forward_samples(model, image, bbreg_examples)
        bbreg = BBRegressor(image.size)
        bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
    else:
        bbreg_examples_ratio = bbreg_examples / [init_width, init_height, init_width, init_height]
        bbreg_feats = forward_roi_samples(model, nchw_image(image), bbreg_examples_ratio)
        bbreg = BBRegressor(image.size)
        bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
    print("End bbox regressing")
    
    # Draw pos/neg samples
    pos_examples = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                               target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

    neg_examples = np.concatenate([
                    gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1), 
                                target_bbox, opts['n_neg_init']//2, opts['overlap_neg_init']),
                    gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                                target_bbox, opts['n_neg_init']//2, opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)

    if args.base:
        # Extract pos/neg features
        pos_feats = forward_samples(model, image, pos_examples)
        neg_feats = forward_samples(model, image, neg_examples)
        # feat_dim = pos_feats.size(-1)
    else:
        init_width, init_height = image.size
        pos_examples_ratio = pos_examples / [init_width, init_height, init_width, init_height]
        neg_examples_ratio = neg_examples / [init_width, init_height, init_width, init_height]

        # Extract pos/neg features
        pos_feats = forward_roi_samples(model, nchw_image(image), pos_examples_ratio)
        neg_feats = forward_roi_samples(model, nchw_image(image), neg_examples_ratio)

    # Initial training
    print("Start initial training")
    train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])
    print("End initial training")
    
    # Init sample generators
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans_f'], opts['scale_f'], valid=True)
    pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2, 1.2)
    neg_generator = SampleGenerator('uniform', image.size, 1.5, 1.2, 1.1)

    # Init pos/neg features for update
    pos_feats_all = [pos_feats[:opts['n_pos_update']]]
    neg_feats_all = [neg_feats[:opts['n_neg_update']]]
    
    spf_total = time.time()-tic
    total_tuning_time = 0

    # Display
    savefig = savefig_dir != ''
    if display or savefig: 
        dpi = 80.0
        figsize = (image.size[0]/dpi, image.size[1]/dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image)

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0,:2]),gt[0,2],gt[0,3], 
                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)
        
        rect = plt.Rectangle(tuple(result_bb[0,:2]),result_bb[0,2],result_bb[0,3], 
                linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir,'0000.jpg'),dpi=dpi)
    
    # Main loop
    for i in range(1,len(img_list)):
        tic = time.time()
        # Load image
        image = Image.open(img_list[i]).convert('RGB')
        image_nchw = nchw_image(image)
        width, height = image.size
        
        # Estimate target bbox
        if args.base:
            samples = gen_samples(sample_generator, target_bbox, opts['n_samples'])
            sample_scores = forward_samples(model, image, samples, out_layer='fc6')
        else:
            samples = gen_samples(sample_generator, target_bbox, opts['n_samples'])
            samples_ratio = samples / [width, height, width, height]
            samples_feats = forward_roi_samples(model, image_nchw, samples_ratio)
            sample_scores = model(samples_feats, in_layer='fc4').data
        
        top_scores, top_idx = sample_scores[:,1].topk(5)
        top_idx = top_idx.cpu().numpy()
        
        target_score = top_scores.mean()
        target_bbox = samples[top_idx].mean(axis=0)

        success = target_score > opts['success_thr']
        
        # Expand search area at failure
        if success:
            sample_generator.set_trans_f(opts['trans_f'])
        else:
            sample_generator.set_trans_f(opts['trans_f_expand'])

        # Bbox regression
        if success:
            if args.base:
                bbreg_samples = samples[top_idx]
                bbreg_feats = forward_samples(model, image, bbreg_samples)
                bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
                bbreg_bbox = bbreg_samples.mean(axis=0)
            else:
                bbreg_samples = samples[top_idx]
                bbreg_samples_ratio = bbreg_samples / [width, height, width, height]
                bbreg_feats = forward_roi_samples(model, image_nchw, bbreg_samples_ratio)
                bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
                bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox
        
        # Copy previous result at failure
        if not success:
            target_bbox = result[i-1]
            bbreg_bbox = result_bb[i-1]
        
        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox
        result_accuracy[i] = overlap_ratio(bbreg_bbox, gt[i])[0]

        # Data collect
        if success:
            # Draw pos/neg samples
            pos_examples = gen_samples(pos_generator, target_bbox, 
                                       opts['n_pos_update'],
                                       opts['overlap_pos_update'])
            neg_examples = gen_samples(neg_generator, target_bbox, 
                                       opts['n_neg_update'],
                                       opts['overlap_neg_update'])

            if args.base:
                # Extract pos/neg features
                pos_feats = forward_samples(model, image, pos_examples)
                neg_feats = forward_samples(model, image, neg_examples)
            else:
                pos_examples_ratio = pos_examples / [width, height, width, height]
                neg_examples_ratio = neg_examples / [width, height, width, height]
                
                pos_feats = forward_roi_samples(model, image_nchw, pos_examples_ratio)
                neg_feats = forward_roi_samples(model, image_nchw, neg_examples_ratio)
                
            pos_feats_all.append(pos_feats)
            neg_feats_all.append(neg_feats)
            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        tuning_time = time.time()
        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'],len(pos_feats_all))
            pos_data = torch.cat(pos_feats_all[-nframes:])
            neg_data = torch.cat(neg_feats_all)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])
        
        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.cat(pos_feats_all)
            neg_data = torch.cat(neg_feats_all)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])
        
        spf = time.time()-tic
        spf_total += spf
        current_tuning_time = time.time() - tuning_time
        total_tuning_time += current_tuning_time
        
        # Display
        display_bboxes(savefig_dir, image, [bbreg_bbox], gt[i], i, display, ["#ff0000", "#0000ff"])
        # display_bboxes(savefig_dir, image, samples[top_idx], gt[i], i, display, ["#ff0000", "#0000ff"])
    

        if gt is None:
            print "Frame %d/%d, Score %.3f, Time %.3f" % \
                (i, len(img_list), target_score, spf)
        else:
            print "Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f / %.3f, Frame %.3f" % \
                (i, len(img_list), overlap_ratio(gt[i],result_bb[i])[0], target_score, spf, current_tuning_time, target_bbox[0])

    fps = len(img_list) / spf_total
    overlap_thres = np.arange(20)/20.
    success_curve = np.array([np.mean(result_accuracy>t) for t in overlap_thres])
    np.savetxt('./result/curve_{}_{:.3f}.txt'.format(args.seq, np.mean(success_curve)), success_curve, delimiter='\n', fmt='%.3f')
    np.savetxt('./result/acc_{}.txt'.format(args.seq), result_accuracy, delimiter='\n', fmt='%.3f')
    
    print("Total Accuracy: %.4f, Success Ratio: %.4f (%.4f fps), tracking time/image: %.4f" % (result_accuracy.mean(), 
                                                              np.mean(success_curve), fps, (spf_total-total_tuning_time)/len(img_list)))
    return result, result_bb, fps, spf_total, total_tuning_time, len(img_list)


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
                             linewidth=1, edgecolor=edgecolor, zorder=1, fill=False)
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
            

def run_tracker(args):
    # Generate sequence config
    img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)

    # Run tracker
    result, result_bb, fps, spf, tun_time, len_img  = run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display, args=args)
    
    # Save result
    res = {}
    res['res'] = result_bb.round().tolist()
    res['type'] = 'rect'
    res['fps'] = fps
    print(result_path)
    json.dump(res, open(result_path, 'w'), indent=2)
    return spf, tun_time, len_img


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')
    parser.add_argument('--base', nargs='?', type=bool, default=False,
                        help='Use ROI Pooling Layer')
    
    args = parser.parse_args()
    print(args)
    assert(args.seq != '' or args.json != '')
    if args.seq == 'all':
        # seqs = list(filter(lambda x: os.path.isdir(os.path.join('../dataset/OTB', x)), os.listdir('../dataset/OTB')))
        seqs = ['Human4', 'Crowds', 'Soccer', 'Car4', 'Bolt', 'Tiger2', 'Deer', 'Woman', 'ClifBar', 'Bird1', 'Panda', 'BlurBody', 'Sylvester', 'David', 'Ironman', 'Liquor', 'Trellis', 'Shaking', 'Human6', 'Singer2', 'BlurCar2', 'Freeman4', 'Matrix', 'BlurOwl', 'DragonBaby', 'Skating2', 'RedTeam', 'Jumping', 'Girl', 'Skiing', 'CarScale', 'Diving', 'Biker', 'Couple', 'Dudek', 'Surfer', 'Human3', 'Football', 'MotorRolling', 'Jump', 'Box', 'Car1', 'Walking2', 'BlurFace', 'Skating1', 'CarDark', 'Human9', 'Walking', 'Basketball']
        total_spf = 0
        total_tun = 0
        total_len = 0
        for seq in seqs:
            args.seq = seq
            spf, tun_time, len_img = run_tracker(args)
            total_spf += spf
            total_tun += tun_time
            total_len += len_img
        print("Total : %.1f images, : %.4f sec, ( :%.4f fps), tracking_time/image : %.4f " % (total_len, total_spf, total_len/total_spf, (total_spf - total_tun)/total_len))
            
    else:
        run_tracker(args)
