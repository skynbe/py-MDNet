from scipy.misc import imresize
import numpy as np
import lycon

def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or 
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

def resize_image(img, img_size=107):
    return lycon.resize(img, width=img_size, height=img_size)


def max_pool(input):
    '''
    :param input: C*H*W
    :return: C*3*3
    '''

    def div(k, d=3):
        a1 = int(round(k / d))
        a2 = int(round(2 * k / d) - a1)
        a3 = int(k - (a1 + a2))
        return a1, a2, a3

    h1, h2, h3 = div(input.shape[1])
    w1, w2, w3 = div(input.shape[2])
    h = [0, h1, h1 + h2, h1 + h2 + h3]
    w = [0, w1, w1 + w2, w1 + w2 + w3]
    if input.shape[1] == 1: h = [0, 0, 0, 1]
    if input.shape[2] == 1: w = [0, 0, 0, 1]

    n = np.zeros((input.shape[0], 3, 3), dtype='float32')
    for i in range(3):
        for j in range(3):
            try:
                n[:, i, j] = np.max(input[:, h[i]:max(h[i + 1], h[i] + 1), w[j]:max(w[j + 1], w[j] + 1)], axis=(1, 2))
            except:
                print("input:", input, input.shape)
                print("h, w, i, j :", h, w, i, j)
                print(input[:, h[i]:max(h[i + 1], h[i] + 1), w[j]:max(w[j + 1], w[j] + 1)])
                assert 1 == 0
    return n

def crop_image(img, bbox, img_size=107, padding=0, valid=False, max_pooling=False):
    x,y,w,h = np.array(bbox,dtype='float32')

    half_w, half_h = w/2, h/2
    center_x, center_y = x + half_w, y + half_h

    padding = padding * img_size/107
    if padding > 0:
        pad_w = padding * w/img_size
        pad_h = padding * h/img_size
        half_w += pad_w
        half_h += pad_h
        
    img_h, img_w, channel = img.shape
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)
    
    if valid:
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)
    
    roi_align = False
    if roi_align:
        def interpolate(array, y, x):
            x_down = int(x)
            x_up = int(np.ceil(x))
            y_down = int(y)
            y_up = int(np.ceil(y))
            return array[y_down, x_down, :]*(1-x)*(1-y)+array[y_down, x_up, :]*x*(1-y)+array[y_up, x_down, :]*y*(1-x)+array[y_up, x_up, :]*x*y
        
        scaled = np.zeros([img_size, img_size, channel])
        y_lin = np.linspace(min_y, max_y, 2*img_size+1)[1::2]
        x_lin = np.linspace(min_x, max_x, 2*img_size+1)[1::2]
        for i, x in enumerate(x_lin):
            for j, y in enumerate(y_lin):
                # x_down = int(x)
                # x_up = int(np.ceil(x))
                # y_down = int(y)
                # y_up = int(np.ceil(y))
                # crop = img[y_down:y_up+1, x_down:x_up+1, :]
                # scaled[j, i, :] = interpolate(crop, y-y_down, x-x_down)
                scaled[j, i, :] = interpolate(img, y, x)
        return scaled
    
    if min_x >=0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        if min_y == max_y:
            if max_y == img_h:
                min_y -= 1
            else:
                max_y += 1
        if min_x == max_x:
            if max_x == img_w:
                min_x -= 1
            else:
                max_x += 1
        
    
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)
        
        cropped = 128 * np.ones((max_y-min_y, max_x-min_x, 3), dtype='uint8')
        try:
            cropped[min_y_val-min_y:max_y_val-min_y, min_x_val-min_x:max_x_val-min_x, :] \
                = img[min_y_val:max_y_val, min_x_val:max_x_val, :]
        except:
            print(min_x, min_y, max_x, max_y, img_w, img_h)
            print(bbox)
            print(cropped.shape)
            assert 1==2
    
    if max_pooling:
        scaled = max_pool(cropped.transpose(2, 0, 1)).transpose(1, 2, 0)
        return scaled
        
    try:
        # scaled = imresize(cropped, (img_size, img_size))
        scaled = lycon.resize(cropped, width=img_size, height=img_size, interpolation=0)
    except:
        print(img_size)
        print(cropped.shape)
        print(bbox)
        scaled = lycon.resize(cropped, width=img_size, height=img_size, interpolation=0)
    return scaled
