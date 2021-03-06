from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
import cv2
from PIL import Image

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default='./models/pretrained_sceneflow.tar',
                    help='loading model')
parser.add_argument('--leftimg', default= './VO04_L.png',
                    help='load model')
parser.add_argument('--rightimg', default= './VO04_R.png',
                    help='load model')                                      
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--imgdirectory', default= './assets_sceneflow/',
                    help='image directory')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    print('load PSMNet')
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = imgL.cuda()
           imgR = imgR.cuda()     

        with torch.no_grad():
            disp, uncertainty = model(imgL,imgR)

        disp = torch.squeeze(disp)
        uncertainty = torch.squeeze(uncertainty)
        pred_disp = disp.data.cpu().numpy()
        pred_uncertainty = uncertainty.data.cpu().numpy()

        return pred_disp, pred_uncertainty

def single(left_img, right_img, i):
        normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}
        infer_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(**normal_mean_var)])    

        imgL_o = Image.open(left_img).convert('RGB')
        imgR_o = Image.open(right_img).convert('RGB')

        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o) 
       

        # pad to width and hight to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16       
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16                       
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0    

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

        start_time = time.time()
        pred_disp, pred_uncertainty = test(imgL,imgR)
        print('time = %.2f' %(time.time() - start_time))

        
        if top_pad !=0 or right_pad != 0:
            img = pred_disp[top_pad:,:]
            img_unc = pred_uncertainty[top_pad:,:]
        else:
            img = pred_disp
            img_unc = pred_uncertainty 
        
        img = (img*256).astype('uint16')
        img = Image.fromarray(img)

        img.save( "result/" + str(i) + 'disp.png')
        np.savetxt( "result/uncerainty_" + str(i) + 'disp.txt', img_unc)

def previous_main():
        normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}
        infer_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(**normal_mean_var)])

        imgL_o = Image.open(args.leftimg).convert('RGB')
        imgR_o = Image.open(args.rightimg).convert('RGB')

        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o)


        # pad to width and hight to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

        print(imgL.shape)

        start_time = time.time()
        pred_disp = test(imgL,imgR)
        print('time = %.2f' %(time.time() - start_time))


        if top_pad !=0 or right_pad != 0:
            img = pred_disp[top_pad:,:]
        else:
            img = pred_disp

        img = (img*256).astype('uint16')
        img = Image.fromarray(img)
        img.save('Test_disparity.png')



def main():
    img_directory = args.imgdirectory
    for i in range(150):
        if i < 10:
            num = "00" + str(i)
        elif i >= 100:
            num = str(i)
        else:
            num = "0" + str(i)
        left_name = img_directory + 'leftscene_00_0' + str(num) + '.png'
        right_name = img_directory + 'rightscene_00_0' + str(num) + '.png'
        try:
            single(left_name, right_name, i)
        except Exception as e:
            print(e)
            break



if __name__ == '__main__':
   # previous_main()
    main()






