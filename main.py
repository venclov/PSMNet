from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models import *
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default='/vol/bitbucket/pv819/sceneflow_data/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=0,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
         batch_size= 3, shuffle= True, num_workers= 8, drop_last=False)

# TestImgLoader = torch.utils.data.DataLoader(
#          DA.myImageFloder(test_left_img[:100],test_right_img[:100],test_left_disp[:100], False), 
#          batch_size= 2, shuffle= False, num_workers= 4, drop_last=False)

writer = SummaryWriter('/vol/bitbucket/pv819/logs_validate/further/')



if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

def train(imgL,imgR, disp_L):
        model.train()

        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
        else:
            disp_true = disp_L

       #---------
        mask = disp_true < args.maxdisp
        mask.detach_()
        #----
        optimizer.zero_grad()
        
        if args.model == 'stackhourglass':
            output1, output2, output3 = model(imgL,imgR)
            output1 = torch.squeeze(output1,1)
            output2 = torch.squeeze(output2,1)
            output3 = torch.squeeze(output3,1)
            loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
        elif args.model == 'basic':
            output_pred, output_var  = model(imgL,imgR)
            print('After model Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            output_pred = torch.squeeze(output_pred,1)
            output_var= torch.squeeze(output_var,1)
            var_weigth = 1
            var = output_var[mask] * var_weigth  
            shape = output_pred.shape
            loss_pred = F.smooth_l1_loss(output_pred[mask], disp_true[mask], reduction='none')
            loss1 = torch.mul(torch.exp(-var), loss_pred)
            loss2 = var
            loss = 1/2 * (loss1 + loss2)
            loss_mean = loss.mean() 

        print('Before mean Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        loss_mean.backward()
        optimizer.step()

        return loss_mean.item()

def validate(imgL,imgR,disp_L):
        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
        else:
            disp_true = disp_L

       #---------
        mask = disp_true < args.maxdisp
        mask.detach_()
        #----
        
        if args.model == 'stackhourglass':
            output1, output2, output3 = model(imgL,imgR)
            output1 = torch.squeeze(output1,1)
            output2 = torch.squeeze(output2,1)
            output3 = torch.squeeze(output3,1)
            loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
        elif args.model == 'basic':
            # print('Before model Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            output_pred, output_var  = model(imgL,imgR)
            # print('Before after Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            output_pred = torch.squeeze(output_pred,1)
            output_var= torch.squeeze(output_var,1)
            var_weigth = 1
            var = output_var[mask] * var_weigth  
            shape = output_pred.shape
            loss_pred = F.smooth_l1_loss(output_pred[mask], disp_true[mask], reduction='none')
            loss1 = torch.mul(torch.exp(-var), loss_pred)
            loss2 = var
            loss = 1/2 * (loss1 + loss2)
            loss_mean = loss.mean() 

        return loss_mean.item()


def test(imgL,imgR,disp_true):

        model.eval()
  
        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()
        #---------
        mask = disp_true < 192
        #----

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16       
            top_pad = (times+1)*16 -imgL.shape[2]
        else:
            top_pad = 0

        if imgL.shape[3] % 16 != 0:
            times = imgL.shape[3]//16                       
            right_pad = (times+1)*16-imgL.shape[3]
        else:
            right_pad = 0  

        imgL = F.pad(imgL,(0,right_pad, top_pad,0))
        imgR = F.pad(imgR,(0,right_pad, top_pad,0))

        with torch.no_grad():
            output3 = model(imgL,imgR)
            output3 = torch.squeeze(output3)
        
        if top_pad !=0:
            img = output3[:,top_pad:,:]
        else:
            img = output3

        if len(disp_true[mask])==0:
           loss = 0
        else:
           loss = F.l1_loss(img[mask],disp_true[mask]) #torch.mean(torch.abs(img[mask]-disp_true[mask]))  # end-point-error

        return loss.data.cpu()

def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    epoch_len = len(TrainImgLoader)
    print(f"Epoch len is {epoch_len} iterations")
    start_full_time = time.time()
    for epoch in range(0, args.epochs):
        print('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)

        # training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss = train(imgL_crop,imgR_crop, disp_crop_L)
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
                
            if batch_idx % 3 == 0:
                # log loss
                writer.add_scalar('Loss/train', loss, batch_idx + epoch_len * epoch)

            # if batch_idx % 5 == 0:
            #     # validate model
            #     # print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            #     with torch.no_grad():
            #         total_val_loss = 0
            #         for batch_idx_v, (imgL_crop_v, imgR_crop_v, disp_crop_L_v) in enumerate(TestImgLoader):
            #             loss_v = validate(imgL_crop_v,imgR_crop_v, disp_crop_L_v)
            #             total_val_loss += loss_v
            #         writer.add_scalar('Loss/validation', total_val_loss / len(TestImgLoader), batch_idx + epoch_len * epoch)
                    # print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')



        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
        #SAVE
        savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
                    'train_loss': total_train_loss/len(TrainImgLoader),
        }, savefilename)

    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))

	#------------- TEST ------------------------------------------------------------
	# total_test_loss = 0
	# for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
	#        test_loss = test(imgL,imgR, disp_L)
	#        print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
	#        total_test_loss += test_loss

	# print('total test loss = %.3f' %(total_test_loss/len(TestImgLoader)))
	# #----------------------------------------------------------------------------------
	# #SAVE test information
	# savefilename = args.savemodel+'testinformation.tar'
	# torch.save({
	# 	    'test_loss': total_test_loss/len(TestImgLoader),
	# 	}, savefilename)


if __name__ == '__main__':
    main()
    
