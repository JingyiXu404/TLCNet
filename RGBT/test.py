import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import platform
from time import *
from tqdm import tqdm, trange
import torch.cuda
from dataset_mm import Dataset_Middle_LU,Dataset_RGBDD,NYU_v2_datset,Dataset_CIDIS,Dataset_CIDIS_c3,Dataset_CIDIS_c3_nogt
from torchvision.transforms import transforms
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
import argparse
import warnings
import psnr
import math
import cv2
from utils import *

warnings.filterwarnings(action='ignore')
matplotlib.use('agg')


from TLCNet import TLCNet as TLCNet

def padding_size(x, d):
    x = x + 2
    return math.ceil(x / d) * d - x
def pad(img):
    h, w = img.shape[2], img.shape[3]
    h_psz = padding_size(h, 4)
    w_psz = padding_size(w, 4)
    padding = torch.nn.ReflectionPad2d((0, w_psz, 0, h_psz))
    img = padding(img)
    return img
def data_process_npy(data_in,times=255):
    data_out=data_in.detach().float().cpu().numpy()
    data_out=np.transpose(data_out,(0,2,3,1))
    data_out = data_out.squeeze()
    return data_out*times
def get_PA(data):
    magnitude, phase = amp_pha(data)  # 做傅里叶变换得到幅度谱和相位谱
    P_data = IFFT_xp(phase)  # 相位谱逆变换
    A_data = IFFT_zm(magnitude)  # 幅度谱逆变换
    return P_data, A_data
def get_overall_run_params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size-train", dest="batch_size_train", default=16, type=int)
    parser.add_argument("--batch-size-test", dest="batch_size_test", default=1, type=int)
    parser.add_argument("--scale", dest="scale", default=16, type=int)
    parser.add_argument("--lr", dest="lr", default=1 * 1e-4, type=float, help="learning rate")
    parser.add_argument("--epoch", dest="EPOCH", default=1000, type=int, help="epochs")
    parser.add_argument("--phase", dest="phase", default='train', type=str, help="train or test")
    parser.add_argument("--patch-size", dest="patch_size", default=256, type=int, help="size of train and test dataset")
    parser.add_argument("--layer", dest="layer", default=5, type=int, help="number of layers")
    parser.add_argument("--layer_Pz", dest="layer_Pz", default=2, type=int, help="number of layers")
    parser.add_argument("--modelname", dest="model_name", default="TLCNet", type=str, help="PxAy or else")
    parser.add_argument('--gpu_id', type=str, default=None, help='train use gpu')  #
    parser.add_argument('--load_pre', type=str, default="cpts/", help='train from checkpoints')
    parser.add_argument('--PSNR', type=str, default='10', help='Current best PSNR')
    parser.add_argument('--testset', dest="testset", type=str, default='CIDIS_200', help='Overlapping of different tiles')

    params = parser.parse_args()
    return params

def cal_multi_loss(lossfn, preds, gt):
    losses = None
    for i, pred in enumerate(preds):
        loss = lossfn(pred, gt)
        if i != len(preds) - 1:
            loss *= (1 / (len(preds) - 1))
        if i == 0:
            losses = loss
        else:
            losses += loss
    return losses

def process_networks(params):
    EPOCH = params.EPOCH
    phase = params.phase
    scale = params.scale
    testname = params.testset
    cpt_path = params.load_pre
    device = torch.device('cuda' if params.gpu_id is not None else 'cpu')
    if params.model_name == "TLCNet":
        model = TLCNet(num_of_layers=params.layer)
    else:
        pass
    if params.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id
        print('USE GPU- ', params.gpu_id)
        model = nn.DataParallel(model).to(device)
        model_path = cpt_path+'x'+str(scale)+'/best_'+testname+'.pth'
        print(model_path)
        if (model_path is not None):
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path)
                new_checkpoint = {}
                for k in checkpoint.keys():
                    new_checkpoint[k] = checkpoint[k]
                model.load_state_dict(new_checkpoint, strict=True)
                print('load model from ', model_path)

    print("loading dataset-------------------------------------")

    root_path_train = '/mnt/ssd1/XJY/GDSR/RGBT/'+str(testname)+'/train/'
    root_path_test = '/mnt/ssd1/XJY/GDSR/RGBT/'+str(testname)+'/test/'
    dr_dataset_test = Dataset_CIDIS_c3(dataset=testname,path=root_path_test, patch=params.patch_size, scale=scale,train=False)
    test_loader_CIDIS = DataLoader(dr_dataset_test, batch_size=params.batch_size_test, num_workers=4, shuffle=False)
    data_loaders = {'test_'+testname: test_loader_CIDIS}#}

    if phase == 'train':
        pass
    elif phase == 'test':
        os.makedirs('Results/' +str(scale)+'_'+params.model_name+'_l'+str(params.layer)+'_'+testname+'/',exist_ok=True)
        with torch.no_grad():
            val_bar = tqdm(data_loaders['test_'+testname])
            batch_ind = 0
            rmse_list = []
            psnr_list = []
            ssim_list = []
            for data in val_bar:
                target, guide, gt = data[0].repeat(1,3,1,1).to(device),data[1].to(device),data[2].to(device)
                if testname=='NYU':
                    minmax = test_minmax[:, batch_ind]
                    minmax = torch.from_numpy(minmax).to(device)
                batch_ind += 1
                model = model.eval()
                h, w = gt.size()[-2:]
                output, output_else = model(target,guide)

                out = output[-1][..., :h, :w]
                if testname =='NYU':
                    rmse = calc_rmse(gt[0,0],out[0,0],minmax)
                    rmse_list.append(rmse.cpu().numpy())
                else:
                    rmse = midd_calc_rmse(gt[0, 0], out[0, 0])
                    rmse_list.append(rmse.cpu().numpy())

                out = data_process_npy(out)
                gt = data_process_npy(gt)
                psnr_list.append(psnr.psnr(gt,out))
                ssim_list.append(psnr.SSIM(gt,out))

                path_output = 'Results/' +str(scale)+'_'+params.model_name+'_l'+str(params.layer)+'_'+testname+'/' + f"{batch_ind:03d}" + '.png'
                cv2.imwrite(path_output, out)

            PSNR = np.mean(psnr_list)
            RMSE = np.mean(rmse_list)
            SSIM = np.mean(ssim_list)

            print("Test: PSNR=%.2fdB RMSE=%.4f SSIM=%.4f" % (PSNR,RMSE,SSIM))


if __name__ == '__main__':
    params = get_overall_run_params()
    process_networks(params)