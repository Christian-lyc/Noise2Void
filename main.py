import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from measure import compute_measure
from torch.optim import Adam, lr_scheduler
import torchvision.transforms as transforms
from datasets import load_dataset

from unet import UNet

parser = argparse.ArgumentParser(description='PyTorch Tick Training')
parser.add_argument('-t', '--train-dir', help='training set path', default='/export/data/../LDCT/train')
parser.add_argument('-v', '--valid-dir', help='test set path', default='/export/data/../LDCT/test')
parser.add_argument('--num_workers', type=int, default=40)
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--step_numbers', help='Maximum steps per episode',
                    default=1, type=int)
parser.add_argument('-n', '--noise-type', help='noise type',
        choices=['gaussian', 'poisson', 'hybrid', 'mc'], default='hybrid', type=str)
parser.add_argument('-p', '--noise-param', help='noise parameter (e.g. std for gaussian)', default=10, type=float)
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--patch_n', type=int, default=10)
parser.add_argument('-s', '--seed', help='fix random seed', type=int)
parser.add_argument('-vs', '--valid-size', help='size of valid dataset', type=int)
parser.add_argument('--norm_range_min', type=float, default=-1024.0)
parser.add_argument('--norm_range_max', type=float, default=3072.0)
parser.add_argument('--trunc_min', type=float, default=-160.0)
parser.add_argument('--trunc_max', type=float, default=240.0)

args = parser.parse_args()

param_dict = vars(args)
pretty = lambda x: x.replace('_', ' ').capitalize()
print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()),flush=True)

trunc_min=args.trunc_min
trunc_max=args.trunc_max
norm_range_max=args.norm_range_max
norm_range_min=args.norm_range_min


def generate_n2v_mask(shape, perc_pixels):
    """
    Generate stratified blind-spot coords and a binary mask.
    
    Args:
        shape (tuple): (H, W)
        perc_pixels (float): fraction of pixels to mask (e.g., 0.2 = 20%)
        
    Returns:
        mask (torch.BoolTensor): same spatial shape, True at blind spots
        coords (list of tuples): list of positions selected to mask
    """
    H, W = shape
    total = H * W
    num_masked = max(1, int(total * perc_pixels))
    
    # Sample unique blind-spot indices
    indices = np.random.choice(total, size=num_masked, replace=False)
    ys = indices // W
    xs = indices % W
    
    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[ys, xs] = True
    
    coords = list(zip(ys.tolist(), xs.tolist()))
    return mask, coords

def replace_with_neighbors(x, coords):
    """
    Replace pixel values at coords with a random neighbor value.
    
    Args:
        x (Tensor): [H, W]
        coords (list of (y,x) tuples)
    
    Returns:
        x_out (Tensor): input with blind spots replaced
    """
    H, W = x.shape
    x_out = x.clone()
    
    for y, x0 in coords:
        # sample neighbor displacement (dy, dx)
        dy = np.random.choice([-1, 0, 1])
        dx = np.random.choice([-1, 0, 1])
        
        ny = np.clip(y + dy, 0, H - 1)
        nx = np.clip(x0 + dx, 0, W - 1)
        
        x_out[y, x0] = x_out[ny, nx]
    
    return x_out

def denormalize_( image):
        image = image * (norm_range_max - norm_range_min) + norm_range_min
        return image

def trunc( mat):
        mat[mat <= trunc_min] = trunc_min
        mat[mat >= trunc_max] = trunc_max
        return mat

class Training(object):
    def __init__(self,train_loader,test_loader,image_size,args,gamma=0.9):
        self.image_size=(image_size,image_size)
        self.step_numbers=args.step_numbers
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.batch_size=args.batch_size
        self.gamma=gamma
        self.max_episodes=args.epochs
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.online_model=UNet().cuda()
        self.target_model=UNet().cuda()
        self.target_model.train(False)
        for p in self.target_model.parameters():
            p.requires_grad = False
        self.optimizer_td=torch.optim.Adam(self.online_model.parameters(),lr=args.lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer_td,
                patience=args.epochs/4, factor=0.5)
        self.copy_to_target_model()
        self.tau_base=0.1
        
    def tau_(self, step, max_steps):
        return self.tau_base + (1 - self.tau_base) * (np.cos(np.pi * step / max_steps) + 1) / 2


    def train(self):
        eps=1e-3
        best_psnr=0
        global_step=0
        total_steps=self.max_episodes* len(self.train_loader)
        
        for episode in range(self.max_episodes):
            losses=[]
            losses_n=[]
            self.online_model.train()
            for i, (low,full) in enumerate(self.train_loader):
                global_step+=1
                low = low.to(torch.float32).to(device)
                full= full.to(torch.float32).to(device)
                
                if args.patch_size:
                    low = low.view(-1, 1, args.patch_size, args.patch_size)
                    full = full.view(-1, 1, args.patch_size, args.patch_size)
                obs_low=low.clone().detach()
                
                for step in range(self.step_numbers):
                    masked_imgs = []
                    masks = []

                    for b in range(obs_low.shape[0]):
                        mask, coords = generate_n2v_mask((128, 128), 0.2)
                        masked = replace_with_neighbors(obs_low[b, 0], coords)
                        masked_imgs.append(masked.unsqueeze(0))
                        masks.append(mask.unsqueeze(0))
                    masked_imgs = torch.stack(masked_imgs)  # input to network
                    masks = torch.stack(masks)

                    denoise_t1=self.online_model(masked_imgs)

                    loss_td= ((denoise_t1 - obs_low)**2)[masks].mean() #F.mse_loss(denoise_t1,full)
                    self.optimizer_td.zero_grad()
                    loss_td.backward()
                    self.optimizer_td.step()
                    losses.append(loss_td)
                    obs_low=F.relu(denoise_t1).clone().detach()
            self.scheduler.step(loss_td)


            if episode%1==0:
                print('TD loss: epochs:{0} \t' 'batch_loss:{1} \t'.format(episode,torch.mean(torch.stack(losses))))
            if episode%2==0:
                ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
                pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
                self.online_model.eval()
                for i,(low,full) in enumerate(self.test_loader):
                    shape_ = low.shape[-1]
                    
                    obs_low_t=low.clone().to(torch.float32).detach().to(device)
                    self.online_model.eval()
                    preds = []
                    with torch.no_grad():
                       denoise_t1_t=self.online_model(obs_low_t).to(device)

                    obs_low_t=denoise_t1_t
#
                    full=full.to(torch.float64)
                    low=low.to(torch.float64)
                    low = trunc(denormalize_(low.view(shape_, shape_).cpu().detach()))
                    full = trunc(denormalize_(full.view(shape_, shape_).cpu().detach()))
                    pred = trunc(denormalize_(obs_low_t.to(torch.float64).view(shape_, shape_).cpu().detach()))

                    data_range = trunc_max - trunc_min
                    original_result, pred_result = compute_measure(low, full, pred, data_range)
                    ori_psnr_avg += original_result[0]
                    ori_ssim_avg += original_result[1]
                    ori_rmse_avg += original_result[2]
                    pred_psnr_avg += pred_result[0]
                    pred_ssim_avg += pred_result[1]
                    pred_rmse_avg += pred_result[2]
                print('\n')
                if episode==0:
                    print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(test_loader),
                                                                                            ori_ssim_avg/len(test_loader),
                                                                                            ori_rmse_avg/len(test_loader)))
                print('\n')
                print('epoch:{} \n Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(episode,pred_psnr_avg/len(test_loader),
                                                                                                            pred_ssim_avg/len(test_loader),
                                                                                                            pred_rmse_avg/len(test_loader)))
                if pred_psnr_avg/len(test_loader)>best_psnr:
                    best_psnr=pred_psnr_avg/len(test_loader)

                    net_state_dict=self.online_model.state_dict()
                    torch.save({
                        'epoch': episode,
                        'net_state_dict': net_state_dict},
                        os.path.join(save_dir, 'unet_{}.pt'.format(episode)))

    def copy_to_target_model(self,tau=0.995):
#        self.target_model.load_state_dict(self.online_model.state_dict())
        
        for target_param, online_param in zip(
            self.target_model.parameters(),
            self.online_model.parameters()):
            target_param.data.mul_(tau)
            target_param.data.add_((1 - tau) * online_param.data)
        



os.environ['CUDA_VISIBLE_DEVICES'] = '7'
torch.cuda.set_device('cuda:0')
device = torch.device("cuda:0")

train_loader = load_dataset(args.train_dir, args.patch_size, args, shuffled=True,clean_targets=False)
test_loader = load_dataset(args.valid_dir, args.valid_size, args, shuffled=False,clean_targets=True,single=True)
save_dir="checkpoints"
Training(train_loader,test_loader,256,args).train()







