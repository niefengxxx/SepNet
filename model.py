import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter 
import numpy as np
import math

SRM_npy3 = np.load('kernels_3*3.npy')
SRM_npy5 = np.load('kernels_5*5.npy')
#3*3 padding 1
#5*5 pading 2

class Preprocess(nn.Module):
    def __init__(self,stride = 1):
        super(Preprocess,self).__init__()
        self.in_channel = 1
        self.out_channel_3 = 25
        self.out_channel_5 = 5
        self.kernels_3 = (3,3)
        self.kernels_5 = (5,5)
        self.stride = (stride,stride)
        self.padding_3 = (1,1)
        self.padding_5 = (2,2)
        self.weight_3 = Parameter(torch.Tensor(25,1,3,3),requires_grad=True)
        self.weight_5 = Parameter(torch.Tensor(5,1,5,5),requires_grad=True)
        self.bias_3 = Parameter(torch.Tensor(25),requires_grad=True)
        self.bias_5 = Parameter(torch.Tensor(5),requires_grad=True)

        self.set_parameters()

    def set_parameters(self):
        self.weight_3.data.numpy()[:] = SRM_npy3
        self.bias_3.data.zero_()

        self.weight_5.data.numpy()[:] = SRM_npy5
        self.bias_5.data.zero_()
    
    def forward(self,input):
        out1 = F.conv2d(input,self.weight_3,self.bias_3,self.stride,self.padding_3)
        out2 = F.conv2d(input,self.weight_5,self.bias_5,self.stride,self.padding_5)
        #print(out1.shape)
        #print(out2.shape)
        out = torch.cat((out1,out2),1)
        return out      

class SPP(nn.Module):
    def __init__(self,num_levels,pool_type = 'avg_pool'):
        super(SPP,self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type
    
    def forward(self,input):
        num,c,h,w = input.size()
        level = 1
        for i  in range(self.num_levels):
            
            kernel_size = (math.ceil(h/level),math.ceil(w/level))
            stride = (math.ceil(h/level),math.ceil(w/level))
            padding = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(input,kernel_size = kernel_size,stride = stride,padding = padding).view(num,-1)
            else:
                tensor = F.avg_pool2d(input,kernel_size = kernel_size,stride = stride,padding = padding).view(num,-1)

            if i == 0:
                out = tensor.view(num,-1)
            else:
                out = torch.cat((out,tensor.view(num,-1)),1)
            level *= 2
        return out

class Sepconv(nn.Module):
    def __init__(self,Abs = True):
        super(Sepconv,self).__init__()
        self.Abs = Abs

        self.in_channels = 30
        self.mid_channels = 60
        self.out_channels = 30
        self.conv1 = nn.Conv2d(self.in_channels,self.mid_channels,kernel_size=3,stride=1,padding=1,groups=30)
        self.conv2 = nn.Conv2d(self.mid_channels,self.out_channels,kernel_size=1,stride=1)
        self.norm = nn.BatchNorm2d(self.out_channels)

        self.set_parameters()
    
    def forward(self,input):
        if self.Abs == True:
            out1 = torch.abs(self.conv1(input))
        else:
            out1 = self.conv1(input)
        out = self.norm(self.conv2(out1))
        return out
    
    def set_parameters(self):
        nn.init.xavier_uniform(self.conv1.weight)
        nn.init.xavier_uniform(self.conv2.weight)
        self.norm.reset_parameters()

class Conv_Block(nn.Module):
    def __init__(self,in_channels,out_channels,stride = 1):
        super(Conv_Block,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = 1
        self.padding = 1
        self.conv = nn.Conv2d(self.in_channels,self.out_channels,kernel_size=3,stride=self.stride, padding=self.padding)
        self.norm = nn.BatchNorm2d(self.out_channels)

        self.set_parameters()
    
    def forward(self,input):
        return F.relu(self.norm(self.conv(input)))
    
    def set_parameters(self):
        nn.init.xavier_uniform(self.conv.weight)
        self.norm.reset_parameters()


class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()

        self.pre = Preprocess()

        self.sepconv_block1 = Sepconv(Abs=True)
        self.sepconv_block2 = Sepconv(Abs=False)

        self.block1 = Conv_Block(30,32,1)
        self.block2 = Conv_Block(32,32,1)
        self.block3 = Conv_Block(32,64,1)
        self.block4 = Conv_Block(64,128,1)

        self.spp = SPP(3,pool_type='avg_pool')

        self.fc1 = nn.Linear(1*2688,1024)
        self.fc2 = nn.Linear(1*1024,2)
        self.set_parameters()
    
    def forward(self,x):
        x =x.float()
        x = self.pre(x)
        res = self.sepconv_block1(x)
        res = self.sepconv_block2(res)
        x = torch.add(x,res)
        x = self.block1(x)
        x = F.avg_pool2d(x,kernel_size = 5,stride =2,padding =2)
        x = self.block2(x)
        x = F.avg_pool2d(x,kernel_size = 5,stride =2,padding =2)
        x = self.block3(x)
        x = F.avg_pool2d(x,kernel_size = 5,stride =2,padding =2)
        x = self.block4(x)
        x = self.spp(x)
        print(x.shape)
        x = self.fc1(x)
        x = F.softmax(self.fc2(x))
        return x
    
    def set_parameters(self):
        for mod in self.modules():
            if isinstance(mod,Preprocess) or isinstance(mod,Conv_Block) or isinstance(mod,Sepconv):
                mod.set_parameters()
            elif isinstance(mod,nn.Linear):
                nn.init.normal(mod.weight, 0. ,0.01)
                mod.bias.data.zero_()
