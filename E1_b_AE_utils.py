import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import h5py
from prettytable import PrettyTable
from torch.nn.functional import mse_loss
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d, BatchNorm2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torch.nn import Sequential
from typing import Callable, Optional
from torch.nn import functional as F
from torch import Tensor
import torch

# ------------------------------------------------------------------------------------------
# -----------------------     make reproducible results   ----------------------------------
# ------------------------------------------------------------------------------------------

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_random_seeds(random_seed=0):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# ------------------------------------------------------------------------------------------
# ----------------------------------     drop out    ---------------------------------------
# ------------------------------------------------------------------------------------------
	
def freeze_some_layers(model, fraction, epoch):
    parameters = list(model.parameters()) 
    to_freeze = int(len(parameters) * fraction)
    np.random.seed(epoch)
    for i in np.random.choice(len(parameters), to_freeze, replace=False):
        parameters[i].requires_grad = False

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

# ------------------------------------------------------------------------------------------
# ---------------------------     count network parameter    -------------------------------
# ------------------------------------------------------------------------------------------

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# ------------------------------------------------------------------------------------------
# --------------------------------     loss function    ------------------------------------
# ------------------------------------------------------------------------------------------

def lossFunc(pred, y):
    L2 = mse_loss(pred, y, size_average=None, reduce=None, reduction='mean')
    return L2**0.5 

# ------------------------------------------------------------------------------------------
# ---------------------------------     data loader    -------------------------------------
# ------------------------------------------------------------------------------------------

class LoadTCFDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir 
        with h5py.File(self.img_dir, "r") as f:
            self.totalSamples = f['vel_fl'].shape[0] # samples,channel,z,y,x

    def __len__(self): 
        return self.totalSamples

    def __getitem__(self, idx): 
        with h5py.File(self.img_dir, "r") as f: 
            vel = np.array(f['vel_fl'][idx,:,:,:,:], dtype='float32') # samples,channel,z,y,x
        return vel 


# ------------------------------------------------------------------------------------------
# ------------------ functions and classes related to the AE model    ----------------------
# ------------------------------------------------------------------------------------------
	
def conv3x3(in_planes: int, out_planes: int,
    stride = 1, 
    groups = 1, 
    dilation = 1
) -> Conv2d:
    return Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(Module):
    expansion: int = 1

    def __init__(self,inplanes: int, planes: int,
        stride: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., Module]] = None,
    ) -> None:

        super().__init__()
        
        if norm_layer is None:
            norm_layer = BatchNorm2d

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

    def forward(self, x: Tensor) -> Tensor:        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
	

class ResBlock(Module):
    expansion: int = 1

    def __init__(self,inplanes: int, planes: int,
        downsample: Optional[Module] = None,
        stride: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., Module]] = None,
    ) -> None:

        super().__init__()
        
        if norm_layer is None:
            norm_layer = BatchNorm2d
        
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None: 
            x = self.downsample(x)

        out = out + x 
        out = self.relu(out)

        return out
	
# ------------------------------------------------------------------------------------------
# encoder
	
class BasicEncoder(Module):
	def __init__(self, block, layers=(1,2,2,2), channels=(64,128,256,512), resnet=False, norm_layer=None):
		super().__init__()
		self.channels = channels
		self.layers = layers
		if norm_layer is None:
			norm_layer = BatchNorm2d
        
		self.inplanes = channels[0]

		self.conv1 = Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = norm_layer(self.inplanes)
		self.relu = ReLU(inplace=True)
		self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
		
		if resnet:
			self.downsample = ModuleList([Sequential(Conv2d(channels[i], channels[i+1], kernel_size = 1, stride = 2), norm_layer(channels[i+1]))for i in range(len(channels) - 1)])
			self.enc_blocks1 = ModuleList([block(channels[i], channels[i + 1], self.downsample[i], stride = 2) for i in range(len(channels) - 1)])
		else:
			self.enc_blocks1 = ModuleList([block(channels[i], channels[i + 1], stride = 2) for i in range(len(channels) - 1)]) 
			
		if (1 not in self.layers):
			self.enc_blocks2 = ModuleList([block(channels[i], channels[i]) for i in range(1,len(channels))]) 
		else:
			self.enc_blocks2 = ModuleList()
			for i in range(1,len(channels)):
				if self.layers[i] > 1:
					self.enc_blocks2.append(block(channels[i], channels[i]))
		self.enc_start_block = block(channels[0], channels[0])
  
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		m = 0
		for i in range(len(self.channels)):
			if i == 0:
				x = self.enc_start_block(x)
			else:
				x = self.enc_blocks1[i-1](x)
            
			for k in range(self.layers[i]-1): 
				if i == 0:
					x = self.enc_start_block(x)
				else:
					x = self.enc_blocks2[m](x)
					if k == self.layers[i]-2:
						m = m+1
				k = k+1
		
		return x
	
# ------------------------------------------------------------------------------------------
# decoder
	    
class InterpolateDecoder(Module):	
	def __init__(self, block, layers=(1,1,1), channels=(512, 256, 128, 64), scaleFactor=2.0, resnet=False):
		super().__init__()
		self.channels = channels
		self.layers = layers
		self.scaleFactor = scaleFactor
		if resnet:
			self.downsample = ModuleList([Sequential(Conv2d(channels[i+1]*2, channels[i+1], kernel_size = 1, stride = 1))for i in range(len(channels) - 1)])
			self.dec_blocks1 = ModuleList([block(channels[i+1]*2, channels[i+1], self.downsample[i]) for i in range(len(channels) - 1)])
		else:
			self.dec_blocks1 = ModuleList([block(channels[i+1]*2, channels[i+1]) for i in range(len(channels) - 1)])
		if (1 not in self.layers):
			self.dec_blocks2 = ModuleList([block(channels[i+1], channels[i+1]) for i in range(len(channels)-1)])
		else:
			self.dec_blocks2 = ModuleList()
			for i in range(len(channels)-1):
				if self.layers[i] > 1:
					self.dec_blocks2.append(block(channels[i+1], channels[i+1]))

	def forward(self, x):
		m=0
		for i in range(len(self.channels)-1):
			x = F.interpolate(x, scale_factor=self.scaleFactor, mode='nearest-exact')
			x = self.dec_blocks1[i](x)
			for k in range(self.layers[i]-1):
				x = self.dec_blocks2[m](x)
				if k == self.layers[i]-2:
					m = m+1
		
		return x
		

# ------------------------------------------------------------------------------------------
# -------------------------------- final AE model    --------------------------------------
# ------------------------------------------------------------------------------------------

class Model(Module):
	def __init__(self, enclayers, planes, declayers, decchannels, outputSize, scaleFactor=2.0, resnet=False):
		super().__init__()
		if resnet:
			self.encoder = BasicEncoder(ResBlock, enclayers, planes, resnet)
			self.decoder = InterpolateDecoder(ResBlock, declayers, decchannels, scaleFactor, resnet)	
		else:
			self.encoder = BasicEncoder(BasicBlock, enclayers, planes)  
			self.decoder = InterpolateDecoder(BasicBlock, declayers, decchannels, scaleFactor)		
		           
		self.head1 = ConvTranspose2d(decchannels[-1], decchannels[-1], 2, 2)
		self.head2 = ConvTranspose2d(decchannels[-1], 1, 1)
		self.relu = ReLU(inplace=True)
		self.outSize = outputSize               
    
	def forward(self,x):        
		encFeatures = self.encoder(x)        
		output = self.decoder(encFeatures)
		while list(output.shape[-2:]) != self.outSize:
			output = self.head1(output)
			output = self.relu(output)
		output = self.head2(output) 

		return output