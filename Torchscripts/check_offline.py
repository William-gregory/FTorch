import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from collections import OrderedDict

def pad(data,label,size):
    """
    Pad a 3D array by 'size' along its spatial dimensions.
    If data are velocity files (UI or VI), first compute
    2-pt moving average so that fields are defined on the 
    same tracer grid as scalar fields
    """
    if label == 'UI':
        n,dX,dY = data.shape
        if dY == 361:
            data = np.nansum([data[:,:,1:],data[:,:,:-1]],0)/2
        sign = -1
    elif label == 'VI':
        n,dX,dY = data.shape
        if dX == 321:
            data = np.nansum([data[:,1:],data[:,:-1]],0)/2
        sign = -1
    else:
        sign = 1
        
    n,dX,dY = data.shape
    data = np.hstack((data,sign*np.flip(data[:,-size:,:],axis=(1,2))))
    return np.hstack((np.zeros((n,size,dY+2*size)),np.dstack((data[:,:,-size:],np.dstack((data,data[:,:,:size]))))))

def strip_prefix_in_state_dict(state_dict, prefix):
    """
    Strips a specific prefix from the keys in a state_dict.

    Args:
        state_dict (dict): The state_dict to process.
        prefix (str): The prefix to remove.

    Returns:
        dict: The updated state_dict with the prefix removed.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            # Remove the prefix
            new_key = key[len(prefix):]  # Strip the prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

class CNN(nn.Module):
    """
    Fully CNN with 4 convolutional layers
    The input 'args' should be a dictionary containing
    details of the network hyperparameters and architecture
    """

    def __init__(self,argsA,argsB):
        super(CNN, self).__init__()
        torch.manual_seed(argsA['seed'])

        self.conv_netA = nn.Sequential(OrderedDict([
            ('C1', nn.Conv2d(argsA['n_channels'], argsA['h_channels'][0], kernel_size=argsA['kernel_size'],\
                             padding=argsA['zero_padding'],stride=argsA['stride'], bias=argsA['bias'])),
            ('Relu1', nn.ReLU()),
            ('C2', nn.Conv2d(argsA['h_channels'][0], argsA['h_channels'][1], kernel_size=argsA['kernel_size'],\
                             padding=argsA['zero_padding'],stride=argsA['stride'],bias=argsA['bias'])),
            ('Relu2', nn.ReLU()),
            ('C3', nn.Conv2d(argsA['h_channels'][1], argsA['h_channels'][2], kernel_size=argsA['kernel_size'],\
                             padding=argsA['zero_padding'],stride=argsA['stride'],bias=argsA['bias'])),
            ('Relu3', nn.ReLU()),
            ('C4', nn.Conv2d(argsA['h_channels'][2], argsA['n_classes'], kernel_size=argsA['kernel_size'],\
                             padding=argsA['zero_padding'],stride=argsA['stride'],bias=argsA['bias'])),
        ]))

        self.conv_netB = nn.Sequential(OrderedDict([
            ('C1', nn.Conv2d(argsB['n_channels'], argsB['h_channels'][0], kernel_size=argsB['kernel_size'],\
                             padding=argsB['zero_padding'],stride=argsB['stride'], bias=argsB['bias'])),
            ('Relu1', nn.ReLU()),
            ('C2', nn.Conv2d(argsB['h_channels'][0], argsB['h_channels'][1], kernel_size=argsB['kernel_size'],\
                             padding=argsB['zero_padding'],stride=argsB['stride'],bias=argsB['bias'])),
            ('Relu2', nn.ReLU()),
            ('C3', nn.Conv2d(argsB['h_channels'][1], argsB['h_channels'][2], kernel_size=argsB['kernel_size'],\
                             padding=argsB['zero_padding'],stride=argsB['stride'],bias=argsB['bias'])),
            ('Relu3', nn.ReLU()),
            ('C4', nn.Conv2d(argsB['h_channels'][2], argsB['n_classes'], kernel_size=argsB['kernel_size'],\
                             padding=argsB['zero_padding'],stride=argsB['stride'],bias=argsB['bias'])),
        ]))

    def forward(self, x):
        dSIC = (self.conv_netA(x[:,:8]) + 0.0009238007032701131)/0.03622488757495426 #normalization for dSIC
        halo = (x.shape[2] - dSIC.shape[2])//2
        dSIC[x[:,-1:,halo:-halo,halo:-halo]==0] = 0
        dCN = torch.permute(torch.squeeze(self.conv_netB(torch.hstack((dSIC,x[:,8:,halo:-halo,halo:-halo])))),(1,2,0))
        ni,nj,ct = dCN.shape
        dCN = torch.vstack((torch.vstack((torch.zeros((halo,nj,ct)),dCN)),torch.zeros((halo,nj,ct))))
        return torch.hstack((torch.hstack((torch.zeros((ni+2*halo,halo,ct)),dCN)),torch.zeros((ni+2*halo,halo,ct))))
        

inA = ['siconc','SST','UI','VI','HI','TS','SSS','mask']
widths = [32,64,128]
argsA = {
'kernel_size':3,
'zero_padding':0,
'h_channels':widths,
'n_channels':int(len(inA)),
'n_classes':1,
'stride':1,
'bias':False,
'seed':711,
}
argsB = {
'kernel_size':1,
'zero_padding':0,
'h_channels':widths,
'n_channels':7,    
'n_classes':5,
'stride':1,
'bias':False,
'seed':711,
}

model = CNN(argsA,argsB)
pathA = '/gpfs/f5/gfdl_o/scratch/William.Gregory/FTorch/weights/NetworkA_weights_SICpSSTpVelpHIpTSpSSS_SPEAR.pt'
pathB = '/gpfs/f5/gfdl_o/scratch/William.Gregory/FTorch/weights/NetworkB_weights_SICpSSTpVelpHIpTSpSSS_SPEAR.pt'

paramA = torch.load(pathA,map_location=torch.device('cpu'))
paramB = torch.load(pathB,map_location=torch.device('cpu'))

f = xr.open_dataset('/gpfs/f5/gfdl_o/scratch/William.Gregory/SPEAR_reforecast/i20180301_ODA_OTA_IceDA_coupledCNN_FTorch/ncrc5.intel-classic-repro-openmp/history/20180301.ice_daily.ens_01.nc')
model.conv_netA.load_state_dict(strip_prefix_in_state_dict(paramA, "conv_net."))
model.conv_netB.load_state_dict(strip_prefix_in_state_dict(paramB, "conv_net."))
model.eval()

NetA_stats = np.load('/gpfs/f5/gfdl_o/scratch/William.Gregory/CNNForpy/NetworkA_statistics_SPEAR_1982-2017.npz')
NetB_stats = np.load('/gpfs/f5/gfdl_o/scratch/William.Gregory/CNNForpy/NetworkB_statistics_SPEAR_1982-2017.npz')
X = np.zeros((14,328,368))
for ix,label in enumerate(inA):
    if label != 'mask':
        X[ix] = pad(f[label].to_numpy(),label,4)
land_mask = np.copy(X[1])
land_mask[~np.isnan(land_mask)] = 1
land_mask[np.isnan(land_mask)] = 0
land_mask[:,:4] = 0
X[7] = land_mask
X[13] = land_mask
for N in range(7):
    X[N] = (X[N] - NetA_stats['mu'][N])/NetA_stats['sigma'][N]
    X[N][land_mask==0] = 0

for CN in range(5):
    X[8+CN] = pad((f['CN'].isel(ct=CN).to_numpy() - NetB_stats['mu'][CN+1])/NetB_stats['sigma'][CN+1],'CN',4)
    X[8+CN][land_mask==0] = 0

X[np.isnan(X)] = 0
X_torch = torch.from_numpy(X[None].astype(np.float32))
pred = model(X_torch).detach().numpy()
np.save('offline_Torch_output.npy',pred)
np.save('offline_Torch_input.npy',X)
    

