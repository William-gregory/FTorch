import torch
import torch.nn as nn
from collections import OrderedDict

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

    def __init__(self,argsA, argsB, NetA_mu, NetA_sig, NetB_mu, NetB_sig):
        super(CNN, self).__init__()
        #torch.manual_seed(argsA['seed'])

        self.register_buffer('NetA_mu',NetA_mu)
        self.register_buffer('NetA_sig',NetA_sig)
        self.register_buffer('NetB_mu',NetB_mu)
        self.register_buffer('NetB_sig',NetB_sig)

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
        x[torch.isnan(x)] = 0.0
        x[torch.isinf(x)] = 0.0
        for N in range(len(self.NetA_mu)):
            x[:,N] = (x[:,N] - self.NetA_mu[N])/self.NetA_sig[N] #normalize inputs
            x[:,N][x[:,-1]==0] = 0.0 #set land values to zero
        dSIC = self.conv_netA(x[:,:8]) #generate prediction from first CNN
        halo = (x.shape[2] - dSIC.shape[2])//2 #halo size used for first CNN (default 4)
        dSIC[x[:,-1:,halo:-halo,halo:-halo]==0] = 0 #set land values in prediction to zero
        dSIC[torch.isnan(dSIC)] = 0.0
        dSIC[torch.isinf(dSIC)] = 0.0
        
        x = torch.hstack((dSIC,x[:,8:,halo:-halo,halo:-halo])) #combine first CNN prediction with remain state variables for second CNN
        for N in range(len(self.NetB_mu)):
            x[:,N] = (x[:,N] - self.NetB_mu[N])/self.NetB_sig[N] #normalize inputs
            x[:,N][x[:,-1]==0] = 0.0 #set land value to zero
        dCN = torch.permute(torch.squeeze(self.conv_netB(x)),(1,2,0)) #permute is the same as transpose
        ni,nj,ct = dCN.shape
        for cat in range(ct):
            dCN[:,:,cat][x[0,-1]==0] = 0 #set land values in prediction to zero
        dCN = torch.vstack((torch.vstack((torch.zeros((halo,nj,ct)),dCN)),torch.zeros((halo,nj,ct)))) #re-pad prediction with halos of value zero
        dCN = torch.hstack((torch.hstack((torch.zeros((ni+2*halo,halo,ct)),dCN)),torch.zeros((ni+2*halo,halo,ct)))) #padding y axis
        dCN[torch.isnan(dCN)] = 0.0
        dCN[torch.isinf(dCN)] = 0.0
        return dCN
        

inA = ['siconc','SST','UI','VI','HI','TS','SSS','mask']
widths = [18,32,64]
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

NetA_mu = torch.load('/gpfs/f5/gfdl_o/scratch/William.Gregory/FTorch/statistics/NetworkA_mean_statistics_SICpSSTpVelpHIpTSpSSS_SPEAR.pt').to(torch.float32)
NetA_sig = torch.load('/gpfs/f5/gfdl_o/scratch/William.Gregory/FTorch/statistics/NetworkA_sigma_statistics_SICpSSTpVelpHIpTSpSSS_SPEAR.pt').to(torch.float32)
NetB_mu = torch.load('/gpfs/f5/gfdl_o/scratch/William.Gregory/FTorch/statistics/NetworkB_mean_statistics_dSICpCN_SPEAR.pt').to(torch.float32)
NetB_sig = torch.load('/gpfs/f5/gfdl_o/scratch/William.Gregory/FTorch/statistics/NetworkB_sigma_statistics_dSICpCN_SPEAR.pt').to(torch.float32)

model = CNN(argsA,argsB,NetA_mu,NetA_sig,NetB_mu,NetB_sig)
pathA = '/gpfs/f5/gfdl_o/scratch/William.Gregory/FTorch/weights/NetworkA_weights_SICpSSTpVelpHIpTSpSSS_18x32x64_SPEAR.pt'
pathB = '/gpfs/f5/gfdl_o/scratch/William.Gregory/FTorch/weights/NetworkB_weights_SICpSSTpVelpHIpTSpSSS_18x32x64_SPEAR.pt'

paramA = torch.load(pathA,map_location=torch.device('cpu'))
paramB = torch.load(pathB,map_location=torch.device('cpu'))

model.conv_netA.load_state_dict(strip_prefix_in_state_dict(paramA, "conv_net."))
model.conv_netB.load_state_dict(strip_prefix_in_state_dict(paramB, "conv_net."))
model.eval()
with torch.no_grad():
    script = torch.jit.script(model)
    script.save('NetworkAB_script_SICpSSTpVelpHIpTSpSSS_18x32x64_SPEAR.pt')

