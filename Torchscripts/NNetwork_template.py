import torch
import torch.nn as nn
from collections import OrderedDict

class CNN(nn.Module):
    """
    Fully CNN with 4 convolutional layers
    The input 'args' should be a dictionary containing
    details of the network hyperparameters and architecture
    """

    def __init__(self,args):
        super(CNN, self).__init__()
        torch.manual_seed(args['seed'])

        self.conv_net = nn.Sequential(OrderedDict([
            ('C1', nn.Conv2d(args['n_channels'], args['h_channels'][0], kernel_size=args['kernel_size'],\
                             padding=args['zero_padding'],stride=args['stride'], bias=args['bias'])),
            ('Relu1', nn.ReLU()),
            ('C2', nn.Conv2d(args['h_channels'][0], args['h_channels'][1], kernel_size=args['kernel_size'],\
                             padding=args['zero_padding'],stride=args['stride'],bias=args['bias'])),
            ('Relu2', nn.ReLU()),
            ('C3', nn.Conv2d(args['h_channels'][1], args['h_channels'][2], kernel_size=args['kernel_size'],\
                             padding=args['zero_padding'],stride=args['stride'],bias=args['bias'])),
            ('Relu3', nn.ReLU()),
            ('C4', nn.Conv2d(args['h_channels'][2], args['n_classes'], kernel_size=args['kernel_size'],\
                             padding=args['zero_padding'],stride=args['stride'],bias=args['bias'])),
        ]))

    def forward(self, x):
        return self.conv_net(x)




inA = ['siconc','SST','UI','VI','HI','TS','mask']
argsA = {
'kernel_size':3,
'zero_padding':0,
'h_channels':[18,32,64],
'n_channels':int(len(inA)),
'n_classes':1,
'stride':1,
'bias':False,
'seed':711,
}
modelA = CNN(argsA)
pathA = '/gpfs/f5/gfdl_o/scratch/William.Gregory/FTorch/weights/NetworkA_weights_SICpSSTpVelpHIpTS_18x32x64_SPEAR.pt'
modelA.load_state_dict(torch.load(pathA,map_location=torch.device('cpu')))
modelA.eval()
scriptA = torch.jit.script(modelA)
scriptA.save('NetworkA_script_SICpSSTpVelpHIpTS_18x32x64_SPEAR.pt')


argsB = {
'kernel_size':1,
'zero_padding':0,
'h_channels':[18,32,64],
'n_channels':7,    
'n_classes':5,
'stride':1,
'bias':False,
'seed':711,
}
modelB = CNN(argsB)
pathB = '/gpfs/f5/gfdl_o/scratch/William.Gregory/FTorch/weights/NetworkB_weights_SICpSSTpVelpHIpTS_18x32x64_SPEAR.pt'
modelB.load_state_dict(torch.load(pathB,map_location=torch.device('cpu')))
modelB.eval()
scriptB = torch.jit.script(modelB)
scriptB.save('NetworkB_script_SICpSSTpVelpHIpTS_18x32x64_SPEAR.pt')

