import torch
import xarray as xr
import numpy as np

weightsA = torch.load('NetworkA_weights_SICpSSTpVelpHIpTSpSSS_SPEAR.pt',map_location=torch.device('cpu'))
weightsB = torch.load('NetworkB_weights_SICpSSTpVelpHIpTSpSSS_SPEAR.pt',map_location=torch.device('cpu'))

weightsA_dict = {'C1':(['w'],np.transpose(weightsA['conv_net.C1.weight'].numpy(),(1,0,3,2)).ravel()),\
                'C2':(['x'],np.transpose(weightsA['conv_net.C2.weight'].numpy(),(1,0,3,2)).ravel()),\
                'C3':(['y'],np.transpose(weightsA['conv_net.C3.weight'].numpy(),(1,0,3,2)).ravel()),\
                'C4':(['z'],np.transpose(weightsA['conv_net.C4.weight'].numpy(),(1,0,3,2)).ravel())}

weightsB_dict = {'C1':(['w'],np.squeeze(weightsB['conv_net.C1.weight'].numpy()).T.ravel()),\
                'C2':(['x'],np.squeeze(weightsB['conv_net.C2.weight'].numpy()).T.ravel()),\
                'C3':(['y'],np.squeeze(weightsB['conv_net.C3.weight'].numpy()).T.ravel()),\
                'C4':(['z'],np.squeeze(weightsB['conv_net.C4.weight'].numpy()).T.ravel())}


dsA = xr.Dataset(data_vars=weightsA_dict)
dsA.to_netcdf('NetworkA_weights_SICpSSTpVelpHIpTSpSSS_SPEAR.nc')

dsB = xr.Dataset(data_vars=weightsB_dict)
dsB.to_netcdf('NetworkB_weights_SICpSSTpVelpHIpTSpSSS_SPEAR.nc')

