import torch
import numpy as np

class lin_regr_model(torch.nn.Module):
    def __init__(self, nb_of_input_features, nb_of_output_features):
        super().__init__()
        self.nb_of_input_features = nb_of_input_features
        self.nb_of_output_features = nb_of_output_features
        
        self.cut_border_pix_output = None
        self.cut_border_pix_input = None
        
        self.lin1 = torch.nn.Linear(self.nb_of_input_features, self.nb_of_output_features, bias=False)
        
        # initialization 
        self.lin1.weight.data = torch.Tensor([[0.1]])

    def forward(self, x):
        batch_len, nb_of_levels, nb_of_channels, output_h, output_w = x.shape
        
        # deattach levels into batch entities by flattening
        res = x.flatten(start_dim=0, end_dim=1) # shape [N',C,H,W]
        new_batch_len = batch_len*nb_of_levels
        
        # first split the input 4D torch tensor into individual pixels (equivalent to patches of size 1x1)
        res = torch.nn.functional.unfold(res, kernel_size=1, dilation=1, padding=0, stride=1)
        res = torch.permute(res, dims=(0,2,1))
        res = torch.flatten(res, end_dim=1).to(torch.float32)
        
        # perform linear regression
        res = self.lin1(res)
        
        # reshape the model output back to a 4D torch tensor
        res = torch.permute(res.unflatten(dim=0, sizes=[new_batch_len,-1]),dims=(0,2,1))
        res = torch.nn.functional.fold(res, output_size=(output_h,output_w), kernel_size=1, dilation=1, padding=0, stride=1)

        # unflatten the levels
        res = res.unflatten(dim=0, sizes=(batch_len, nb_of_levels))
        return res
    
class FCNN(torch.nn.Module):
    def __init__(self, nb_of_input_features, nb_of_output_features, input_patch_size, output_patch_size, int_layer_width=50):
        super().__init__()
        self.input_patch_size = input_patch_size
        self.output_patch_size = output_patch_size
        
        self.lin1 = torch.nn.Linear(nb_of_input_features*input_patch_size**2, int_layer_width, bias=True)
        self.lin2 = torch.nn.Linear(int_layer_width, int_layer_width, bias=True)
        self.lin3 = torch.nn.Linear(int_layer_width, nb_of_output_features*output_patch_size**2, bias=True)
        
        self.cut_border_pix_output = self.input_patch_size//2 - self.output_patch_size//2
        if (self.cut_border_pix_output < 1) :
            self.cut_border_pix_output = None
        self.cut_border_pix_input = None

    def forward(self, x):
        batch_len, nb_of_levels, nb_of_channels = x.shape[0:3]
        output_h = x.shape[3]-2*(self.cut_border_pix_output or 0)
        output_w = x.shape[4]-2*(self.cut_border_pix_output or 0)
        
        # deattach levels into batch entities by flattening
        res = x.flatten(start_dim=0, end_dim=1) # shape [N',C,H,W]
        new_batch_len = batch_len*nb_of_levels
        
        # create patches of size 'input_patch_size' and join them into batches (zero padding - will remove border pixels)
        res = torch.nn.functional.unfold(res, kernel_size=self.input_patch_size, dilation=1, padding=0, stride=1)
        res = torch.permute(res, dims=(0,2,1))
        res = torch.flatten(res, end_dim=1)
        
        # pass though the FCNN
        res = self.lin1(res)
        res = torch.nn.functional.relu(res)
        res = self.lin2(res)
        res = torch.nn.functional.relu(res)
        res = self.lin3(res)
        
        # reshape the output patches back into a 4D torch tensor
        res = res.unflatten(dim=0, sizes=(new_batch_len,-1))
        res = torch.permute(res,dims=(0,2,1))
        res = torch.nn.functional.fold(res, output_size=(output_h,output_w), \
                                       kernel_size=self.output_patch_size, dilation=1, padding=0, stride=1)
        # compute the divider needed to get correct values in case of overlapping patches (will give mean over all overlapping patches)
        mask_ones = torch.ones((1,1,output_h,output_w)).to(x.device)
        divisor = torch.nn.functional.fold(torch.nn.functional.unfold(mask_ones, kernel_size=self.output_patch_size), \
                                           kernel_size=self.output_patch_size, output_size=(output_h,output_w))   
        res = res/divisor.view(1,1,output_h,output_w)
        
        # unflatten the levels
        res = res.unflatten(dim=0, sizes=(batch_len, nb_of_levels))
        
        return res
    
class CNN(torch.nn.Module):
    def __init__(self, nb_of_input_features, nb_of_output_features, padding='same', padding_mode='replicate', \
                 kernel_size=3, int_layer_width=64):
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.padding_mode = 'replicate'
        
        self.cut_border_pix_input = None
        if self.padding == 'same' :
            self.cut_border_pix_output = self.cut_border_pix_input
        if self.padding == 'valid' :
            self.cut_border_pix_output = (self.cut_border_pix_input or 0) + self.kernel_size//2
        
        self.conv1 = torch.nn.Conv2d(in_channels=nb_of_input_features, out_channels=int_layer_width, kernel_size=self.kernel_size, \
                                     padding=self.padding,  padding_mode=self.padding_mode) 
        self.conv2 = torch.nn.Conv2d(int_layer_width, int_layer_width, kernel_size=self.kernel_size, padding='same', padding_mode=self.padding_mode) 
        self.conv3 = torch.nn.Conv2d(int_layer_width, nb_of_output_features, kernel_size=self.kernel_size, padding='same', \
                                     padding_mode=self.padding_mode)

    def forward(self, x):
        batch_len = x.shape[0]
        nb_of_levels = x.shape[1]
        
        # deattach levels into batch entities by flattening
        res = x.flatten(start_dim=0, end_dim=1) # shape [N',C,H,W]
        
        res = self.conv1(res)
        res = torch.nn.functional.relu(res)
        res = self.conv2(res)
        res = torch.nn.functional.relu(res)
        res = self.conv3(res)
        
        # unflatten the levels
        res = res.unflatten(dim=0, sizes=(batch_len, nb_of_levels))
        return res        