import numpy as np
import xarray as xr
import torch
import pytorch_lightning as pl
from scipy import ndimage
import itertools
import os

def crop_2d_maps(xr_data, h, w) :
    return xr_data.isel(x_c=slice(None,w), y_c=slice(None,h))

def erode_bin_mask(xr_mask) :
    erosion_structure_matrix = np.array([(0,0,1,0,0), (0,1,1,1,0), (1,1,1,1,1), (0,1,1,1,0), (0,0,1,0,0)])
    np_array_mask = ndimage.binary_erosion(xr_mask, structure=erosion_structure_matrix)
    return xr_mask.copy(data=np_array_mask)

def toTorchTensor(xrArray):
    transformed_data = torch.tensor(xrArray.values)
    return transformed_data

def central_diffs(dataArray) :
    if len(dataArray.shape) == 5 : #5d data cube
        batch_len, nb_of_levels, nb_of_channels, output_h, output_w = dataArray.shape
        flatten_data = dataArray.flatten(start_dim=0, end_dim=2)[:,None,:,:]
    if len(dataArray.shape) == 4 : # 1 channel (or 1 level)
        batch_len, nb_of_channels, width, height = dataArray.shape
        flatten_data = dataArray.flatten(start_dim=0, end_dim=1)[:,None,:,:]
    if len(dataArray.shape) == 3 : # 1 channel
        batch_len, width, height = dataArray.shape
        flatten_data = dataArray[:,None,:,:]
    weights = torch.zeros(2,1,3,3).to(dataArray.device) # 2 channels : 1 channel for x-difference, other for y-differences
    weights[0,0,:,:] = torch.tensor([[0,0.,0],[-0.5,0.,0.5],[0,0.,0]]) #dx
    weights[1,0,:,:] = torch.tensor([[0,-0.5,0],[0,0.,0],[0,0.5,0]])   #dy
    res = torch.nn.functional.conv2d(flatten_data.float(), weights, \
                               bias=None, stride=1, padding='same', dilation=1, groups=1)
    if len(dataArray.shape) == 5 :
        res_dx = res[:,0,1:-1,1:-1].unflatten(dim=0, sizes=(batch_len, nb_of_levels, nb_of_channels))
        res_dy = res[:,1,1:-1,1:-1].unflatten(dim=0, sizes=(batch_len, nb_of_levels, nb_of_channels))
    if len(dataArray.shape) == 4 :
        res_dx = res[:,0,1:-1,1:-1].unflatten(dim=0, sizes=(batch_len, nb_of_channels))
        res_dy = res[:,1,1:-1,1:-1].unflatten(dim=0, sizes=(batch_len, nb_of_channels))
    if len(dataArray.shape) == 3 :
        res_dx = res[:,0,1:-1,1:-1]
        res_dy = res[:,1,1:-1,1:-1]
    return res_dx, res_dy

def finite_diffs_sqr_2d_map(dataArray) :
    res_dx, res_dy = central_diffs(dataArray)
    res = torch.pow(res_dx,2) + torch.pow(res_dy,2)
    return res

def masked_mean(data_geometry, tensor, mask, reduction='mean') :
    if (data_geometry == '2D') :
        if (len(tensor.shape) == 4) : # 4D tensor with C features
            batch_len, nb_of_channels, output_h, output_w = tensor.shape  
            channel_dim = 1
            valid_mask_counts = torch.count_nonzero(mask)*nb_of_channels
            mask = mask[:,None,:,:]
        if (len(tensor.shape) == 3) : # 1 feature (=1 channel)-> 3D tensor
            batch_len, output_h, output_w = tensor.shape  
            channel_dim = None
            valid_mask_counts = torch.count_nonzero(mask)
            mask = mask
    if (data_geometry == '3D') :
        if (len(tensor.shape) == 5) : # full 5D tensor
            batch_len, nb_of_levels, nb_of_channels, output_h, output_w = tensor.shape  
            channel_dim = 2
            valid_mask_counts = torch.count_nonzero(mask)*nb_of_levels*nb_of_channels
            mask = mask[:,None,None,:,:]
        if (len(tensor.shape) == 4) : # 1 feature (=1 channel)
            batch_len, nb_of_levels, output_h, output_w = tensor.shape  
            channel_dim = None
            valid_mask_counts = torch.count_nonzero(mask)*nb_of_levels
            mask = mask[:,None,:,:]
        if (len(tensor.shape) == 3) : # 1 feature (=1 channel) and 1 level
            batch_len, output_h, output_w = tensor.shape 
            channel_dim = None
            valid_mask_counts = torch.count_nonzero(mask)
            mask = mask

    if (reduction=='none') :
        return (tensor*mask) # shape [N,(L),(C), H, W]
    
    total = torch.sum(tensor*mask)
    if (reduction=='sum') : 
        return total    # 1 number
    if (reduction=='mean') : 
        return (total/valid_mask_counts) # 1 number
    if (reduction=='vertical_mean') :
        sum_over_each_layer = torch.sum(tensor*mask, dim=(-2,-1))
        valid_counts_each_layer = torch.count_nonzero(mask, dim=(-2,-1))
        vertical_profile_of_each_sample = sum_over_each_layer/valid_counts_each_layer # shape [N,L,(C)]
        res = torch.mean(vertical_profile_of_each_sample, dim=0) # average over the batch, final shape [L, (C)]
        return res
    if (reduction=='horizontal_mean') :
        sum_over_depth_at_each_point = torch.sum(tensor*mask, dim=1) # avg over depth, shape [N, (C), H, W]
        valid_counts = torch.count_nonzero(mask, dim=1)*nb_of_levels
        horizontal_error_of_each_sample = sum_over_depth_at_each_point/valid_counts
        res = torch.mean(horizontal_error_of_each_sample, dim=0) # [(C), H, W]
        return res 
    if (reduction=='sample_mean') :
        reduc_dims = tuple(range(1,len(tensor.shape))) # all dimensions>0
        if not (channel_dim is None) :
            reduc_dims.remove(channel_dim) 
        sum_over_each_sample = torch.sum(tensor*mask, dim=reduc_dims)
        valid_counts = torch.count_nonzero(mask, dim=reduc_dims)*(nb_of_levels if data_geometry == '3D' else 1.)
        res = sum_over_each_sample/valid_counts # shape[N, (C)]
        return res
    if (reduction=='channel_mean') : 
        reduc_dims = list(range(len(tensor.shape)))
        reduc_dims.remove(channel_dim)
        sum_over_each_channel = torch.sum(tensor*mask, dim=reduc_dims)  
        valid_counts = torch.count_nonzero(mask, dim=reduc_dims)*(nb_of_levels if data_geometry == '3D' else 1.)
        res = sum_over_each_channel/valid_counts # shape [C]
        return res
    if (reduction=='normalization_mean') : # by channel, individual for sample (and level if 3D)
        sum_over_each_sample = torch.sum(tensor*mask, dim=(-2,-1)) #shape [N, (L), (C)]
        valid_counts = torch.count_nonzero(mask, dim=(-2,-1))
        mean_of_sample = sum_over_each_sample/valid_counts
        res = torch.unsqueeze(torch.unsqueeze(mean_of_sample, dim=-1), dim=-1) # shape [N, (L), (C), 1, 1]
        return res 

def get_pressure_grad(temp_var, rho_ct_ct, dx, dy, z_l) :
    g = 9.81
    dz = torch.diff(z_l)
    delta_rho = 0.5*temp_var*rho_ct_ct
    dx_rho, dy_rho = central_diffs(delta_rho)
    dx_rho = dx_rho[:,:-1,:,:]/dx[:,None,1:-1,1:-1]
    dy_rho = dy_rho[:,:-1,:,:]/dy[:,None,1:-1,1:-1]
    dx_p = torch.cumsum(dx_rho*g*dz[:,:,None,None], axis=1)   
    dy_p = torch.cumsum(dy_rho*g*dz[:,:,None,None], axis=1)
    return [dx_p, dy_p, torch.sqrt(torch.pow(dx_p,2)+torch.pow(dy_p,2))]

def evaluate_loss_with_mask(data_geometry, metrics, mask, truth, model_output, reduction='mean', normalization=True) :
    if normalization :
        normalization_coef = masked_mean(data_geometry, truth, mask, reduction='normalization_mean')
    else :
        normalization_coef = 1.
    non_reduced_non_masked_metrics = metrics(model_output/normalization_coef, truth/normalization_coef, reduction='none')
    reduced_metrics = masked_mean(data_geometry, non_reduced_non_masked_metrics, mask, reduction=reduction)
    return reduced_metrics
    
def pressure_based_MSEloss(batch, pred_sigma, target_sigma, cut_border_pix, idx_level=100, normalization=True) :
    rho_ct_ct = cut_bords(batch['rho_ct_ct'], cut_border_pix)
    dx = cut_bords(batch['e1t'], cut_border_pix)
    dy = cut_bords(batch['e2t'], cut_border_pix)
    z_l = batch['z_l']
    mask = cut_bords(batch['eroded_mask'], cut_border_pix)

    narrowed_mask = mask[:,1:-1,1:-1] # use narrowed mask since borders are cropped when computing gradient
    
    true_pres_grad_x, true_pres_grad_y, true_pres_grad_norm = get_pressure_grad(target_sigma, rho_ct_ct, dx, dy, z_l)
    pred_pres_grad_x, pred_pres_grad_y, pred_pres_grad_norm = get_pressure_grad(pred_sigma, rho_ct_ct, dx, dy, z_l)

    pres_grad_x_loss = evaluate_loss_with_mask('3D', torch.nn.functional.mse_loss, narrowed_mask, \
                                               pred_pres_grad_x[:,idx_level,:,:], true_pres_grad_x[:,idx_level,:,:], \
                                               reduction='mean', normalization=normalization)
    pres_grad_y_loss = evaluate_loss_with_mask('3D', torch.nn.functional.mse_loss, narrowed_mask, \
                                               pred_pres_grad_y[:,idx_level,:,:], true_pres_grad_y[:,idx_level,:,:], \
                                               reduction='mean', normalization=normalization)
    loss_pres_grad = pres_grad_x_loss+pres_grad_y_loss
    return loss_pres_grad

def cut_bords(tensor, nb_of_border_pix) :
    if nb_of_border_pix is None :
        return tensor
    else :
        if (len(tensor.shape) == 5) :
            return tensor[:,:, :, nb_of_border_pix:-nb_of_border_pix, nb_of_border_pix:-nb_of_border_pix] 
        if (len(tensor.shape) == 4) :
            return tensor[:,:, nb_of_border_pix:-nb_of_border_pix, nb_of_border_pix:-nb_of_border_pix] 
        if (len(tensor.shape) == 3) :
            return tensor[:, nb_of_border_pix:-nb_of_border_pix, nb_of_border_pix:-nb_of_border_pix] 
        
def expand_to_bords(tensor, nb_of_border_pix) :
    if nb_of_border_pix is None :
        return tensor
    else :
        if (len(tensor.shape) == 4) :
            new_tensor = torch.empty((tensor.shape[0],tensor.shape[1], tensor.shape[2]+2*nb_of_border_pix, tensor.shape[3]+2*nb_of_border_pix)).\
            to(tensor.device)
            new_tensor[:,:, nb_of_border_pix:-nb_of_border_pix, nb_of_border_pix:-nb_of_border_pix] = tensor
        if (len(tensor.shape) == 3) :
            new_tensor = torch.empty((tensor.shape[0], tensor.shape[1]+2*nb_of_border_pix, tensor.shape[2]+2*nb_of_border_pix)).to(tensor.device)
            new_tensor[:,nb_of_border_pix:-nb_of_border_pix, nb_of_border_pix:-nb_of_border_pix] = tensor        
        return new_tensor
    
def transform_and_stack_features(batch, features, nb_of_border_pix, normalization_features=None) :
    # check if normalization is needed
    for index, feature in enumerate(features) :
        if feature.startswith('normalized_') :
            not_normalized_feature_name = feature.replace("normalized_", "")
            if (normalization_features is None) :
                norm_feature = None
            else :
                norm_feature = normalization_features[index]
                if not(norm_feature in batch.keys()) :
                    batch = add_transformed_feature(batch, norm_feature)
            batch['normalized_'+not_normalized_feature_name] = tensor_normalize(batch[not_normalized_feature_name], batch, \
                                                                                not_normalized_feature_name, norm_feature)
        if feature.startswith('filtered_') :
            not_filt_feature_name = feature.replace("filtered_", "")
            batch['filtered_'+not_filt_feature_name] = filter_with_convolution(batch[not_filt_feature_name], convolution_kernel_size=3)
    # stack features from sample into channel (create channel dimension in tensor)
    stacked_channels = torch.stack([cut_bords(batch[key], nb_of_border_pix) for key in features])
    if (len(stacked_channels.shape) == 4): # 2d data case -> 4D cubes 
        transform = torch.permute(stacked_channels, (1,0,2,3)).to(torch.float32) #shape [N,C,H,W]
    if (len(stacked_channels.shape) == 5): # 3d data case -> 5d cubes
        transform = torch.permute(stacked_channels, (1,2,0,3,4)).to(torch.float32) #shape [N,L,C,H,W]
    return transform

def add_transformed_feature(batch, missing_feature_name) :
    if missing_feature_name.startswith('filtered_') :
        not_filt_feature_name = missing_feature_name.replace("filtered_", "")
        batch['filtered_'+not_filt_feature_name] = filter_with_convolution(batch[not_filt_feature_name], convolution_kernel_size=3)
    if missing_feature_name.startswith('sqrt_filtered_') :
        feature_name = missing_feature_name.replace("sqrt_filtered_", "")
        batch['sqrt_filtered_'+feature_name] = torch.sqrt(filter_with_convolution(batch[feature_name], convolution_kernel_size=3))
    return batch

def tensor_restore_norm(tensor, batch, reference_feature, normalization_feature=None) :
    if (len(tensor.shape) == 3) :
        std = batch['std_'+reference_feature][:,None,None]
        mean = batch['mean_'+reference_feature][:,None,None]
    if (len(tensor.shape) == 4) :
        std = batch['std_'+reference_feature][:,:,None,None]
        mean = batch['mean_'+reference_feature][:,:,None,None]
    if (len(tensor.shape) == 5) :
        std = batch['std_'+reference_feature][:,:,:,None,None]
        mean = batch['mean_'+reference_feature][:,:,:, None,None]
    if (normalization_feature is None) :
        return tensor*std+mean
    else : 
        return tensor*batch[normalization_feature]+mean

def tensor_normalize(tensor, batch, reference_feature, normalization_feature=None) :
    if (len(tensor.shape) == 3) :
        std = batch['std_'+reference_feature][:,None,None]
        mean = batch['mean_'+reference_feature][:,None,None]
    if (len(tensor.shape) == 4) :
        std = batch['std_'+reference_feature][:,:,None,None]
        mean = batch['mean_'+reference_feature][:,:,None,None]
    if (len(tensor.shape) == 5) :
        std = batch['std_'+reference_feature][:,:,:,None,None]
        mean = batch['mean_'+reference_feature][:,:,:, None,None]
    if (normalization_feature is None) :
        return (tensor-mean)/std
    else : 
        return (tensor-mean)/batch[normalization_feature]

def filter_with_convolution(tensor, convolution_kernel_size=3) :
    if (len(tensor.shape) == 4) : #[N,L,H,W]
        batch_len, nb_levels, height, width = tensor.shape
        flatten_tensor = tensor.flatten(start_dim=0, end_dim=1)[:,None,:,:]
    if (len(tensor.shape) == 3) : #[N,H,W]
        flatten_tensor = tensor[:,None,:,:]
        
    weights = torch.ones(1,1,convolution_kernel_size,convolution_kernel_size).to(tensor.device) #matrix filled with ones for averaging
    padding = torch.nn.ReplicationPad2d(convolution_kernel_size//2)  #pad borders with 1 row/column with the replicated values
    padded_tensor= padding(flatten_tensor)
    res = torch.nn.functional.conv2d(padded_tensor, weights, bias=None, stride=1, padding='valid', dilation=1, groups=1)
    res = res[:,0,:,:]
    if (len(tensor.shape) == 4) :
        res = res.unflatten(dim=0, sizes=(batch_len, nb_levels))
    return res

class torchDataset(torch.utils.data.Dataset):
    """Dataset of 2D maps of surface temperature, salinity"""

    def __init__(self, xarray_dataset, features_to_add_to_sample, auxiliary_features, height, width):
        self.features_to_add_to_sample = features_to_add_to_sample
        self.auxiliary_features = auxiliary_features
        self.height = height
        self.width = width
        
        self.data = crop_2d_maps(xarray_dataset, self.height, self.width).load()
        self.data_file_len = len(self.data.t)
        
    def __len__(self):
        return self.data_file_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            list_idx = idx.tolist()
        else :
            list_idx = idx
        selected_time_frames = self.data.isel(t=list_idx) #still xArray object
        
        # create dictionary of a sample (a batch) containig different features in numpy format. 
        # This dictionary is an intermediate step, preparing xArray data for transform into pytorch tensors
        sample = dict()
        sample['mask'] = toTorchTensor(selected_time_frames['mask'].astype(bool))
        sample['eroded_mask'] = toTorchTensor(erode_bin_mask(selected_time_frames['mask']))
        
        for feature in self.features_to_add_to_sample :
            sample['mean_'+feature] = toTorchTensor(self.data['mean_'+feature])
            sample['std_'+feature] = toTorchTensor(self.data['std_'+feature])
            sample[feature] = toTorchTensor(selected_time_frames[feature])

        for feature in self.auxiliary_features :
            sample[feature] = toTorchTensor(selected_time_frames[feature])
    
        return sample

class PyLiDataModule(pl.LightningDataModule):
    def __init__(self, cloud_data_sets, data_geometry, features_to_add_to_sample, auxiliary_features, height, width, batch_size) :
        super().__init__()
        self.cloud_data_sets = cloud_data_sets
        self.data_geometry = data_geometry
        self.features_to_add_to_sample = features_to_add_to_sample
        self.auxiliary_features = auxiliary_features
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.list_of_xr_datasets = [xr.Dataset() for i in range(len(self.cloud_data_sets))]
        self.list_of_torch_datasets = [{} for i in range(len(self.cloud_data_sets))]
        
    #def prepare_data(self) :
        # preparation of data: mean and std of the dataset (to avoid batch avg), normalization and nan filling
        for i in range(len(self.cloud_data_sets)) :
            # read file
            PERSISTENT_BUCKET = os.environ['PERSISTENT_BUCKET'] 
            if (self.data_geometry =='2D') :
                file_prefix = 'data'
            if (self.data_geometry =='3D') :
                file_prefix = 'data3D_'
            xr_dataset = xr.open_zarr(f'{PERSISTENT_BUCKET}/'+file_prefix+str(i)+'.zarr', chunks='auto')
            rename_rules_dictionary = dict({'votemper':'temp', 'sosstsst':'temp', 'vosaline' : 'saline', 'sosaline' : 'saline'})
            for name_to_replace, new_name in rename_rules_dictionary.items() :
                for var in xr_dataset.variables :
                    if (name_to_replace in var):
                        new_var_name = var.replace(name_to_replace, new_name)
                        xr_dataset = xr_dataset.rename({var : new_var_name})
            xr_dataset = xr_dataset[self.features_to_add_to_sample + self.auxiliary_features + ['mask']]
            for feature in self.features_to_add_to_sample :
                # reapply mask (to avoid issues with nans written in netcdf files)
                xr_dataset[feature] = xr_dataset[feature].where(xr_dataset.mask>0)
                # compute mean, median and std for each level (since temperature/salinity may change a lot with the depth)
                xr_dataset['mean_'+feature] = (xr_dataset[feature].mean(dim=['t', 'x_c', 'y_c']))
                xr_dataset['std_'+feature] = (xr_dataset[feature].std(dim=['t', 'x_c', 'y_c']))
                # fill nans with mean (doesn't the number to be fillted in matter since they will be masked, 
                # but they have to be filled with any numbers so that nans do not propagate everywhere) 
                xr_dataset[feature] = xr_dataset[feature].fillna(xr_dataset['mean_'+feature])
            # save result in a list
            self.list_of_xr_datasets[i] = xr_dataset
            self.list_of_torch_datasets[i] = torchDataset(xr_dataset, self.features_to_add_to_sample, self.auxiliary_features, self.height, self.width)
            
    def setup(self, stage: str) :
        if (stage == 'fit') :
        # takes first 60% of time snapshots for training
            self.train_dataset = torch.utils.data.ConcatDataset([torch.utils.data.Subset(dataset, \
                                                                                     indices=range(0,int(0.6*len(dataset)))) \
                                                                                     for dataset in self.list_of_torch_datasets])
        # takes last 20% of time snapshots for validation (we keep a gap to have validation data decorrelated from trainig data)
            self.val_dataset = torch.utils.data.ConcatDataset([torch.utils.data.Subset(dataset, \
                                                                                     indices=range(int(0.8*len(dataset)),len(dataset))) \
                                                                                     for dataset in self.list_of_torch_datasets])
        # same for test
        if (stage == 'test') :
            self.test_datasets_byregion = [torch.utils.data.Subset(dataset, indices=range(int(0.8*len(dataset)),len(dataset))) \
                                                               for dataset in self.list_of_torch_datasets]
            self.test_dataset_all_data = torch.utils.data.ConcatDataset([torch.utils.data.Subset(dataset, \
                                                                                     indices=range(int(0.8*len(dataset)),len(dataset))) \
                                                                                     for dataset in self.list_of_torch_datasets])
            
    def train_dataloader(self) :
        # create training dataloadder from train_dataset with shuffling with given batch size
        return torch.utils.data.DataLoader(self.train_dataset, \
                                           batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=0)
    
    def val_dataloader(self) :
        # create training dataloadder from val_dataset without shuffling with the same batch size
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=True, num_workers=0) 
    
    def test_dataloader(self) :
        # create a LIST of dataloaders (a dataloader for each dataset) - to enable diagnostics in each region/season individually 
        # batch size is equal to the dataset length, i.e. there is ONLY 1 batch with all dataset inside (can be better since there is no optimisation in testing)
        return ([torch.utils.data.DataLoader(dataset, batch_size=len(dataset), drop_last=True, num_workers=0) for dataset in self.test_datasets_byregion] 
                + [torch.utils.data.DataLoader(self.test_dataset_all_data, batch_size=len(self.test_dataset_all_data), num_workers=0)])
    
    def teardown(self, stage : str) :
        if (stage == 'fit') :
            # clean train and val datasets to free memory
            del self.train_dataset, self.val_dataset
        # if (stage == 'test') :
        #     del self.test_datasets   
        # if (stage == 'predict') :
        #     del self.test_datasets   
        
class GenericPyLiModule(pl.LightningModule):
    def __init__(self, torch_model, input_features, output_features, output_units, loss, optimizer, learning_rate, \
                 input_normalization_features=None, loss_normalization=False):
        super().__init__()
        self.torch_model = torch_model
        self.input_features = input_features
        self.output_features = output_features
        self.output_units = output_units
        self.input_normalization_features = input_normalization_features
        self.loss_normalization = loss_normalization
        
        self.loss = loss
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss_normalization = loss_normalization
        
        ## initialization of weights
        #torch_model.weight.data = torch.Tensor([1.0])

        #construct list of names of features to be predicted
        self.list_of_features_to_predict=list()
        for i, feature in enumerate(self.output_features) :
            self.list_of_features_to_predict.append(feature)
            if feature.startswith('normalized_') :
                # if model output is a normalized feature then compute also the non-normalized feature (for the diagnostics)
                not_normalized_feature = feature.replace("normalized_", "")
                self.list_of_features_to_predict.append(not_normalized_feature)
                
        self.data_geometry = self.torch_model.data_geometry
    
    def common_step(self, batch, batch_idx) :
        x = transform_and_stack_features(batch, self.input_features, self.torch_model.cut_border_pix_input, self.input_normalization_features)
        y_true = transform_and_stack_features(batch, self.output_features, self.torch_model.cut_border_pix_output)
        mask = cut_bords(batch['eroded_mask'], self.torch_model.cut_border_pix_output)

        if (self.output_units is None) :
            y_model = self.torch_model(x)
        else :
            y_units = transform_and_stack_features(batch, self.output_units, self.torch_model.cut_border_pix_output)
            y_model = y_units*self.torch_model(x)
        
        logs = dict()
        first_layer_weights = list(self.torch_model.__dict__['_modules'].values())[0].weight # for logging
        first_weight = np.array(first_layer_weights.cpu().detach().numpy()).flat[0]
        logs['first_weight'] = np.array(first_layer_weights.cpu().detach().numpy()).flat[0]
        
        if (self.loss=='pressure_based_MSEloss') :
            if (self.data_geometry != '3D') :
                print('ERROR: pressure based loss is available only for 3D data')
                return
            index_of_temp_var_feature = self.output_features.index('temp_var')
            pred_sigma = y_model[:, :, index_of_temp_var_feature, :, :]
            target_sigma = y_true[:, :, index_of_temp_var_feature, :, :]
            loss_pres_grad = pressure_based_MSEloss(batch, pred_sigma, target_sigma, \
                                                    self.torch_model.cut_border_pix_output, \
                                                    idx_level=100, normalization=True)
            loss_val = evaluate_loss_with_mask('3D', torch.nn.functional.mse_loss, mask, y_model, y_true, \
                                       reduction='mean', normalization=True)
            alpha=1e-6
            loss_total = alpha*loss_pres_grad+loss_val
            logs = logs | dict({'loss_train' : loss_total,
                 'loss_pressure' : loss_pres_grad,
                 'loss_value' : loss_val})
        else :
            loss_val = evaluate_loss_with_mask(self.data_geometry, self.loss, mask, y_model, y_true, \
                                               reduction='mean', normalization=self.loss_normalization)  
            logs = logs | dict({'loss_train' : loss_val})
        return logs
        
    def training_step(self, batch, batch_idx) :
        logs = self.common_step(batch, batch_idx)
        self.log_dict(logs, on_step=False, on_epoch=True)
        return logs['loss_train']

    # validation logics (is evaluated during the training, but the data is not used to the optimization loop)
    def validation_step(self, batch, batch_idx) :
        logs = self.common_step(batch, batch_idx)
        self.log_dict(dict({'loss_validation' : logs['loss_train']}), on_step=False, on_epoch=True)
    
    # gives model output in a form of a dictionary of batches of 2d fields
    def predict_step(self, batch, batch_idx, dataloader_idx) :
        x = transform_and_stack_features(batch, self.input_features, self.torch_model.cut_border_pix_input)
        
        output_tensor = self.torch_model(x)
        if (self.data_geometry == '2D') :
            batch_len, nb_of_channels, output_h, output_w = output_tensor.shape
        if (self.data_geometry == '3D') :
            batch_len, nb_of_levels, nb_of_channels, output_h, output_w = output_tensor.shape

        if not(self.output_units is None) : # if output of the model is dimensionless -> compute output with physical units
            y_units = transform_and_stack_features(batch, self.output_units, self.torch_model.cut_border_pix_output)
            output_tensor_units = output_tensor*y_units
            
        # construct the dictionary of the predicted features by decomposing the channels into dictionary entities
        pred = dict()
        if (self.data_geometry == '2D') :
            channel_dim = 1
        if (self.data_geometry == '3D') :
            channel_dim = 2
        for i, feature in enumerate(self.output_features) :
            if (self.output_units is None) :
                pred[feature] = output_tensor.select(dim=channel_dim, index=i)
            else :
                pred[feature+'_dimless'] = output_tensor.select(dim=channel_dim, index=i)
                pred[feature] = output_tensor_units.select(dim=channel_dim, index=i)
            # if some outputs are normalized then compute also result in the restored units (not normalized)
            if feature.startswith('normalized_') :
                not_normalized_feature_name = feature.replace("normalized_", "")
                pred[not_normalized_feature_name] = PyLiDataModule.tensor_restore_norm(pred[feature], batch, not_normalized_feature)
        
        # save the mask and masked outputs (use the eroded mask)
        for i, feature in enumerate(self.list_of_features_to_predict) :
            if (self.data_geometry == '2D') :
                 pred['mask'] = batch['eroded_mask']
            if (self.data_geometry == '3D') :
                pred['mask'] = batch['eroded_mask'][:,None,:,:]
            pred[feature+'_masked'] = expand_to_bords(pred[feature], self.torch_model.cut_border_pix_output)
            pred[feature+'_masked'] = pred[feature+'_masked'].where(pred['mask'], torch.ones_like(pred[feature+'_masked'])*np.nan)
        return pred 
    
    # testing logic - to evaluate the model after training
    def test_step(self, batch, batch_idx, dataloader_idx) :
        pred = self.predict_step(batch, batch_idx, dataloader_idx)
        mask = cut_bords(batch['eroded_mask'], self.torch_model.cut_border_pix_output)
        
        test_dict = dict({'relative_loss' : dict(), 'abs_loss' : dict(),\
                          'relative_loss_grad' : dict(), 'abs_loss_grad' : dict(), \
                          'corr_coef' : dict(), 'corr_coef_grad' : dict()})
        dict_for_log = dict()
        
        # global metrics
        for i, feature in enumerate(self.list_of_features_to_predict) :
            truth = cut_bords(batch[feature], self.torch_model.cut_border_pix_output)
            model_output = pred[feature]  # use unmasked prediction here, mask will be applied further on error tensor
            
            test_dict['relative_loss'][feature] = evaluate_loss_with_mask(self.data_geometry, torch.nn.functional.mse_loss, \
                                                                                    mask, model_output, truth, \
                                                                                    reduction='mean', normalization=True)
            test_dict['abs_loss'][feature] = evaluate_loss_with_mask(self.data_geometry, torch.nn.functional.mse_loss, \
                                                                                    mask, model_output, truth, \
                                                                                    reduction='mean', normalization=False)
            test_dict['corr_coef'][feature] = torch.corrcoef(torch.vstack((torch.flatten(model_output).view(1,-1), \
                                                              torch.flatten(truth).view(1,-1))))[1,0]
            # metrics on horizontal gradients
            model_output_grad = finite_diffs_sqr_2d_map(model_output)
            truth_grad = finite_diffs_sqr_2d_map(truth)
            test_dict['relative_loss_grad'][feature] = evaluate_loss_with_mask(self.data_geometry, torch.nn.functional.mse_loss, \
                                                                     mask[:,1:-1,1:-1], model_output_grad, truth_grad, \
                                                                      reduction='mean', normalization=True)
            test_dict['abs_loss_grad'][feature] = evaluate_loss_with_mask(self.data_geometry, torch.nn.functional.mse_loss, \
                                                                     mask[:,1:-1,1:-1], model_output_grad, truth_grad, \
                                                                      reduction='mean', normalization=False)
            test_dict['corr_coef_grad'][feature] = torch.corrcoef(torch.vstack((torch.flatten(model_output_grad).view(1,-1), \
                                                              torch.flatten(truth_grad).view(1,-1))))[1,0]

        # pressure at 100th level
        if (self.data_geometry == '3D') :
            idx_level = 100
            true_temp_var = cut_bords(batch['temp_var'], self.torch_model.cut_border_pix_output)
            model_temp_var = pred['temp_var']
            test_dict['relative_loss']['pressure_grad'] = pressure_based_MSEloss(batch, model_temp_var, true_temp_var, \
                                                    self.torch_model.cut_border_pix_output, \
                                                    idx_level=idx_level, normalization=True)
            test_dict['abs_loss']['pressure_grad'] = pressure_based_MSEloss(batch, model_temp_var, true_temp_var, \
                                                    self.torch_model.cut_border_pix_output, \
                                                    idx_level=idx_level, normalization=False)
        for metrics in list(test_dict.keys()) : 
            for feature in list(test_dict[metrics].keys()) : 
                dict_for_log.update({(metrics+'_'+feature) : test_dict[metrics][feature]})
        self.log_dict(dict_for_log)

    def configure_optimizers(self) :
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer

class lin_regr_model(torch.nn.Module):
    def __init__(self, data_geometry, nb_of_input_features, nb_of_output_features):
        super().__init__()
        self.data_geometry = data_geometry
        self.nb_of_input_features = nb_of_input_features
        self.nb_of_output_features = nb_of_output_features
        
        self.cut_border_pix_output = None
        self.cut_border_pix_input = None
        
        self.lin1 = torch.nn.Linear(self.nb_of_input_features, self.nb_of_output_features, bias=False)
        
        # initialization 
        self.lin1.weight.data = torch.Tensor([[0.1]])

    def forward(self, x):
        if (self.data_geometry == '3D') :
            batch_len, nb_of_levels, nb_of_channels, output_h, output_w = x.shape
            # deattach levels into batch entities by flattening
            res = x.flatten(start_dim=0, end_dim=1) # shape [N',C,H,W]
            new_batch_len = batch_len*nb_of_levels
        if (self.data_geometry == '2D') :
            new_batch_len, nb_of_channels, output_h, output_w = x.shape
            res = x 
        
        # first split the input 4D torch tensor into individual pixels (equivalent to patches of size 1x1)
        res = torch.nn.functional.unfold(res, kernel_size=1, dilation=1, padding=0, stride=1)
        res = torch.permute(res, dims=(0,2,1))
        res = torch.flatten(res, end_dim=1).to(torch.float32)
        
        # perform linear regression
        res = self.lin1(res)
        
        # reshape the model output back to a 4D torch tensor
        res = torch.permute(res.unflatten(dim=0, sizes=[new_batch_len,-1]),dims=(0,2,1))
        res = torch.nn.functional.fold(res, output_size=(output_h,output_w), kernel_size=1, dilation=1, padding=0, stride=1)
        
        if (self.data_geometry == '3D') :
            # unflatten the levels back
            res = res.unflatten(dim=0, sizes=(batch_len, nb_of_levels))
        return res
    
class FCNN(torch.nn.Module):
    def __init__(self, data_geometry, nb_of_input_features, nb_of_output_features, input_patch_size, output_patch_size, \
                 activation_function = torch.nn.functional.relu, int_layer_width=50):
        super().__init__()
        self.data_geometry = data_geometry
        self.input_patch_size = input_patch_size
        self.output_patch_size = output_patch_size
        self.activation_function = activation_function
        self.int_layer_width = int_layer_width
        
        self.lin1 = torch.nn.Linear(nb_of_input_features*input_patch_size**2, int_layer_width, bias=True)
        self.lin2 = torch.nn.Linear(int_layer_width, int_layer_width, bias=True)
        self.lin3 = torch.nn.Linear(int_layer_width, nb_of_output_features*output_patch_size**2, bias=True)
        
        self.cut_border_pix_output = self.input_patch_size//2 - self.output_patch_size//2
        if (self.cut_border_pix_output < 1) :
            self.cut_border_pix_output = None
        self.cut_border_pix_input = None

    def forward(self, x):
        if (self.data_geometry =='3D') :
            batch_len, nb_of_levels, nb_of_channels = x.shape[0:3]
            output_h = x.shape[3]-2*(self.cut_border_pix_output or 0)
            output_w = x.shape[4]-2*(self.cut_border_pix_output or 0)
            # deattach levels into batch entities by flattening
            res = x.flatten(start_dim=0, end_dim=1) # shape [N',C,H,W]
            new_batch_len = batch_len*nb_of_levels
        if (self.data_geometry =='2D') :
            new_batch_len, nb_of_channels = x.shape[0:2]
            output_h = x.shape[2]-2*(self.cut_border_pix_output or 0)
            output_w = x.shape[3]-2*(self.cut_border_pix_output or 0)
            res = x
        
        # create patches of size 'input_patch_size' and join them into batches (zero padding - will remove border pixels)
        res = torch.nn.functional.unfold(res, kernel_size=self.input_patch_size, dilation=1, padding=0, stride=1)
        res = torch.permute(res, dims=(0,2,1))
        res = torch.flatten(res, end_dim=1)
        
        # pass though the FCNN
        res = self.lin1(res)
        res = self.activation_function(res)
        res = self.lin2(res)
        res = self.activation_function(res)
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
        
        if (self.data_geometry =='3D') :
            # unflatten the levels
            res = res.unflatten(dim=0, sizes=(batch_len, nb_of_levels))
        
        return res
    
class CNN(torch.nn.Module):
    def __init__(self, data_geometry, nb_of_input_features, nb_of_output_features, padding='same', padding_mode='replicate', \
                 kernel_size=3, int_layer_width=64, activation_function = torch.nn.functional.relu):
        super().__init__()
        self.data_geometry = data_geometry
        self.padding = padding
        self.kernel_size = kernel_size
        self.padding_mode = 'replicate'
        self.activation_function = activation_function
        
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
        if (self.data_geometry == '3D') :
            nb_of_levels = x.shape[1]
            # deattach levels into batch entities by flattening
            res = x.flatten(start_dim=0, end_dim=1) # shape [N',C,H,W]
        else :
            res = x
        
        res = self.conv1(res)
        res = self.activation_function(res)
        res = self.conv2(res)
        res = self.activation_function(res)
        res = self.conv3(res)
        
        if (self.data_geometry == '3D') :
            # unflatten the levels
            res = res.unflatten(dim=0, sizes=(batch_len, nb_of_levels))
        return res       