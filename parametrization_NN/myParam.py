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

def get_pressure_grad(votemper_var, rho_ct_ct, dx, dy) :
    g = 9.81
    dz = torch.Tensor([ 1.0000261,  1.1568018,  1.3141856,  1.4721599,  1.6307058,
        1.7898054,  1.94944  ,  2.1095896,  2.270236 ,  2.4313574,
        2.5929375,  2.7549553,  2.917385 ,  3.0802155,  3.2434177,
        3.4069748,  3.5708656,  3.7350693,  3.899559 ,  4.0643234,
        4.229328 ,  4.394562 ,  4.5600014,  4.7256126,  4.8913956,
        5.0572968,  5.2233276,  5.38945  ,  5.555626 ,  5.7218704,
        5.888115 ,  6.0543823,  6.2206116,  6.3868027,  6.5529404,
        6.7189636,  6.884903 ,  7.0506744,  7.216324 ,  7.38179  ,
        7.547043 ,  7.7120667,  7.876877 ,  8.041382 ,  8.205627 ,
        8.369553 ,  8.533157 ,  8.696396 ,  8.859283 ,  9.021744 ,
        9.183807 ,  9.345459 ,  9.506622 ,  9.667328 ,  9.827545 ,
        9.987244 , 10.146393 , 10.305023 , 10.463043 , 10.620514 ,
       10.777374 , 10.933563 , 11.089172 , 11.24411  , 11.398346 ,
       11.55191  , 11.704773 , 11.856903 , 12.008301 , 12.158936 ,
       12.308777 , 12.457916 , 12.606201 , 12.753632 , 12.900391 ,
       13.046143 , 13.191162 , 13.335266 , 13.478516 , 13.62085  ,
       13.762329 , 13.902893 , 14.042603 , 14.181274 , 14.319092 ,
       14.455872 , 14.591858 , 14.726746 , 14.860718 , 14.993713 ,
       15.125732 , 15.256775 , 15.38678  , 15.515808 , 15.64386  ,
       15.770813 , 15.896851 , 16.02179  , 16.14569  , 16.268677 ,
       16.390503 , 16.511353 , 16.631165 , 16.749939 , 16.867676 ,
       16.984314 ]).to(votemper_var.device)
    delta_rho = 0.5*votemper_var*rho_ct_ct
    dx_rho, dy_rho = central_diffs(delta_rho)
    dx_rho = dx_rho[:,:-1,:,:]/dx[:,None,1:-1,1:-1]
    dy_rho = dy_rho[:,:-1,:,:]/dy[:,None,1:-1,1:-1]
    dx_p = torch.cumsum(dx_rho*g*dz[None,:,None,None], axis=1)   
    dy_p = torch.cumsum(dy_rho*g*dz[None,:,None,None], axis=1)
    return [dx_p, dy_p, torch.sqrt(torch.pow(dx_p,2)+torch.pow(dy_p,2))]

def evaluate_tensor_metrics_with_mask(data_geometry, metrics, mask, truth, model_output, reduction='mean') :
    if (data_geometry == '2D') :
        if (len(model_output.shape) == 4) : # 4D tensor with C features
            batch_len, nb_of_channels, output_h, output_w = model_output.shape  
            valid_mask_counts = torch.count_nonzero(mask)*nb_of_channels
            mask = mask[:,None,:,:]
        if (len(model_output.shape) == 3) : # 1 feature (1 channel)- 3D tensor
            batch_len, output_h, output_w = model_output.shape  
            valid_mask_counts = torch.count_nonzero(mask)
            mask = mask
    if (data_geometry == '3D') :
        if (len(model_output.shape) == 5) : # full 5D tensor
            batch_len, nb_of_levels, nb_of_channels, output_h, output_w = model_output.shape  
            valid_mask_counts = torch.count_nonzero(mask)*nb_of_levels*nb_of_channels
            mask = mask[:,None,None,:,:]
        if (len(model_output.shape) == 4) : # 1 feature (1 channel)
            batch_len, nb_of_levels, output_h, output_w = model_output.shape  
            valid_mask_counts = torch.count_nonzero(mask)*nb_of_levels
            mask = mask[:,None,:,:]
        if (len(model_output.shape) == 3) : # 1 feature (1 channel) and 1 level
            batch_len, output_h, output_w = model_output.shape  
            valid_mask_counts = torch.count_nonzero(mask)
            mask = mask[:,:,:]

    if (reduction=='none') : 
        return metrics(model_output*mask, truth*mask, reduction='none')

    total_metrics = metrics(model_output*mask, truth*mask, reduction='sum')
    if (reduction=='mean') : 
        return (total_metrics/valid_mask_counts)
    if (reduction=='sum') : 
        return (total_metrics)
    if (reduction=='vertical') :
        sum_over_each_layer = torch.sum(metrics(model_output*mask, truth*mask, reduction='none'), dim=(2,3))
        valid_counts_each_layer = torch.count_nonzero(mask, dim=(1,2))
        vertical_profile_of_each_sample = sum_over_each_layer/valid_counts_each_layer
        return torch.mean(vertical_profile_of_each_sample, dim=0)
    if (reduction=='horizontal') :
        sum_over_depth_at_each_point = torch.sum(metrics(model_output*mask, truth*mask, reduction='none'), dim=1)
        valid_counts = torch.count_nonzero(mask)
        horizontal_error_of_each_sample = sum_over_depth_at_each_point/valid_counts
        return torch.mean(horizontal_error_of_each_sample, dim=0)
    
def pressure_based_MSEloss(pred_sigma, target_sigma, rho_ct_ct, idx_level, mask, dx, dy) :
    true_pres_grad_x, true_pres_grad_y, true_pres_grad_norm = get_pressure_grad(target_sigma, rho_ct_ct, dx, dy)
    pred_pres_grad_x, pred_pres_grad_y, pred_pres_grad_norm = get_pressure_grad(pred_sigma, rho_ct_ct, dx, dy)

    grad_x_loss = evaluate_tensor_metrics_with_mask('3D', torch.nn.functional.mse_loss, mask[:,1:-1,1:-1], \
                                                        pred_pres_grad_x[:,idx_level,:,:], true_pres_grad_x[:,idx_level,:,:], reduction='mean')
    grad_y_loss = evaluate_tensor_metrics_with_mask('3D', torch.nn.functional.mse_loss, mask[:,1:-1,1:-1], \
                                                        pred_pres_grad_y[:,idx_level,:,:], true_pres_grad_y[:,idx_level,:,:], reduction='mean')
    grad_loss = grad_x_loss+grad_y_loss
    return grad_loss

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
    
def transform_features(batch, features, nb_of_border_pix) :
    # check if normalization is needed
    for feature in features :
        if feature.startswith('normalized_') :
            not_normalized_feature_name = feature.replace("normalized_", "")
            batch['normalized_'+not_normalized_feature_name] = tensor_normalize(batch[not_normalized_feature_name], batch, not_normalized_feature_name)
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

def tensor_restore_norm(tensor, batch, reference_feature) :
    if (len(tensor.shape) == 3) :
        std = batch['std_'+reference_feature][:,None,None]
        mean = batch['mean_'+reference_feature][:,None,None]
    if (len(tensor.shape) == 4) :
        std = batch['std_'+reference_feature][:,:,None,None]
        mean = batch['mean_'+reference_feature][:,:,None,None]
    if (len(tensor.shape) == 5) :
        std = batch['std_'+reference_feature][:,:,:,None,None]
        mean = batch['mean_'+reference_feature][:,:,:, None,None]
    return tensor*std+mean

def tensor_normalize(tensor, batch, reference_feature) :
    if (len(tensor.shape) == 3) :
        std = batch['std_'+reference_feature][:,None,None]
        mean = batch['mean_'+reference_feature][:,None,None]
    if (len(tensor.shape) == 4) :
        std = batch['std_'+reference_feature][:,:,None,None]
        mean = batch['mean_'+reference_feature][:,:,None,None]
    if (len(tensor.shape) == 5) :
        std = batch['std_'+reference_feature][:,:,:,None,None]
        mean = batch['mean_'+reference_feature][:,:,:, None,None]
    return (tensor-mean)/std

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
            xr_dataset = xr.open_zarr(f'{PERSISTENT_BUCKET}/'+file_prefix+str(i)+'.zarr', chunks='auto')\
            [self.features_to_add_to_sample + self.auxiliary_features + ['mask']]
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
            self.test_datasets = [torch.utils.data.Subset(dataset, indices=range(int(0.8*len(dataset)),len(dataset))) \
                                                               for dataset in self.list_of_torch_datasets]
            
                
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
        return [torch.utils.data.DataLoader(dataset, batch_size=len(dataset), drop_last=True, num_workers=0) for dataset in self.test_datasets]
    
    def teardown(self, stage : str) :
        if (stage == 'fit') :
            # clean train and val datasets to free memory
            del self.train_dataset, self.val_dataset
        # if (stage == 'test') :
        #     del self.test_datasets   
        # if (stage == 'predict') :
        #     del self.test_datasets   
        
class GenericPyLiModule(pl.LightningModule):
    def __init__(self, torch_model, input_features, output_features, output_units, loss, optimizer, learning_rate):
        super().__init__()
        self.torch_model = torch_model
        self.input_features = input_features
        self.output_features = output_features
        self.output_units = output_units
        
        self.loss = loss
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        
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
    
    def step(self, batch, batch_idx) :
        x = transform_features(batch, self.input_features, self.torch_model.cut_border_pix_input)
        y_true = transform_features(batch, self.output_features, self.torch_model.cut_border_pix_output)
        mask = cut_bords(batch['eroded_mask'], self.torch_model.cut_border_pix_output)

        if (self.output_units is None) :
            y_model = self.torch_model(x)
        else :
            y_units = transform_features(batch, self.output_units, self.torch_model.cut_border_pix_output)
            y_model = y_units*self.torch_model(x)
            
        first_layer_weights = list(self.torch_model.__dict__['_modules'].values())[0].weight # for logging
        
        if (self.loss=='pressure_based_MSEloss') :
            if (self.data_geometry != '3D') :
                print('ERROR: pressure based loss is available only for 3D data')
                return
            rho_ct_ct = cut_bords(batch['rho_ct_ct'], self.torch_model.cut_border_pix_output)
            dx = cut_bords(batch['e1t'], self.torch_model.cut_border_pix_output)
            dy = cut_bords(batch['e2t'], self.torch_model.cut_border_pix_output)
            pred_sigma = y_model[:, :, 0, :, :]
            target_sigma = y_true[:, :, 0, :, :]
            idx_level=100
            loss_pres = pressure_based_MSEloss(pred_sigma, target_sigma, rho_ct_ct, idx_level, mask, dx, dy)
            loss_val = evaluate_tensor_metrics_with_mask(self.data_geometry, torch.nn.functional.mse_loss, mask, y_model, y_true)
            loss_total = 1e6*loss_pres+loss_val
            logs = dict({'loss_train' : loss_total,
                           'loss_pressure' : loss_pres,
                           'loss_val' : loss_val,
                      'first_weight' : np.array(first_layer_weights.cpu().detach().numpy()).flat[0]})
        else :
            loss_val = evaluate_tensor_metrics_with_mask(self.data_geometry, self.loss, mask, y_model, y_true)  
            logs= dict({'loss_train' : loss_val,
                      'first_weight' : np.array(first_layer_weights.cpu().detach().numpy()).flat[0]})   
        return logs
        
    def training_step(self, batch, batch_idx) :
        logs = self.step(batch, batch_idx)
        self.log_dict(logs, on_step=False, on_epoch=True)
        return logs['loss_train']

    # validation logics (is evaluated during the training, but the data is not used to the optimization loop)
    def validation_step(self, batch, batch_idx) :
        logs = self.step(batch, batch_idx)
        self.log_dict({'loss_validation' : logs['loss_train']}, on_step=False, on_epoch=True)
    
    # gives model output in a form of a dictionary of batches of 2d fields
    def predict_step(self, batch, batch_idx, dataloader_idx) :
        x = transform_features(batch, self.input_features, self.torch_model.cut_border_pix_input)
        
        output_tensor = self.torch_model(x)
        if (self.data_geometry == '2D') :
            batch_len, nb_of_channels, output_h, output_w = output_tensor.shape
        if (self.data_geometry == '3D') :
            batch_len, nb_of_levels, nb_of_channels, output_h, output_w = output_tensor.shape

        if not(self.output_units is None) : # if output of the model is dimensionless -> compute output with physical units
            y_units = transform_features(batch, self.output_units, self.torch_model.cut_border_pix_output)
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
        
        test_dict = dict({'loss_val' : dict(), 'loss_grad' : dict(), 'corr_coef' : dict(), 'corr_coef_grad' : dict()})
        dict_for_log = dict()
        
        # global metrics
        for i, feature in enumerate(self.list_of_features_to_predict) :
            truth = cut_bords(batch[feature], self.torch_model.cut_border_pix_output)
            model_output = pred[feature]  # use unmasked prediction here, mask will be applied further on error tensor
            
            test_dict['loss_val'][feature] = evaluate_tensor_metrics_with_mask(self.data_geometry, torch.nn.functional.mse_loss, \
                                                                                    mask, model_output, truth, reduction='mean')
            test_dict['corr_coef'][feature] = torch.corrcoef(torch.vstack((torch.flatten(model_output).view(1,-1), \
                                                              torch.flatten(truth).view(1,-1))))[1,0]
            # metrics on horizontal gradients
            model_output_grad = finite_diffs_sqr_2d_map(model_output)
            truth_grad = finite_diffs_sqr_2d_map(truth)
            test_dict['loss_grad'][feature] = evaluate_tensor_metrics_with_mask(self.data_geometry, torch.nn.functional.mse_loss, \
                                                                                     mask[:,1:-1,1:-1], model_output_grad, \
                                                                                     truth_grad, reduction='mean')
            test_dict['corr_coef_grad'][feature] = torch.corrcoef(torch.vstack((torch.flatten(model_output_grad).view(1,-1), \
                                                              torch.flatten(truth_grad).view(1,-1))))[1,0]

        # pressure at 100th level
        if (self.data_geometry == '3D') :
            idx_level = 100
            true_votemper_var = cut_bords(batch['votemper_var'], self.torch_model.cut_border_pix_output)
            rho_ct_ct = cut_bords(batch['rho_ct_ct'], self.torch_model.cut_border_pix_output)
            dx = cut_bords(batch['e1t'], self.torch_model.cut_border_pix_output)
            dy = cut_bords(batch['e2t'], self.torch_model.cut_border_pix_output)
            model_votemper_var = pred['votemper_var']
            true_pres_grad_x, true_pres_grad_y, true_pres_grad_norm = get_pressure_grad(true_votemper_var, rho_ct_ct, dx, dy)
            pred_pres_grad_x, pred_pres_grad_y, pred_pres_grad_norm = get_pressure_grad(model_votemper_var, rho_ct_ct, dx, dy)
            test_dict['loss_val']['pressure_grad_x'] = evaluate_tensor_metrics_with_mask(self.data_geometry, torch.nn.functional.mse_loss, mask[:,1:-1,1:-1], \
                                                                             pred_pres_grad_x[:,idx_level,:,:], true_pres_grad_x[:,idx_level,:,:], reduction='mean')
            test_dict['loss_val']['pressure_grad_y'] = evaluate_tensor_metrics_with_mask(self.data_geometry, torch.nn.functional.mse_loss, mask[:,1:-1,1:-1], \
                                                                             pred_pres_grad_y[:,idx_level,:,:], true_pres_grad_y[:,idx_level,:,:], reduction='mean')
            test_dict['loss_val']['pressure_grad'] = test_dict['loss_val']['pressure_grad_x'] + test_dict['loss_val']['pressure_grad_y']
        
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
    def __init__(self, data_geometry, nb_of_input_features, nb_of_output_features, input_patch_size, output_patch_size, int_layer_width=50):
        super().__init__()
        self.data_geometry = data_geometry
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
        
        if (self.data_geometry =='3D') :
            # unflatten the levels
            res = res.unflatten(dim=0, sizes=(batch_len, nb_of_levels))
        
        return res
    
class CNN(torch.nn.Module):
    def __init__(self, data_geometry, nb_of_input_features, nb_of_output_features, padding='same', padding_mode='replicate', \
                 kernel_size=3, int_layer_width=64):
        super().__init__()
        self.data_geometry = data_geometry
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
        if (self.data_geometry == '3D') :
            nb_of_levels = x.shape[1]
            # deattach levels into batch entities by flattening
            res = x.flatten(start_dim=0, end_dim=1) # shape [N',C,H,W]
        else :
            res = x
        
        res = self.conv1(res)
        res = torch.nn.functional.relu(res)
        res = self.conv2(res)
        res = torch.nn.functional.relu(res)
        res = self.conv3(res)
        
        if (self.data_geometry == '3D') :
            # unflatten the levels
            res = res.unflatten(dim=0, sizes=(batch_len, nb_of_levels))
        return res       

