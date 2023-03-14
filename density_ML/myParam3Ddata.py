import numpy as np
import xarray as xr
import torch
import pytorch_lightning as pl
from scipy import ndimage
import itertools
import os

# iterates over time snapshots in an xarray data file and transforms it into a dictionary of pytorch tensors, also performs normalization
class torchDataset(torch.utils.data.Dataset):
    """Dataset of 2D maps of surface temperature, salinity"""

    def __init__(self, xarray_dataset, features_to_add_to_sample, auxiliary_features, transform=None):
        self.transform = transform
        self.features_to_add_to_sample = features_to_add_to_sample
        self.auxiliary_features = auxiliary_features
        full_data_file_len = len(xarray_dataset.t)
        
        h = 45 # height of images
        w = 40 # width of images
        self.data = (xarray_dataset.isel(x_c=slice(None,w), y_c=slice(None,h))).load()
        self.data_file_len = len(self.data.t)
        
    def __len__(self):
        return self.data_file_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            list_idx = idx.tolist()
        else :
            list_idx = idx
        selected_time_frames = self.data.isel(t=list_idx)
        
        # create dictionary of a sample (a batch) containig different features in numpy format. 
        # This dictionary is an intermediate step, preparing xArray data for trasform into pytorch tensors
        sample = dict()
        sample['mask'] = (selected_time_frames['mask'].values).astype(bool)
        #sample['mask'] = selected_time_frames['votemper'].where(not(selected_time_frames['votemper'].isnan())).values.astype(bool)
        erosion_structure_matrix = np.array([(0,0,1,0,0), (0,1,1,1,0), (1,1,1,1,1), (0,1,1,1,0), (0,0,1,0,0)])
        sample['eroded_mask'] = ndimage.binary_erosion(selected_time_frames['mask'].values, structure=erosion_structure_matrix)
        for feature in self.features_to_add_to_sample :
            sample['mean_'+feature] = self.data['mean_'+feature].values
            sample['std_'+feature] = self.data['std_'+feature].values
            sample[feature] = selected_time_frames[feature].values
            sample['normalized_'+feature] = selected_time_frames['normalized_'+feature].values
        for feature in self.auxiliary_features :
            sample[feature] = selected_time_frames[feature].values
        if self.transform:
            sample = self.transform(sample, self.features_to_add_to_sample)
        return sample
    
class ToTensor(object):
    """Convert each numpy array in sample to Tensors."""
    def __call__(self, sample, features_to_add_to_sample):
        transformed_sample=sample.copy()
        for feature in features_to_add_to_sample :
            transformed_sample[feature] = torch.tensor(sample[feature])
        return transformed_sample
    
class PyLiDataModule(pl.LightningDataModule):
    def __init__(self, cloud_data_sets, features_to_add_to_sample, auxiliary_features, batch_size) :
        super().__init__()
        self.cloud_data_sets = cloud_data_sets
        self.features_to_add_to_sample = features_to_add_to_sample
        self.auxiliary_features = auxiliary_features
        self.batch_size = batch_size
        self.list_of_xr_datasets = [xr.Dataset() for i in range(len(self.cloud_data_sets))]
        self.list_of_torch_datasets = [{} for i in range(len(self.cloud_data_sets))]
        
    #def prepare_data(self) :
        # preparation of data: mean and std of the dataset (to avoid batch avg), normalization and nan filling
        for i in range(len(self.cloud_data_sets)) :
            # read file
            PERSISTENT_BUCKET = os.environ['PERSISTENT_BUCKET'] 
            xr_dataset = xr.open_zarr(f'{PERSISTENT_BUCKET}/data3D_'+str(i)+'.zarr', chunks='auto')[self.features_to_add_to_sample + self.auxiliary_features + ['mask']]
            for feature in self.features_to_add_to_sample :
                # reapply mask (to avoid issues with nans written in netcdf files)
                xr_dataset[feature] = xr_dataset[feature].where(xr_dataset.mask>0)
                # compute mean, median and std for each level (since temperature/salinity may change a lot with the depth)
                xr_dataset['mean_'+feature] = (xr_dataset[feature].mean(dim=['t', 'x_c', 'y_c']))
                xr_dataset['std_'+feature] = (xr_dataset[feature].std(dim=['t', 'x_c', 'y_c']))
                # fill nans with mean (doesn't the number to be fillted in matter since they will be masked, 
                # but they have to be filled with any numbers so that nans do not propagate everywhere) 
                xr_dataset[feature] = xr_dataset[feature].fillna(xr_dataset['mean_'+feature])
                # normalize data by shifting with mean value and dividing by std (mean and std are computed above for each vertical level)
                xr_dataset['normalized_'+feature] = ((xr_dataset[feature]-xr_dataset['mean_'+feature])/xr_dataset['std_'+feature]) 
            # save result in a list
            self.list_of_xr_datasets[i] = xr_dataset
            self.list_of_torch_datasets[i] = torchDataset(xr_dataset, self.features_to_add_to_sample, self.auxiliary_features, transform=ToTensor())
            
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
        
    def tensor_restore_units(tensor, sample, reference_feature) :
        return tensor*(sample['std_'+reference_feature][:,:,None,None])+sample['mean_'+reference_feature][:,:,None,None]

    def tensor_normalize(tensor, sample, reference_feature) :
        return (tensor-sample['mean_'+reference_feature][:,:,None,None])/(sample['std_'+reference_feature][:,:,None,None])
    
    
def central_diffs(dataArray) :
    if len(dataArray.shape) == 5 : #5d data cube
        batch_len, nb_of_levels, nb_of_channels, output_h, output_w = dataArray.shape
        flatten_data = dataArray.flatten(start_dim=0, end_dim=2)[:,None,:,:]
    if len(dataArray.shape) == 4 : # 1 channel
        batch_len, nb_of_levels, width, height = dataArray.shape
        flatten_data = dataArray.flatten(start_dim=0, end_dim=1)[:,None,:,:]
    weights = torch.zeros(2,1,3,3).to(dataArray.device) # 2 channels : 1 channel for x-difference, other for y-differences
    weights[0,0,:,:] = torch.tensor([[0,0.,0],[-0.5,0.,0.5],[0,0.,0]]) #dx
    weights[1,0,:,:] = torch.tensor([[0,-0.5,0],[0,0.,0],[0,0.5,0]])   #dy
    res = torch.nn.functional.conv2d(flatten_data.float(), weights, \
                               bias=None, stride=1, padding='same', dilation=1, groups=1)
    res_dx = res[:,0,1:-1,1:-1].unflatten(dim=0, sizes=(batch_len, nb_of_levels))
    res_dy = res[:,1,1:-1,1:-1].unflatten(dim=0, sizes=(batch_len, nb_of_levels))
    return res_dx, res_dy

def finite_diffs_sqr_2d_array(dataArray) :
    res_dx, res_dy = central_diffs(dataArray)
    res = torch.pow(res_dx,2) + torch.pow(res_dy,2)
    return res

def gradient_based_MSEloss(outputs, targets, reduction='mean') :
    outputs_grad = finite_diffs_sqr_2d_array(outputs)
    targets_grad = finite_diffs_sqr_2d_array(targets)

    value_loss = torch.nn.functional.mse_loss(outputs, targets, reduction=reduction)
    grad_loss = torch.nn.functional.mse_loss(outputs_grad, targets_grad, reduction=reduction)

    return (value_loss+0.1*grad_loss)     

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
                # if model output is a normalized feature then compute also the non-normalized feature for the diagnostics 
                not_normalized_feature = feature.replace("normalized_", "")
                self.list_of_features_to_predict.append(not_normalized_feature)
        
    def cut_bords(self, tensor, nb_of_border_pix) :
        if nb_of_border_pix is None :
            return tensor
        else :
            if (len(tensor.shape) == 5):
                return tensor[:,:, :, nb_of_border_pix:-nb_of_border_pix, nb_of_border_pix:-nb_of_border_pix] 
            if (len(tensor.shape) == 4):
                return tensor[:,:, nb_of_border_pix:-nb_of_border_pix, nb_of_border_pix:-nb_of_border_pix] 
            if (len(tensor.shape) == 3):
                return tensor[:, nb_of_border_pix:-nb_of_border_pix, nb_of_border_pix:-nb_of_border_pix] 
        
    def expand_to_bords(self, tensor, nb_of_border_pix) :
        if nb_of_border_pix is None :
            return tensor
        else :
            new_tensor = torch.empty((tensor.shape[0],tensor.shape[1], tensor.shape[2]+2*nb_of_border_pix, tensor.shape[3]+2*nb_of_border_pix)).\
            to(tensor.device)
            new_tensor[:,:, nb_of_border_pix:-nb_of_border_pix, nb_of_border_pix:-nb_of_border_pix] = tensor
            return new_tensor

    def transform_features(self, batch, features, cut_border_pix) :
        # transform dictionary issued from the data loader into 4D torch arrays of shape [N,C,H,W]
        stacked_channels = torch.stack([self.cut_bords(batch[key], cut_border_pix) for key in features])
        transform = torch.permute(stacked_channels, (1,2,0,3,4)).to(torch.float32)
        return transform
    
    def transform_mask(self, batch) : 
        nb_of_border_pix = self.torch_model.cut_border_pix_output
        if nb_of_border_pix is None :
            return batch['eroded_mask']
        else :
            return batch['eroded_mask'][:, nb_of_border_pix:-nb_of_border_pix, nb_of_border_pix:-nb_of_border_pix] 

    def evaluate_tensor_metrics_with_mask(self, metrics, mask, truth, model_output, reduction='mean') :
        if (len(model_output.shape) == 3) : # 1 feature (1 channel) and 1 level - 3D tensor
            batch_len, output_h, output_w = model_output.shape  
            valid_mask_counts = torch.count_nonzero(mask)
            mask = mask
        
        if (len(model_output.shape) == 4) : # 1 feature (1 channel) - 4D tensor
            batch_len, nb_of_levels, output_h, output_w = model_output.shape  
            valid_mask_counts = torch.count_nonzero(mask)*nb_of_levels
            mask = mask[:,None,:,:]
            
        if (len(model_output.shape) == 5) : # full 5D tensor
            batch_len, nb_of_levels, nb_of_channels, output_h, output_w = model_output.shape  
            valid_mask_counts = torch.count_nonzero(mask)*nb_of_levels*nb_of_channels
            mask = mask[:,None,None,:,:]
            
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

    def pressure_based_MSEloss(self, pred_sigma, target_sigma, rho_ct_ct, idx_level, mask, dx, dy) :
        true_pres_grad_x, true_pres_grad_y, true_pres_grad_norm = get_pressure_grad(target_sigma, rho_ct_ct, dx, dy)
        pred_pres_grad_x, pred_pres_grad_y, pred_pres_grad_norm = get_pressure_grad(pred_sigma, rho_ct_ct, dx, dy)

        grad_x_loss = self.evaluate_tensor_metrics_with_mask(torch.nn.functional.mse_loss, mask[:,1:-1,1:-1], \
                                                            pred_pres_grad_x[:,idx_level,:,:], true_pres_grad_x[:,idx_level,:,:], reduction='mean')
        grad_y_loss = self.evaluate_tensor_metrics_with_mask(torch.nn.functional.mse_loss, mask[:,1:-1,1:-1], \
                                                            pred_pres_grad_y[:,idx_level,:,:], true_pres_grad_y[:,idx_level,:,:], reduction='mean')
        grad_loss = grad_x_loss+grad_y_loss
        return grad_loss
        
    def training_step(self, batch, batch_idx) :
        x = self.transform_features(batch, self.input_features, self.torch_model.cut_border_pix_input)
        y_true = self.transform_features(batch, self.output_features, self.torch_model.cut_border_pix_output)
        mask = self.transform_mask(batch)
        
        if (self.output_units is None) :
            y_model = self.torch_model(x)
        else :
            y_units = self.transform_features(batch, self.output_units, self.torch_model.cut_border_pix_output)
            y_model = y_units*self.torch_model(x)
            
        if (self.loss=='pressure_based_MSEloss') :
            rho_ct_ct = self.cut_bords(batch['rho_ct_ct'], self.torch_model.cut_border_pix_output)
            dx = self.cut_bords(batch['e1t'], self.torch_model.cut_border_pix_output)
            dy = self.cut_bords(batch['e2t'], self.torch_model.cut_border_pix_output)
            pred_sigma = y_model[:, :, 0, :, :]
            target_sigma = y_true[:, :, 0, :, :]
            idx_level=100
            loss_pres = self.pressure_based_MSEloss(pred_sigma, target_sigma, rho_ct_ct, idx_level, mask, dx, dy)
            loss_val = self.evaluate_tensor_metrics_with_mask(torch.nn.functional.mse_loss, mask, y_model, y_true)
            loss_total = 1e6*loss_pres+loss_val
        else :
            loss_total = self.evaluate_tensor_metrics_with_mask(self.loss, mask, y_model, y_true)  
        self.log_dict({'loss_train' : loss_total}, on_step=False, on_epoch=True)
        return loss_total

    # validation logics (is evaluated during the training, but the data is not used to the optimization loop)
    def validation_step(self, batch, batch_idx) :
        x = self.transform_features(batch, self.input_features, self.torch_model.cut_border_pix_input)
        y_true = self.transform_features(batch, self.output_features, self.torch_model.cut_border_pix_output)
        mask = self.transform_mask(batch)

        if (self.output_units is None) :
            y_model = self.torch_model(x)
        else :
            y_units = self.transform_features(batch, self.output_units, self.torch_model.cut_border_pix_output)
            y_model = y_units*self.torch_model(x)
            
        first_layer_weights = list(self.torch_model.__dict__['_modules'].values())[0].weight
        
        if (self.loss=='pressure_based_MSEloss') :
            rho_ct_ct = self.cut_bords(batch['rho_ct_ct'], self.torch_model.cut_border_pix_output)
            dx = self.cut_bords(batch['e1t'], self.torch_model.cut_border_pix_output)
            dy = self.cut_bords(batch['e2t'], self.torch_model.cut_border_pix_output)
            pred_sigma = y_model[:, :, 0, :, :]
            target_sigma = y_true[:, :, 0, :, :]
            idx_level=100
            loss_pres = self.pressure_based_MSEloss(pred_sigma, target_sigma, rho_ct_ct, idx_level, mask, dx, dy)
            loss_val = self.evaluate_tensor_metrics_with_mask(torch.nn.functional.mse_loss, mask, y_model, y_true)
            loss_total = 1e6*loss_pres+loss_val
            self.log_dict({'loss_total' : loss_total,
                           'loss_pressure' : loss_pres,
                           'loss_val' : loss_val,
                      'first_weight' : np.array(first_layer_weights.cpu()).flat[0]}, on_step=False, on_epoch=True)
        else :
            loss_val = self.evaluate_tensor_metrics_with_mask(self.loss, mask, y_model, y_true)  
            self.log_dict({'loss_val' : loss_val,
                      'first_weight' : np.array(first_layer_weights.cpu()).flat[0]}, on_step=False, on_epoch=True) 
    
    # gives model output in a form of a dictionary of batches of 2d fields
    def predict_step(self, batch, batch_idx, dataloader_idx) :
        x = self.transform_features(batch, self.input_features, self.torch_model.cut_border_pix_input)
        
        output_5d_tensor = self.torch_model(x)
        batch_len, nb_of_levels, nb_of_channels, output_h, output_w = output_5d_tensor.shape

        if not(self.output_units is None) : 
            y_units = self.transform_features(batch, self.output_units, self.torch_model.cut_border_pix_output)
            output_5d_tensor_units = output_5d_tensor*y_units
            
        # construct the dictionary of the predicted features by decomposing the channels in the 4d torch tensor
        pred = dict()
        for i, feature in enumerate(self.output_features) :
            if (self.output_units is None) :
                pred[feature] = output_5d_tensor[:, :, i, :, :]
            else :
                pred[feature+'_dimless'] = output_5d_tensor[:, :, i, :, :]
                pred[feature] = output_5d_tensor_units[:, :, i, :, :]
            # if some outputs are normalized then compute also result in the restored units (not normalized)
            if feature.startswith('normalized_') :
                not_normalized_feature = feature.replace("normalized_", "")
                pred[not_normalized_feature] = PyLiDataModule.tensor_restore_units(pred[feature], batch, not_normalized_feature)
                
        for i, feature in enumerate(self.list_of_features_to_predict) :
            pred['mask'] = batch['eroded_mask'][:,None,:,:]
            pred[feature+'_masked'] = self.expand_to_bords(pred[feature], self.torch_model.cut_border_pix_output)
            pred[feature+'_masked'] = pred[feature+'_masked'].where(pred['mask'], torch.ones_like(pred[feature+'_masked'])*np.nan)
        return pred 
    
    # testing logic - to evaluate the model after training
    def test_step(self, batch, batch_idx, dataloader_idx) :
        pred = self.predict_step(batch, batch_idx, dataloader_idx)
        mask = self.transform_mask(batch)
        
        test_dict = dict({'loss_val' : dict(), 'loss_grad' : dict(), 'corr_coef' : dict(), 'corr_coef_grad' : dict()})
        dict_for_log = dict()

        for i, feature in enumerate(self.list_of_features_to_predict) :
            truth = self.cut_bords(batch[feature], self.torch_model.cut_border_pix_output)
            model_output = pred[feature]
            
            test_dict['loss_val'][feature] = self.evaluate_tensor_metrics_with_mask(torch.nn.functional.mse_loss, mask, model_output, truth, reduction='mean')
            test_dict['corr_coef'][feature] = torch.corrcoef(torch.vstack((torch.flatten(model_output).view(1,-1), \
                                                              torch.flatten(truth).view(1,-1))))[1,0]
            # metrics on horizontal gradients
            model_output_grad = finite_diffs_sqr_2d_array(model_output)
            truth_grad = finite_diffs_sqr_2d_array(truth)
            test_dict['loss_grad'][feature] = self.evaluate_tensor_metrics_with_mask(torch.nn.functional.mse_loss, mask[:,1:-1,1:-1], \
                                                                         model_output_grad, \
                                                                         truth_grad, reduction='mean')
            test_dict['corr_coef_grad'][feature] = torch.corrcoef(torch.vstack((torch.flatten(model_output_grad).view(1,-1), \
                                                              torch.flatten(truth_grad).view(1,-1))))[1,0]

        # pressure at 100th level
        idx_level = 100
        true_votemper_var = self.cut_bords(batch['votemper_var'], self.torch_model.cut_border_pix_output)
        rho_ct_ct = self.cut_bords(batch['rho_ct_ct'], self.torch_model.cut_border_pix_output)
        dx = self.cut_bords(batch['e1t'], self.torch_model.cut_border_pix_output)
        dy = self.cut_bords(batch['e2t'], self.torch_model.cut_border_pix_output)
        model_votemper_var = pred['votemper_var']
        true_pres_grad_x, true_pres_grad_y, true_pres_grad_norm = get_pressure_grad(true_votemper_var, rho_ct_ct, dx, dy)
        pred_pres_grad_x, pred_pres_grad_y, pred_pres_grad_norm = get_pressure_grad(model_votemper_var, rho_ct_ct, dx, dy)
        test_dict['loss_val']['pressure_grad_x'] = self.evaluate_tensor_metrics_with_mask(torch.nn.functional.mse_loss, mask[:,1:-1,1:-1], \
                                                                         pred_pres_grad_x[:,idx_level,:,:], true_pres_grad_x[:,idx_level,:,:], reduction='mean')
        test_dict['loss_val']['pressure_grad_y'] = self.evaluate_tensor_metrics_with_mask(torch.nn.functional.mse_loss, mask[:,1:-1,1:-1], \
                                                                         pred_pres_grad_y[:,idx_level,:,:], true_pres_grad_y[:,idx_level,:,:], reduction='mean')
        test_dict['loss_val']['pressure_grad'] = test_dict['loss_val']['pressure_grad_x'] + test_dict['loss_val']['pressure_grad_y']
        
        for metrics in list(test_dict.keys()) : 
            for feature in list(test_dict[metrics].keys()) : 
                dict_for_log.update({(metrics+'_'+feature) : test_dict[metrics][feature]})
        self.log_dict(dict_for_log)

    def configure_optimizers(self) :
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    
