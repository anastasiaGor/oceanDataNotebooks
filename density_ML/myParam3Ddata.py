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

    def __init__(self, xarray_dataset, features_to_add_to_sample, transform=None):
        self.transform = transform
        self.features_to_add_to_sample = features_to_add_to_sample
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
    def __init__(self, cloud_data_sets, features_to_add_to_sample, batch_size) :
        super().__init__()
        self.cloud_data_sets = cloud_data_sets
        self.features_to_add_to_sample = features_to_add_to_sample
        self.batch_size = batch_size
        self.list_of_xr_datasets = [xr.Dataset() for i in range(len(self.cloud_data_sets))]
        self.list_of_torch_datasets = [{} for i in range(len(self.cloud_data_sets))]
        
    #def prepare_data(self) :
        # preparation of data: mean and std of the dataset (to avoid batch avg), normalization and nan filling
        for i in range(len(self.cloud_data_sets)) :
            # read file
            PERSISTENT_BUCKET = os.environ['PERSISTENT_BUCKET'] 
            xr_dataset = xr.open_zarr(f'{PERSISTENT_BUCKET}/data3D_'+str(i)+'.zarr', chunks='auto')[self.features_to_add_to_sample + ['mask', 'z_l', 'f']]
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
            self.list_of_torch_datasets[i] = torchDataset(xr_dataset, self.features_to_add_to_sample, transform=ToTensor())
            
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
    
    
def central_diffs(dataArray, dim) :
    res = 0.5*(torch.roll(dataArray, shifts=-1, dims=dim) - torch.roll(dataArray, shifts=1, dims=dim))
    return res

def finite_diffs_sqr_2d_array(dataArray, dim_x, dim_y) :
    res = torch.pow(central_diffs(dataArray, dim_x),2) + torch.pow(central_diffs(dataArray, dim_y),2)
    # cut the borders of the image
    res = torch.narrow(res, dim=dim_x, start=1, length=res.shape[dim_x]-2)
    res = torch.narrow(res, dim=dim_y, start=1, length=res.shape[dim_y]-2)
    return res

def gradient_based_MSEloss(outputs, targets, reduction='mean') :
        if (len(outputs.shape) == 4) : 
            dim_x = 3
            dim_y = 2
        if (len(outputs.shape) == 5) : 
            dim_x = 4
            dim_y = 3
        outputs_grad = finite_diffs_sqr_2d_array(outputs, dim_x=dim_x, dim_y=dim_y)
        targets_grad = finite_diffs_sqr_2d_array(targets, dim_x=dim_x, dim_y=dim_y)
        
        value_loss = torch.nn.functional.mse_loss(outputs, targets, reduction=reduction)
        grad_loss = torch.nn.functional.mse_loss(outputs_grad, targets_grad, reduction=reduction)

        return (value_loss+0.1*grad_loss)    
    
class GenericPyLiModule(pl.LightningModule):
    def __init__(self, torch_model, input_features, output_features, loss, optimizer, learning_rate):
        super().__init__()
        self.torch_model = torch_model
        self.input_features = input_features
        self.output_features = output_features
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
            return tensor[:,:, nb_of_border_pix:-nb_of_border_pix, nb_of_border_pix:-nb_of_border_pix] 
        
    def expand_to_bords(self, tensor, nb_of_border_pix) :
        if nb_of_border_pix is None :
            return tensor
        else :
            new_tensor = torch.empty((tensor.shape[0],tensor.shape[1], tensor.shape[2]+2*nb_of_border_pix, tensor.shape[3]+2*nb_of_border_pix)).\
            to(tensor.device)
            new_tensor[:,:, nb_of_border_pix:-nb_of_border_pix, nb_of_border_pix:-nb_of_border_pix] = tensor
            return new_tensor

    def transform_input(self, batch) :
        # transform dictionary issued from the data loader into 5D torch arrays of shape [N,L,C,H,W]
        stacked_channels = torch.stack([self.cut_bords(batch[key], self.torch_model.cut_border_pix_input) for key in self.input_features])
        transform = torch.permute(stacked_channels, (1,2,0,3,4)).to(torch.float32)
        return transform
    
    def transform_target(self, batch) :
        stacked_channels = torch.stack([self.cut_bords(batch[key], self.torch_model.cut_border_pix_output) for key in self.output_features])
        transform = torch.permute(stacked_channels, (1,2,0,3,4)).to(torch.float32)
        return transform
    
    def transform_mask(self, batch) : 
        nb_of_border_pix = self.torch_model.cut_border_pix_output
        if nb_of_border_pix is None :
            return batch['eroded_mask']
        else :
            return batch['eroded_mask'][:, nb_of_border_pix:-nb_of_border_pix, nb_of_border_pix:-nb_of_border_pix] 

    def evaluate_tensor_metrics_with_mask(self, metrics, mask, truth, model_output, reduction='mean') :
        if (len(model_output.shape) == 4) : # 1 feature (1 channel) - 4D tensor
            batch_len, nb_of_levels, output_h, output_w = model_output.shape  
            valid_mask_counts = torch.count_nonzero(mask)*nb_of_levels
            mask = mask[:,None,:,:]
            
        if (len(model_output.shape) == 5) : # 5D tensor
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
        
    def training_step(self, batch, batch_idx) :
        x = self.transform_input(batch)
        y_true = self.transform_target(batch)
        mask = self.transform_mask(batch)
        
        y_model = self.torch_model(x)
        loss_val = self.evaluate_tensor_metrics_with_mask(self.loss, mask, y_model, y_true)  
        self.log_dict({'loss_train' : loss_val}, on_step=False, on_epoch=True)
        return loss_val

    # validation logics (is evaluated during the training, but the data is not used to the optimization loop)
    def validation_step(self, batch, batch_idx) :
        x = self.transform_input(batch)
        y_true = self.transform_target(batch)
        mask = self.transform_mask(batch)
        y_model = self.torch_model(x)
        
        loss_val = self.evaluate_tensor_metrics_with_mask(self.loss, mask, y_model, y_true)  
        first_layer_weights = list(self.torch_model.__dict__['_modules'].values())[0].weight
        self.log_dict({'loss_val' : loss_val,
                      'first_weight' : np.array(first_layer_weights.cpu()).flat[0]}, on_step=False, on_epoch=True) 
    
    # gives model output in a form of a dictionary of batches of 2d fields
    def predict_step(self, batch, batch_idx, dataloader_idx) :
        x = self.transform_input(batch)
        
        output_5d_tensor = self.torch_model(x)
        batch_len, nb_of_levels, nb_of_channels, output_h, output_w = output_5d_tensor.shape
            
        # construct the dictionary of the predicted features by decomposing the channels in the 4d torch tensor
        pred = dict()
        for i, feature in enumerate(self.output_features) :
            pred[feature] = output_5d_tensor[:, :, i, :, :]
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
            model_output_grad = finite_diffs_sqr_2d_array(model_output, dim_x=3, dim_y=2)
            truth_grad = finite_diffs_sqr_2d_array(truth, dim_x=3, dim_y=2)
            test_dict['loss_grad'][feature] = self.evaluate_tensor_metrics_with_mask(torch.nn.functional.mse_loss, mask[:,1:-1,1:-1], \
                                                                         model_output_grad, \
                                                                         truth_grad, reduction='mean')
            test_dict['corr_coef_grad'][feature] = torch.corrcoef(torch.vstack((torch.flatten(model_output_grad).view(1,-1), \
                                                              torch.flatten(truth_grad).view(1,-1))))[1,0]
            for metrics in list(test_dict.keys()) : 
                dict_for_log.update({(metrics+'_'+feature) : test_dict[metrics][feature]})
        self.log_dict(dict_for_log)

    def configure_optimizers(self) :
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    
