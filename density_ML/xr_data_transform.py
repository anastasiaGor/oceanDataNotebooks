import glob, os
import numpy as np
import xarray as xr
import dask

import gsw
import xesmf as xe
import xgcm
import sys
import s3fs

metrics = {
    ('X',): ['e1t'], # X distances
    ('Y',): ['e2t'], # Y distances
    ('X', 'Y',): ['area_t'] # Areas 
}

eORCA025_global_mesh = xr.open_dataset('../global_eORCA025_mesh.nc').load()

def coarsen(XdataSet) :
    # prepare input dataset for coarseinig
    input_ds = XdataSet
    # generate corner lat lon
    input_ds = xgcm.autogenerate.generate_grid_ds(input_ds, axes_dims_dict={'X' : 'x', 'Y' : 'y'}, position=('center', 'right'))
    input_ds = xgcm.autogenerate.generate_grid_ds(input_ds, axes_dims_dict={'X' : 'x', 'Y' : 'y'}, axes_coords_dict={'X' : 'lon', 'Y' : 'lat'}, position=('center', 'outer'))
    # adjust names of coords for regridder
    input_ds = input_ds.rename({'x_outer': 'x_b','y_outer': 'y_b', 'x_right' : 'x_r', 'y_right' : 'y_r'})
    input_ds.coords['mask'] = input_ds['tmask']
    input_ds.coords['area_t'] = input_ds['e1t']*input_ds['e2t']
    input_ds.coords['area_u'] = input_ds['e1u']*input_ds['e2u']
    input_ds.coords['area_v'] = input_ds['e1v']*input_ds['e2v']
    input_ds.coords['area_f'] = input_ds['e1f']*input_ds['e2f']
    xgcm_grid_HR = xgcm.Grid(input_ds, metrics=metrics, periodic=False, coords={'X' : {'center' : 'x', 'outer' : 'x_b', 'right' : 'x_r'}, \
                                           'Y' : {'center' : 'y', 'outer' : 'y_b', 'right' : 'y_r'}})
    input_ds.coords['lon_b'] = xgcm_grid_HR.interp(da=input_ds['lon_outer'], axis='Y', to='outer',boundary='extend')
    input_ds.coords['lat_b'] = xgcm_grid_HR.interp(da=input_ds['lat_outer'], axis='X', to='outer',boundary='extend')
    
    # prepare output grid
    grid_LR = eORCA025_global_mesh # - quarter degree global grid
    # to take a part of the global grid corresponding to the data cut
    selection = grid_LR.lon.where((grid_LR.lon>=input_ds.lon.min().values) & (grid_LR.lon<=input_ds.lon.max().values) &\
                                  (grid_LR.lat>=input_ds.lat.min().values) & (grid_LR.lat<=input_ds.lat.max().values), drop=True)
    selected_x = list(selection.x.values)
    selected_y = list(selection.y.values)
    #selected_x_b = [ (element - 0.5) for element in range(selected_x[0], selected_x[-1]+2)]
    #selected_y_b = [ (element - 0.5) for element in range(selected_y[0], selected_y[-1]+2)]
    grid_LR = grid_LR.isel(x=selected_x[0:-1], x_b=selected_x, y=selected_y[0:-1], y_b=selected_y)
    # reinit x and y arrays
    grid_LR.coords['x'] = np.arange(1,len(selected_x))
    grid_LR.coords['y'] = np.arange(1,len(selected_y))
    grid_LR.coords['x_b'] = np.arange(1,len(selected_x)+1)-0.5
    grid_LR.coords['y_b'] = np.arange(1,len(selected_y)+1)-0.5
    # create regridder, i.e. compute weights
    # TODO : save weights and reuse them (can accelerate pre-treatment)
    regridder_with_mask = xe.Regridder(input_ds, grid_LR, method="conservative_normed")
    # perform regridding
    coarsened_ds = regridder_with_mask(input_ds).drop(['x_b', 'y_b', 'x_r', 'y_r', 'lon_b', 'lat_b', 'tmask'])
    coarsened_ds.coords['mask'] = grid_LR['mask']
    coarsened_ds.coords['e1t'] = grid_LR['e1t']
    coarsened_ds.coords['e2t'] = grid_LR['e2t']
    coarsened_ds.coords['area_t'] = coarsened_ds.coords['e1t']*coarsened_ds.coords['e2t']
    coarsened_ds = xgcm.autogenerate.generate_grid_ds(coarsened_ds, axes_dims_dict={'X' : 'x', 'Y' : 'y'}, position=('center', 'right'))
    return coarsened_ds

# only xarray data pretreatment here: slicing, filtering, computation of some variables
def xr_data_prep_pipeline(in_xr_dataset, data_dict_entity) :
    # drop not used variables
    HR_data = in_xr_dataset.drop(['fmask','qt_oce', 'somecrty','sometauy','sossheig','sowaflup','sozocrtx','sozotaux','umask','vmask'],errors='ignore')
    # time and space slicing
    data_slice = dict(data_dict_entity['time_slice'] | data_dict_entity['xy_slice'])
    HR_data=HR_data.isel(data_slice)
    
    # compute "true" density (before filtering
    # density anomaly (sigma0). denoted as 'true' since is computed on the basis of original (non-filtered) HR data
    HR_data['sigma0_true'] = xr.apply_ufunc(gsw.density.sigma0, HR_data['sosaline'], HR_data['sosstsst'], dask='parallelized')
    # sqr of conservative temperature (to compute the subgrid variance further)
    HR_data['sst_sqr'] = HR_data['sosstsst']**2
    HR_data['sal_sqr'] = HR_data['sosaline']**2
    
    # apply filtering/coarsening
    LR_data = coarsen(HR_data)
    
    # variables to compute after filtering:
    LR_data = LR_data.assign(var_T = LR_data['sst_sqr'] - LR_data['sosstsst']**2)
    LR_data = LR_data.assign(var_S = LR_data['sal_sqr'] - LR_data['sosaline']**2)
    LR_data = LR_data.assign(sigma0_model = xr.apply_ufunc(gsw.density.sigma0, LR_data['sosaline'], LR_data['sosstsst'], dask='parallelized'))
    LR_data = LR_data.assign(sigma0_delta = LR_data['sigma0_true'] - LR_data['sigma0_model'])
    
    # density derivative (EOS)
    # a weird round-about because of issues in implementation of gsw python (does not support dask, needs numpy input/output)
    LR_data = LR_data.assign(pressure = xr.zeros_like(LR_data.sosstsst))
    temp_array = xr.apply_ufunc(gsw.rho_second_derivatives, LR_data.sosaline, LR_data.sosstsst, LR_data.pressure, \
                            input_core_dims = [('time_counter', 'y', 'x') for i in range(3)],\
                           output_core_dims = [('time_counter', 'y', 'x') for i in range(5)],\
                           dask='parallelized', dask_gufunc_kwargs=dict({'allow_rechunk' : True})) 
    LR_data = LR_data.assign(rho_sa_sa = xr.DataArray(temp_array[0], dims=['time_counter', 'y', 'x']))
    LR_data = LR_data.assign(rho_sa_ct = xr.DataArray(temp_array[1], dims=['time_counter', 'y', 'x']))
    LR_data = LR_data.assign(rho_ct_ct = xr.DataArray(temp_array[2], dims=['time_counter', 'y', 'x']))
    LR_data = LR_data.assign(rho_sa_p = xr.DataArray(temp_array[3], dims=['time_counter', 'y', 'x']))
    LR_data = LR_data.assign(rho_ct_p = xr.DataArray(temp_array[4], dims=['time_counter', 'y', 'x']))
    del temp_array
    
    # spatial finite differences
    LR_data = xgcm.autogenerate.generate_grid_ds(LR_data, axes_dims_dict={'X' : 'x', 'Y' : 'y'}, position=('center', 'right'))
    xgcm_grid_LR = xgcm.Grid(LR_data, metrics=metrics, periodic=False, coords={'X' : {'center' : 'x', 'right' : 'x_right'}, \
                                                                                  'Y' : {'center' : 'y', 'right' : 'y_right'}})
    for var in ['sosstsst', 'sosaline', 'sigma0_delta'] :
        LR_data['diff_'+var+'_sqr'] = xgcm_grid_LR.interp(xgcm_grid_LR.diff(LR_data[var], 'X', boundary='extend')**2, 'X') + \
                                   xgcm_grid_LR.interp(xgcm_grid_LR.diff(LR_data[var], 'Y', boundary='extend')**2, 'Y')
    
    # predictors for the parametrization
    LR_data = LR_data.assign(predictor_T = 0.5*LR_data['rho_ct_ct']*LR_data['diff_sosstsst_sqr'])
    LR_data = LR_data.assign(predictor_S = 0.5*LR_data['rho_sa_sa']*LR_data['diff_sosaline_sqr'])
    LR_data = LR_data.assign(predictor_TS = 0.5*LR_data['rho_sa_ct']*np.sqrt(LR_data['diff_sosaline_sqr']*LR_data['diff_sosstsst_sqr']))

    # again drop non-used vars and coords
    LR_data = LR_data.drop(['time_centered', 'tmask', 'nav_lat', 'nav_lon', 'depth', 'e1f', 'e2f', 'e1u', 'e2u', \
                                      'e1v', 'e2v', 'e1t', 'e2t', 'sigma0_true', 'sst_sqr', 'sal_sqr'], errors='ignore')
    return LR_data

def read_and_perform_transform(dask_server, list_data_dict) :
    client = dask.distributed.Client(dask_server)
    sys.path.insert(1, '..')
    from validate_catalog import all_params
    params_dict, cat = all_params()
    SCRATCH_BUCKET = os.environ['SCRATCH_BUCKET'] 

    for i, dictionary in enumerate(list_data_dict) :
        original_xr_ds = cat.eNATL60(region=dictionary['region'],datatype='surface_hourly', season=dictionary['season']).to_dask()
        transformed_xr_ds = xr_data_prep_pipeline(original_xr_ds, dictionary)
        
        # write on scratch (can be also tmp)
        # netcdf format
        transformed_xr_ds.to_netcdf('/tmp/dataset'+str(i)+'.nc')
        ## or zarr
        #transformed_xr_ds.to_zarr(f'{SCRATCH_BUCKET}/dataset'+str(i)+'.zarr') 