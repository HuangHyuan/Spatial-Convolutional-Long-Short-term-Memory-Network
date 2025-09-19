import numpy as np
from sklearn.preprocessing import StandardScaler

def split_data(data, pad_len):
    """
    Split data into non-overlapping patches of size (pad_len, pad_len), with edge padding by duplication.

    Args:
        data: Input array of shape (dim, time, level, height, width)
        pad_len: Size of each patch (assumed square)

    Returns:
        Reshaped array of shape (num_patches, dim, level, pad_len, pad_len)
    """
    dim, time, level, num_rows, num_cols = data.shape
    num_rows_new = num_rows // pad_len
    num_cols_new = num_cols // pad_len
    remainder_rows = num_rows % pad_len
    remainder_cols = num_cols % pad_len

    # Reshape the divisible part into patches
    data_reshaped = data[:,:,:,:num_rows_new*pad_len, :num_cols_new*pad_len].reshape(dim, time, level, num_rows_new, pad_len, num_cols_new, pad_len).transpose(1, 3, 5, 0, 2, 4, 6).reshape(-1,dim,level,pad_len,pad_len)

    # Handle remainder parts by duplicating edge regions
    if remainder_rows != 0 or remainder_cols != 0:
        last_row_piece = data[:,:,:,-pad_len:, :num_cols_new*pad_len].reshape(dim, time, level,1,pad_len,-1,pad_len).transpose(1, 3, 5, 0, 2, 4, 6).reshape(-1,dim,level,pad_len,pad_len)
        last_col_piece = data[:,:,:,:num_rows_new*pad_len, -pad_len:].reshape(dim, time, level,-1,pad_len,1,pad_len).transpose(1, 3, 5, 0, 2, 4, 6).reshape(-1,dim,level,pad_len,pad_len)
        last_corner_piece = data[:,:,:,-pad_len:, -pad_len:].reshape(dim, time, level,1,pad_len,1,pad_len).transpose(1, 3, 5, 0, 2, 4, 6).reshape(-1,dim,level,pad_len,pad_len)
        if remainder_rows != 0:
            data_reshaped = np.concatenate((data_reshaped, last_row_piece), axis=0)
        if remainder_cols != 0:
            data_reshaped = np.concatenate((data_reshaped, last_col_piece), axis=0)
        if  remainder_rows != 0 and remainder_cols != 0:
            data_reshaped = np.concatenate((data_reshaped, last_corner_piece), axis=0)

    return data_reshaped

def split_data_random(data, pad_len, num_samples=None, overlap_ratio=0.5):
    """
    Randomly crop patches from data to increase sample diversity.

    Args:
        data: Input data of shape (dim, time, level, height, width)
        pad_len: Size of each patch
        num_samples: Number of patches to generate. If None, estimated based on stride.
        overlap_ratio: Controls stride as pad_len * (1 - overlap_ratio)

    Returns:
        Array of shape (num_samples, dim, level, pad_len, pad_len)
    """
    dim, time, level, height, width = data.shape
    
   
    max_row_start = height - pad_len
    max_col_start = width - pad_len
    
    # If image is smaller than patch size, fall back to regular split
    if max_row_start <= 0 or max_col_start <= 0:
        return split_data(data, pad_len)
    
    # Estimate number of samples if not provided
    if num_samples is None:
        stride = int(pad_len * (1 - overlap_ratio))
        stride = max(1, stride) 
        num_samples = ((height - pad_len) // stride + 1) * ((width - pad_len) // stride + 1) * time
    

    samples = []
    for _ in range(num_samples):
        t = np.random.randint(0, time)
        row_start = np.random.randint(0, max_row_start + 1)
        col_start = np.random.randint(0, max_col_start + 1)
        patch = data[:, t:t+1, :, row_start:row_start+pad_len, col_start:col_start+pad_len]
        samples.append(patch)
    
    result = np.concatenate(samples, axis=0)
    result = result.reshape(-1, dim, level, pad_len, pad_len)
    
    return result

def generate_dataset(x, y, longitude, latitude, pad_len=64, Scale=True):
    """
    Generate dataset with standard patching and normalization.

    Args:
        x: Input features
        y: Target labels
        longitude: Longitude coordinates
        latitude: Latitude coordinates
        pad_len: Patch size for spatial splitting
        Scale: Whether to apply standardization

    Returns:
        Processed datasets and scalers (if scaling is applied)
    """
    x_shape = x.shape
    if len(x_shape) == 5 :
        level_num=[1000,  925,  850,  700,  600,  500,  400,  300,  250,  200,  150,  100, 50]
        dim, ntime, level, lat, lon =x_shape[0], x_shape[1],x_shape[2], x_shape[3], x_shape[4]
    else:
        dim, ntime, lat, lon =x_shape[0], x_shape[1],x_shape[2], x_shape[3]
    if dim == 5:  # Upper-air dataset
        # Create level, longitude, and latitude value arrays
        level_values = np.ones((1,ntime,level,lat, lon), dtype='float32')
        for step,lev in enumerate(level_num):
            level_values[0,:,step,:,:] = level_values[0,:,step,:,:]*lev
        lon_values= np.concatenate([np.concatenate([np.concatenate([longitude.reshape(1,lon)]*lat, axis=0).reshape(-1,lat,lon)]*level,axis=0).reshape(-1,level,lat,lon)]*ntime,axis=0).reshape(-1,ntime,level,lat,lon)
        lat_values = np.concatenate([np.concatenate([np.concatenate([np.sort(latitude)[::-1].reshape(lat,1)]*lon, axis=1).reshape(-1,lat,lon)]*level,axis=0).reshape(-1,level,lat,lon)]*ntime,axis=0).reshape(-1,ntime,level,lat,lon)

        x_input=x
        y_input=y

        # Concatenate level, lon, and lat as additional channels
        x_input = np.concatenate([x_input, level_values], axis=0)
        x_input = np.concatenate([x_input, lon_values], axis=0)
        x_input = np.concatenate([x_input, lat_values], axis=0)

       # Normalize each level independently
        x_shape,y_shape=x_input.shape,y_input.shape
        
        x_input = np.transpose(x_input, (0, 2, 1, 3, 4))
        y_input = np.transpose(y_input, (0, 2, 1, 3, 4))

        flattened_x = x_input.reshape((x_shape[0] , x_shape[2] , x_shape[1]*x_shape[3]*x_shape[4]))
        flattened_y = y_input.reshape((y_shape[0] , y_shape[2] , y_shape[1]*y_shape[3]*y_shape[4]))
        scaler_X=[]
        scaler_y=[]
        normalized_x=np.zeros((x_shape[0] , x_shape[2] , x_shape[1]*x_shape[3]*x_shape[4])).astype('float32')
        normalized_y=np.zeros((y_shape[0] , y_shape[2] , y_shape[1]*y_shape[3]*y_shape[4])).astype('float32')
        for i in range(x_shape[2]):# Normalize each level independently
            if Scale == True:
                scaler_X_i = StandardScaler()
                scaler_y_i = StandardScaler()
                # Fit y scaler on a subset of x data (first 5 features)
                normalized_x[:,i,:]=scaler_X_i.fit_transform(flattened_x[:,i,:].T).T
                normalized_y_fit = scaler_y_i.fit_transform(flattened_x[:5,i,:].T).T
                normalized_y[:,i,:] = scaler_y_i.transform(flattened_y[:,i,:].T).T
                scaler_X.append(scaler_X_i)
                scaler_y.append(scaler_y_i)
            else:
                scaler_X_i = Scale[0][i]
                scaler_y_i = Scale[1][i]
                normalized_x[:,i,:]=scaler_X_i.transform(flattened_x[:,i,:].T).T
                normalized_y[:,i,:] = scaler_y_i.transform(flattened_y[:,i,:].T).T                
        #  Reshape back to original structure
        x_input = normalized_x.reshape((x_shape[0] , x_shape[2] , x_shape[1],x_shape[3],x_shape[4]))
        y_input = normalized_y.reshape((y_shape[0] , y_shape[2] , y_shape[1],y_shape[3],y_shape[4]))
        x_input = np.transpose(x_input,(0,2,1,3,4))
        y_input = np.transpose(y_input,(0,2,1,3,4))   
    else:  # Surface dataset
        land = np.load('./constant/terrain.npy')
        plant = np.load('./constant/plant.npy')
        land_values= np.concatenate([land.reshape(-1,lat,lon)]*ntime,axis=0).reshape(-1,ntime,lat,lon)
        plant_values = np.concatenate([plant.reshape(-1,lat,lon)]*ntime,axis=0).reshape(-1,ntime,lat,lon)
        x_input=x
        y_input=y

        x_input = np.concatenate([x_input, land_values], axis=0)
        x_input = np.concatenate([x_input, plant_values], axis=0)
        
        x_shape,y_shape=x_input.shape,y_input.shape

        flattened_x = x_input.reshape((x_shape[0] , x_shape[1] * x_shape[2]*x_shape[3])).astype('float32')
        flattened_y = y_input.reshape((y_shape[0] , y_shape[1] * y_shape[2]*y_shape[3])).astype('float32')
        if Scale == True:

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            normalized_x = scaler_X.fit_transform(flattened_x.T).T
            normalized_y_fit = scaler_y.fit_transform(flattened_x[:4,:].T).T
            normalized_y = scaler_y.transform(flattened_y.T).T
        else:
            normalized_x = Scale[0].transform(flattened_x.T).T
            normalized_y = Scale[1].transform(flattened_y.T).T

        x_input = normalized_x.reshape((x_shape[0] , x_shape[1] , 1, x_shape[2] , x_shape[3]))
        y_input = normalized_y.reshape((y_shape[0] , y_shape[1] , 1, y_shape[2] , y_shape[3]))
    # Apply standard patching
    x_dataset = split_data(x_input, pad_len)
    y_dataset = split_data(y_input, pad_len)
    
    if  Scale == True:
        return x_dataset,y_dataset,scaler_X,scaler_y
    else:
        return x_dataset,y_dataset

def generate_dataset_random(x, y, longitude, latitude, pad_len=64, Scale=True, num_samples=None, overlap_ratio=0.5):
    """
    Generate dataset using random patching to increase data diversity.

    Args:
        x, y: Input features and labels
        longitude, latitude: Coordinate arrays
        pad_len: Patch size
        Scale: Whether to apply standardization
        num_samples: Number of random patches to generate
        overlap_ratio: Controls patch density via stride

    Returns:
        Dataset with random patches and scalers (if Scale=True)
    """
    x_shape = x.shape
    if len(x_shape) == 5:
        level_num = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
        dim, ntime, level, lat, lon = x_shape[0], x_shape[1], x_shape[2], x_shape[3], x_shape[4]
    else:
        dim, ntime, lat, lon = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
    
    if dim == 5:  # Upper-air data
        level_values = np.ones((1, ntime, level, lat, lon), dtype='float32')
        for step, lev in enumerate(level_num):
            level_values[0, :, step, :, :] = level_values[0, :, step, :, :] * lev
        
        lon_values = np.concatenate([np.concatenate([np.concatenate([longitude.reshape(1, lon)] * lat, axis=0).reshape(-1, lat, lon)] * level, axis=0).reshape(-1, level, lat, lon)] * ntime, axis=0).reshape(-1, ntime, level, lat, lon)
        lat_values = np.concatenate([np.concatenate([np.concatenate([np.sort(latitude)[::-1].reshape(lat, 1)] * lon, axis=1).reshape(-1, lat, lon)] * level, axis=0).reshape(-1, level, lat, lon)] * ntime, axis=0).reshape(-1, ntime, level, lat, lon)

        x_input = x
        y_input = y

        x_input = np.concatenate([x_input, level_values], axis=0)
        x_input = np.concatenate([x_input, lon_values], axis=0)
        x_input = np.concatenate([x_input, lat_values], axis=0)

        x_shape, y_shape = x_input.shape, y_input.shape
        
        x_input = np.transpose(x_input, (0, 2, 1, 3, 4))
        y_input = np.transpose(y_input, (0, 2, 1, 3, 4))

        flattened_x = x_input.reshape((x_shape[0], x_shape[2], x_shape[1] * x_shape[3] * x_shape[4]))
        flattened_y = y_input.reshape((y_shape[0], y_shape[2], y_shape[1] * y_shape[3] * y_shape[4]))
        
        scaler_X = []
        scaler_y = []
        normalized_x = np.zeros((x_shape[0], x_shape[2], x_shape[1] * x_shape[3] * x_shape[4])).astype('float32')
        normalized_y = np.zeros((y_shape[0], y_shape[2], y_shape[1] * y_shape[3] * y_shape[4])).astype('float32')
        
        for i in range(x_shape[2]):
            if Scale == True:
                scaler_X_i = StandardScaler()
                scaler_y_i = StandardScaler()
                normalized_x[:, i, :] = scaler_X_i.fit_transform(flattened_x[:, i, :].T).T
                normalized_y_fit = scaler_y_i.fit_transform(flattened_x[:5, i, :].T).T
                normalized_y[:, i, :] = scaler_y_i.transform(flattened_y[:, i, :].T).T
                scaler_X.append(scaler_X_i)
                scaler_y.append(scaler_y_i)
            else:
                scaler_X_i = Scale[0][i]
                scaler_y_i = Scale[1][i]
                normalized_x[:, i, :] = scaler_X_i.transform(flattened_x[:, i, :].T).T
                normalized_y[:, i, :] = scaler_y_i.transform(flattened_y[:, i, :].T).T
        
        x_input = normalized_x.reshape((x_shape[0], x_shape[2], x_shape[1], x_shape[3], x_shape[4]))
        y_input = normalized_y.reshape((y_shape[0], y_shape[2], y_shape[1], y_shape[3], y_shape[4]))
        x_input = np.transpose(x_input, (0, 2, 1, 3, 4))
        y_input = np.transpose(y_input, (0, 2, 1, 3, 4))
    else:  # Surface data
        land = np.load('./constant/terrain.npy')
        plant = np.load('./constant/plant.npy')
        land_values = np.concatenate([land.reshape(-1, lat, lon)] * ntime, axis=0).reshape(-1, ntime, lat, lon)
        plant_values = np.concatenate([plant.reshape(-1, lat, lon)] * ntime, axis=0).reshape(-1, ntime, lat, lon)
        
        x_input = x
        y_input = y
        x_input = np.concatenate([x_input, land_values], axis=0)
        x_input = np.concatenate([x_input, plant_values], axis=0)
        
        x_shape, y_shape = x_input.shape, y_input.shape
        
        flattened_x = x_input.reshape((x_shape[0], x_shape[1] * x_shape[2] * x_shape[3])).astype('float32')
        flattened_y = y_input.reshape((y_shape[0], y_shape[1] * y_shape[2] * y_shape[3])).astype('float32')
        
        if Scale == True:
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            normalized_x = scaler_X.fit_transform(flattened_x.T).T
            normalized_y_fit = scaler_y.fit_transform(flattened_x[:4, :].T).T
            normalized_y = scaler_y.transform(flattened_y.T).T
        else:
            normalized_x = Scale[0].transform(flattened_x.T).T
            normalized_y = Scale[1].transform(flattened_y.T).T
        
        x_input = normalized_x.reshape((x_shape[0], x_shape[1], 1, x_shape[2], x_shape[3]))
        y_input = normalized_y.reshape((y_shape[0], y_shape[1], 1, y_shape[2], y_shape[3]))
    
    # Use random patching instead of regular splitting
    x_dataset = split_data_random(x_input, pad_len, num_samples, overlap_ratio)
    y_dataset = split_data_random(y_input, pad_len, num_samples, overlap_ratio)
    
    if Scale == True:
        return x_dataset, y_dataset, (scaler_X if dim != 5 else scaler_X), (scaler_y if dim != 5 else scaler_y)
    else:
        return x_dataset, y_dataset

def restore_data_with_overlap(split_data_result, original_shape):
    """
    Reconstruct full image from patched data.

    Args:
        split_data_result: Patches of shape (num_patches, dim, level, pad_len, pad_len)
        original_shape: Original shape (dim, time, level, height, width)

    Returns:
        Reconstructed full array
    """
    pad_len = split_data_result.shape[-1]
    dim, time, level, num_rows, num_cols = original_shape
    num_rows_new = num_rows // pad_len
    num_cols_new = num_cols // pad_len
    remainder_rows = num_rows % pad_len
    remainder_cols = num_cols % pad_len
    restored_data = np.zeros((original_shape))  
    norepeat_piece = split_data_result[:num_rows_new*num_cols_new*time,:,:, :, :].reshape(time,num_rows_new,num_cols_new,dim,level,pad_len,pad_len)
    norepeat_piece = norepeat_piece.transpose(3,0,4,1,5,2,6).reshape(dim,time,level,num_rows_new*pad_len,num_cols_new*pad_len)
    restored_data[:,:,:,:num_rows_new*pad_len, :num_cols_new*pad_len] = norepeat_piece
    if remainder_rows != 0:
        last_row_piece = split_data_result[num_rows_new*num_cols_new*time:num_rows_new*num_cols_new*time+num_cols_new*time,:,:, :, :].reshape(time,1,num_cols_new,dim,level,pad_len,pad_len)
        last_row_piece = last_row_piece.transpose(3,0,4,1,5,2,6).reshape(dim,time,level,1*pad_len,num_cols_new*pad_len)
        restored_data[:,:,:,-pad_len:, :num_cols_new*pad_len] = last_row_piece
    if remainder_cols != 0:
        last_col_piece = split_data_result[num_rows_new*num_cols_new*time+num_cols_new*time:num_rows_new*num_cols_new*time+num_cols_new*time+num_rows_new*time,:,:, :, :].reshape(time,num_rows_new,1,dim,level,pad_len,pad_len)
        last_col_piece = last_col_piece.transpose(3,0,4,1,5,2,6).reshape(dim,time,level,num_rows_new*pad_len,1*pad_len)
        restored_data[:,:,:,:num_rows_new*pad_len, -pad_len:] = last_col_piece
    if  remainder_rows != 0 and remainder_cols != 0:
        last_corner_piece =split_data_result[-1,:,:, :, :].reshape(time,1,1,dim,level,pad_len,pad_len)
        last_corner_piece =last_corner_piece.transpose(3,0,4,1,5,2,6).reshape(dim,time,level,1*pad_len,1*pad_len)
        restored_data[:,:,:,-pad_len:, -pad_len:] = last_corner_piece

    return restored_data

def recovery_dataset(x, x_shape, scale):
    """
    Reverse normalization and reconstruct full data from patches.

    Args:
        x: Normalized patches
        x_shape: Original full data shape
        scale: Scaler(s) used for normalization

    Returns:
        Denormalized full data array
    """
    x = restore_data_with_overlap(x, x_shape)
    try:
        x=np.transpose(x, (0,2,1,3,4))
        x=x.reshape((x_shape[0] , x_shape[2] , x_shape[1]*x_shape[3]*x_shape[4]))
        for i in range(len(scale)):
            x[:,i,:]=scale[i].inverse_transform(x[:,i,:].T).T
        x=x.reshape(x_shape[0],x_shape[2],x_shape[1],x_shape[3],x_shape[4])
        x=np.transpose(x, (0,2,1,3,4))
    except TypeError as e:
        x=x.reshape((x_shape[0] , x_shape[1]*x_shape[3]*x_shape[4]))
        x=scale.inverse_transform(x.T).T
        x=x.reshape(x_shape[0],x_shape[1],x_shape[3],x_shape[4])
    return x