import numpy as np
from numpy import linalg
import pandas as pd
import time

# function to help print the numpy arrays in a cleaner manner
def print_numpy(ndarray):
    with np.printoptions(precision = 4, suppress = True, formatter = {'float': '{:0.4f}'.format}, linewidth = 100):
        print(ndarray)

def scale_dataframe(dataframe: pd.DataFrame(), return_as_dataframe: bool = False):
    """Scales/normalizes the data such that each column has zero mean and unit variance"""
    
    # convert dataframe to array
    ndarray = np.asanyarray(dataframe)
    
    # calculate the mean and standard deviations of the columns
    means = ndarray.mean(axis = 0, keepdims = True)
    std_devs = ndarray.std(axis = 0, ddof = 1, keepdims = True)
    
    # subtract mean and divide by std. dev. element-wise
    scaled_array = (ndarray - means) / std_devs
    
    # return the scaled data in the appropriate form
    if return_as_dataframe:
        return pd.DataFrame(data = scaled_array, columns = dataframe.columns)
    else:
        return scaled_array

def calculate_covariance_matrix(ndarray):
    """Since we are handling scaled data with zero mean, when computing the covariance matrix
    there is no need to subtract the mean from the data points. We just need the dot product of
    the transposed matrix with itself, and each element of the product divided by the number of samples.
    Transpose has dimensions [n_features, n_samples].
    Non-transpose matrix has dimensions [n_samples, n_features].
    The dot product gives [n_features, n_features] shape."""
    
    return (np.dot(ndarray.T, ndarray) * np.true_divide(1, ndarray.shape[0])).squeeze()

def pca(df: pd.DataFrame, svd: bool = False, print_steps: bool = True):
    if print_steps:
        print('\nDataset dimensions:')
        print(df.shape)
        
        print('\nStarting dataset head:')
        print(df.head())
    
    # scale the data into a numpy array, further manipulation is easier with numpy
    dataset_scaled_array = scale_dataframe(df)
    
    # also get dataframe version to view the scaled data
    dataset_scaled = pd.DataFrame(data = dataset_scaled_array, columns = df.columns)
    
    if print_steps:
        print('\nScaled dataset head:')
        print(dataset_scaled.head())
    
    # calculate the covariance matrix
    covariance_matrix = calculate_covariance_matrix(dataset_scaled_array)
    
    if print_steps:
        print('\nCovariance matrix:')
        print_numpy(covariance_matrix)
    
    if svd == False:
        start = time.time()
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        exec_time = time.time() - start
        
        explained_variances = []
        for i in range(len(eigenvalues)):
            explained_variances.append(eigenvalues[i] / np.sum(eigenvalues))
        
        pca_dict = dict(
            method = 'eigen decomposition',
            eigenvalues = eigenvalues,
            eigenvectors = eigenvectors,
            exec_time = exec_time,
            explained_var = explained_variances
        )
        
        if print_steps:
            print('_____________________________________________________________')
            print('\nUsing EIGEN DECOMPOSITION:')
            print('_____________________________________________________________')
            print('\nExecution time:')
            print(exec_time)
            print('\nEigenvalues:')
            print_numpy(eigenvalues)
            print('\nEigenvectors: ')
            print_numpy(eigenvectors)
            print('Explained variances of the principal components:')
            print_numpy(np.asarray(explained_variances))
    
    else:
        start = time.time()
        U, sigma, V = np.linalg.svd(covariance_matrix, full_matrices = False)
        exec_time = time.time() - start
        
        explained_variances = []
        for i in range(len(sigma)):
            explained_variances.append(sigma[i] / np.sum(sigma))
            
            pca_dict = dict(
                method = 'eigen decomposition',
                eigenvalues = sigma,
                eigenvectors = U,
                exec_time = exec_time,
                explained_var = explained_variances
            )
        
        if print_steps:
            print('_____________________________________________________________')
            print('\nUsing SVD - SINGULAR VALUE DECOMPOSITION:')
            print('_____________________________________________________________')
            print('\nExecution time:')
            print(exec_time)
            print('\nEigenvalues:')
            print_numpy(sigma)
            print('\nEigenvectors: ')
            print_numpy(U)
            print('Explained variances of the principal components:')
            print_numpy(np.asarray(explained_variances))
    
    return pca_dict

