import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer




class mm():
    
    def __init__(self,parameters: dict, names: list, vmaps: dict,
                 missing_values=np.nan):
       
        

        
        self.names = names
        self.new_names = names
        self.vmaps = vmaps
        self.new_vmaps = vmaps


        #The indexes for categorical features in feature list.
        self.catindx = sorted([names.index(i) for i in vmaps.keys()])
        self.numindx = sorted([names.index(i) for i in names if i not in vmaps.keys()])
        self.cat_names = [self.names[i] for i in self.catindx]
        self.num_names = [self.names[i] for i in self.numindx]
        self.missing_values = missing_values


    def _initial_imputation(self, X):
        
        

        X_filled = X.copy()

        #Code by George Paterakis
        #Mean Impute continous , Mode Impute Categorical


        #Mean Impute
        if len(self.numindx) >0 :
            if self.initial_imputer_Mean is None:
                self.initial_imputer_Mean = SimpleImputer(missing_values=self.missing_values,strategy='mean')
                self.initial_imputer_Mean.fit(X[:,self.numindx])
            else:
                X_filled[:,self.numindx] = self.initial_imputer_Mean.transform(X[:,self.numindx])
           
        
        #Mode Impute
        if len(self.catindx) >0 :
            if self.initial_imputer_Mode is None:
                self.initial_imputer_Mode = SimpleImputer(missing_values=self.missing_values,strategy='most_frequent')
                self.initial_imputer_Mode.fit(X[:,self.catindx])
            else:
                X_filled[:,self.catindx] = self.initial_imputer_Mode.transform(X[:,self.catindx])
            
        return X_filled



    def fit_transform(self, X, y=None):
        """Fits the imputer on X and return the transformed X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where "n_samples" is the number of samples and
            "n_features" is the number of features.

        y : ignored.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
            The imputed input data.
        """
        device = torch.device('cpu')
        
        self.initial_imputer_Mean = None
        self.initial_imputer_Mode = None

        #Code by George Paterakis
        #List of Lists -->> np.array with samples as rows and features as columns.
        X=np.transpose(np.array(X))
        X_r = self._initial_imputation(X)
        X=np.transpose(np.array(X_r)).tolist()


        return X

    def transform(self, X):
        """Imputes all missing values in X.

        Note that this is stochastic, and that if random_state is not fixed,
        repeated calls, or permuted input, will yield different results.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
             The imputed input data.
        """
        #check_is_fitted(self)

        #Code by George Paterakis
        #List of Lists -->> np.array with samples as rows and features as columns.
        X=np.transpose(np.array(X))        
        X_filled = self._initial_imputation(X)
        X=np.transpose(np.array(X_filled)).tolist()

        return X,self.new_names, self.new_vmaps

    def fit(self, X, y=None):
        """Fits the imputer on X and return self.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where "n_samples" is the number of samples and
            "n_features" is the number of features.

        y : ignored

        Returns
        -------
        self : object
            Returns self.
        """

        self.fit_transform(X)
        return self


