import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from sklearn.impute import SimpleImputer
from algorithms.onehotencode import CustomOneHotEncoder
from sklearn.impute import KNNImputer

class KNN:


    def __init__(self,parameters: dict, names: list, vmaps: dict):

        assert parameters!=None and len(parameters)>0
        assert names!=None and len(names)>0
        assert vmaps!=None and len(vmaps)>=0

        self.k = parameters.get('k',None)
        assert self.k != None

        # Dimensions of the features
        self.dim = len(names)

        # Contains the mapping of categorical to value range.
        self.vmaps = vmaps
        # Convert elements of lists to strings
        """for key, value_list in self.vmaps.items():
            self.vmaps[key] = [str(item) for item in value_list]"""

        self.model = KNNImputer(missing_values=np.nan,n_neighbors=self.k)

        self.one_hot_encoder  = CustomOneHotEncoder(categories=self.vmaps)

        # Contains feature names
        self.names = names

        # In case of mixed data, numerical features and categorical features will be mixed.
        self.new_names = self.names

        

        # How many categorical variables do we got.
        self.n_categoricals = len(vmaps)
        
        # keep names of cat features
        self.cat_names = list(vmaps.keys())


        if len(self.cat_names) > 0:
            # Sanity check that categorical variable names in the column_names.
            assert set(self.cat_names).issubset(names)

        self.num_names = [x for x in self.names if x not in self.cat_names]



    def split_data(self,X_DF,step=None):
        assert step!=None

        dummy_names = [] 
 
        if len(self.num_names) > 0:
            num_df = X_DF[self.num_names].copy()

        if len(self.cat_names) > 0: 
            cat_df = X_DF[self.cat_names].copy()

            X_cat=self.one_hot_encoder.transform(cat_df)

            # Create dummy feature names for the NumPy array
            dummy_names = ['Feature{}'.format(i+1) for i in range(X_cat.shape[1])]

            one_hot_cat_df = pd.DataFrame(X_cat,columns=dummy_names)
        
        
        if len(self.num_names) > 0 and len(self.cat_names) > 0:

            concatenated_names = list(num_df.columns) + dummy_names 
            final_df = pd.concat([num_df,one_hot_cat_df], axis=1)
            final_df.columns = concatenated_names

        elif len(self.num_names) > 0:
            final_df = num_df
        elif len(self.cat_names) > 0:
            final_df = one_hot_cat_df
        else:
            raise RuntimeError
        
        return final_df,dummy_names
    

    def concat_data(self,imputed_data_df,dummy_names):
        
        if len(self.num_names) > 0 and len(self.cat_names) > 0:
            imputed_numerical_df = imputed_data_df[self.num_names].copy()
            imputed_categorical_df = imputed_data_df[dummy_names].copy()
            inverse_imputed_categorical = self.one_hot_encoder.inverse_transform(imputed_categorical_df)
            imputed_categorical_df =pd.DataFrame(inverse_imputed_categorical,columns = self.cat_names)
            imputed_final_df = pd.concat((imputed_numerical_df,imputed_categorical_df),axis=1)
            self.new_names = self.num_names + self.cat_names
        elif len(self.cat_names) > 0:
            inverse_imputed_categorical = self.one_hot_encoder.inverse_transform(imputed_data_df)
            imputed_final_df = pd.DataFrame(inverse_imputed_categorical,columns = self.cat_names)
        elif len(self.num_names) > 0:
            imputed_final_df = imputed_data_df

        return imputed_final_df

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

        #Code by George Paterakis
        #List of Lists -->> np.array with samples as rows and features as columns.
        
        X=np.transpose(np.array(X))
        
        # Need to make sure that X has the same order in names and actual values
        # Check also that the self.vmaps is indeed in range, before and after imputation   
        X_DF = pd.DataFrame(X,columns=self.names)

        final_df,dummy_names = self.split_data(X_DF,'Training')

        self.model.fit(final_df)

        # Here we call the transform. which returns the data to the original space?
        imputed_data = self.model.transform(final_df)
   
        # Special case of input - output data mismatch.
        if final_df.shape[1] != imputed_data.shape[1]:
            raise RuntimeError
        
        imputed_data_df = pd.DataFrame(imputed_data,columns = final_df.columns)

        imputed_final_df = self.concat_data(imputed_data_df,dummy_names)

        #add mask
        X=np.transpose(np.array(imputed_final_df)).tolist()

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

        X_DF = pd.DataFrame(X,columns=self.names)

        final_df,dummy_names = self.split_data(X_DF,'Testing')

        imputed_data = self.model.transform(final_df)

        # Special case of input - output data mismatch.
        if final_df.shape[1] != imputed_data.shape[1]:
            raise RuntimeError
        
        imputed_data_df = pd.DataFrame(imputed_data,columns = final_df.columns)

        imputed_final_df = self.concat_data(imputed_data_df,dummy_names)

        """print('Opos bgainei apo tin KNN transform')
        print(imputed_final_df)
        print(imputed_final_df.info())"""

        X=np.transpose(np.array(imputed_final_df)).tolist()

        return X,self.new_names, self.vmaps

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

