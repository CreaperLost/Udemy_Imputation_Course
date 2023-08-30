import numpy as np
from sklearn.utils import  check_random_state
from sklearn.utils.validation import  check_is_fitted
from sklearn.utils._mask import _get_mask
from sklearn.impute._base import SimpleImputer
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects.numpy2ri as numpy2ri
from rpy2.robjects import r, pandas2ri
from algorithms.onehotencode import CustomOneHotEncoder


class SoftImpute:
    
    def __init__(self,parameters: dict, names: list, vmaps: dict,
                 missing_values=np.nan):

        package_names =  {'softImpute'}

        if not all(rpackages.isinstalled(x) for x in package_names):
            utils = rpackages.importr('utils')
            utils.chooseCRANmirror(ind=1)

            packnames_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
            
            if len(packnames_to_install) >0:
                utils.install_packages(robjects.StrVector(packnames_to_install))
        
        self.softImpute = rpackages.importr('softImpute')
        self.base = rpackages.importr('base')


        pandas2ri.activate()

        self.estimator = None
        self.nPcs = parameters.get("nPcs",5)
        self.lamda = parameters.get("lambda",0)
        self.names = names
        self.new_names = names
        self.vmaps = vmaps
        self.new_vmaps = vmaps

        numpy2ri.activate()

        self.vmaps = vmaps
        # Contains feature names
        self.names = names

        # In case of mixed data, numerical features and categorical features will be mixed.
        self.new_names = self.names
        # How many categorical variables do we got.
        self.n_categoricals = len(vmaps)
        
        # keep names of cat features
        self.cat_names = list(vmaps.keys())

        self.one_hot_encoder  = CustomOneHotEncoder(categories=self.vmaps)

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

    def variance_explained(self,singular_vals):
        # Calculate the sum of squares of all singular values
        print(singular_vals)
        sum_of_squares_all_singular_values = np.sum(np.square(singular_vals))
        # Calculate the variance explained by each singular value
        variances_explained = [(s ** 2) / sum_of_squares_all_singular_values for s in singular_vals]
        # Calculate cumulative variance explained
        cumulative_variances = np.cumsum(variances_explained)
        # Define the desired explained variance threshold
        desired_explained_variance = 0.90
        # Find the index where cumulative variance exceeds the threshold
        num_components_to_keep = np.argmax(cumulative_variances >= desired_explained_variance) + 1
        return num_components_to_keep

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

        X_conc = np.array(final_df)


        #Replace NAN with NA.
        X_conc = robjects.r('''
                        function(X){ 
                        X[X=="NaN"] <- NA 
                        X }''')(X_conc)

        self.nPcs = min([X_conc.shape[0],X_conc.shape[1]]) -1 
        
        # Find the 99% nPCs needed.
        self.estimator = self.softImpute.softImpute(X_conc,self.nPcs,self.lamda,type="svd")
        d_list = list(self.estimator.__getitem__(1))
        # Refit.
        self.nPcs = self.variance_explained(d_list)
        #self.lamda = d_list[self.nPcs]
        self.estimator = self.softImpute.softImpute(X_conc,self.nPcs,self.lamda,type="svd")
        #print(d_list,type(d_list))
        imputed_data = self.softImpute.complete(X_conc,self.estimator,unscale=False)

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
        #Code by George Paterakis
        #List of Lists -->> np.array with samples as rows and features as columns.
        X=np.transpose(np.array(X))


        # Need to make sure that X has the same order in names and actual values
        # Check also that the self.vmaps is indeed in range, before and after imputation   
        X_DF = pd.DataFrame(X,columns=self.names)

        final_df,dummy_names = self.split_data(X_DF,'Testing')

        X_conc = np.array(final_df)
        X_conc = robjects.r('''
                        function(X){ 
                        X[X=="NaN"] <- NA 
                        X }''')(X_conc)

        imputed_data= self.softImpute.complete(X_conc,self.estimator,unscale=False)
        
        # Special case of input - output data mismatch.
        if final_df.shape[1] != imputed_data.shape[1]:
            raise RuntimeError
        
        imputed_data_df = pd.DataFrame(imputed_data,columns = final_df.columns)

        imputed_final_df = self.concat_data(imputed_data_df,dummy_names)

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