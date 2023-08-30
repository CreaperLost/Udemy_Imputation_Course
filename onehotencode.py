import numpy as np
import pandas as pd

class CustomOneHotEncoder:

    # Initialize the object
    # self.categories will probably be provided.
    def __init__(self,categories):
        self.category_per_name = categories
        self.categories_ = {}

        for i,feature_name in enumerate(self.category_per_name):
            self.categories_[i]  = np.array(self.category_per_name[feature_name])
            #self.categories_[i] = [index for index, value in enumerate(np.array(self.category_per_name[feature_name]))]
            #print('Prev',np.array(self.category_per_name[feature_name]))
            #print('New',self.categories_[i])

    def turn_vmaps_to_float(self,unique_values):
        new_l = []
        for i in unique_values:
            try:
                new_l.append(str(float(i)))
            except ValueError:
                new_l.append(str(i))
            
        assert len(new_l) == len(unique_values)
        return new_l
    
    def turn_value_to_float(self,value):
        try:
            return str(float(value))
        except:
            return str(value)

    # One-hot-encode the dataset.
    # for each feature, get the #categories
    # initialize an array with #categories features
    # 
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if not self.categories_:
            raise ValueError("Call fit() before transforming data.")

        encoded_matrix = []
        #print(X.shape[1])
        for feature in range(X.shape[1]):
            unique_categories = self.categories_[feature]
            conv_cat= unique_categories #self.turn_vmaps_to_float() #unique_categories #
            #print(feature,unique_categories)
            # Samples - One-Hot-Features  *FOR ONE FEATURE ONLY!*
            encoded_features = np.zeros((X.shape[0], len(unique_categories)))
            #print(encoded_features)
            # for row i , value - value in a feature.
            
            for i, value in enumerate(X[:, feature]):
                to_test = value #self.turn_value_to_float(value) #value#
                
                if to_test in conv_cat:
                    index = np.where(np.array(conv_cat) == to_test)[0][0]
                    encoded_features[i, index] = 1
                # if the feature value is not in categories or is missing, then whole row == np.nan
                
                elif np.isnan(value): #== np.nan:
                    encoded_features[i, :] = np.nan
                #print(value,np.isnan(value))
            encoded_matrix.append(encoded_features)

        return np.concatenate(encoded_matrix, axis=1)


    
    def inverse_transform(self, encoded_matrix):
        if not self.categories_:
            raise ValueError("Call fit() before inverse transforming data.")

        if isinstance(encoded_matrix, pd.DataFrame):
            encoded_matrix = encoded_matrix.values

        num_features = len(self.categories_)
        num_categories = [len(self.categories_[feature]) for feature in range(num_features)]

        decoded_values = []

        start_index = 0
        # for each feature. (that had being one-hot-encoded)
        for feature in range(num_features):
            #find the index where one-hot-starts
            end_index = start_index + num_categories[feature]

            # keep the sub matrix (one-hot-encoded) for that feature
            feature_matrix = encoded_matrix[:, start_index:end_index]
            decoded_feature = []

            
            for row in feature_matrix:
                indices = np.where(row == 1)[0]
                
                # in this case we probably have missing values
                if len(indices) == 0:
                    max_index = np.argmax(row)
                    decoded_feature.append(self.categories_[feature][max_index])
                else:
                    # get the category value for that index.
                    decoded_feature.append(self.categories_[feature][indices[0]])

            decoded_values.append(decoded_feature)
            #Move the index for the next one-hot sub matrix.
            start_index = end_index

        return np.array(decoded_values).T
