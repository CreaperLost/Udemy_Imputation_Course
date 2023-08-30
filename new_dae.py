import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer



class Autoencoder(nn.Module):
    def __init__(self, dim , theta,dropout):
        super(Autoencoder, self).__init__()
        self.dim = dim

        self.drop_out = nn.Dropout(p=dropout)

        self.encoder = nn.Sequential(
            nn.Linear(dim+(theta*0), dim+(theta*1)),
            nn.Tanh(),
            nn.Linear(dim+(theta*1), dim+(theta*2)),
            nn.Tanh(),
            nn.Linear(dim+(theta*2), dim+(theta*3))
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim+(theta*3), dim+(theta*2)),
            nn.Tanh(),
            nn.Linear(dim+(theta*2), dim+(theta*1)),
            nn.Tanh(),
            nn.Linear(dim+(theta*1), dim+(theta*0))
        )

    def forward(self, x):
        x = x.view(-1, self.dim)
        x_missed = self.drop_out(x)

        z = self.encoder(x_missed)
        out = self.decoder(z)

        out = out.view(-1, self.dim)

        return out



class DAE():

    def __init__(self,parameters: dict, names: list, vmaps: dict,
                 missing_values=np.nan):


        self.theta = parameters.get("theta",7)
        self.drop_out = parameters.get("dropout",0.5)
        self.batch_size = parameters.get("batch_size",64)
        self.epochs = parameters.get("epochs",500)
        self.lr = parameters.get("lr",0.01)
        self.dim = len(names)

        self.model = None


        torch.manual_seed(0)


        self.onehot  = OneHotEncoder(handle_unknown='ignore',sparse=False)


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




    def forward(self, x):
        x = x.view(-1, self.dim)
        x_missed = self.drop_out(x)

        z = self.encoder(x_missed)
        out = self.decoder(z)

        out = out.view(-1, self.dim)

        return out


    def _initial_imputation(self, X):



        X_filled = X.copy()

        #Code by George Paterakis
        #Mean Impute continous , Mode Impute Categorical


        #Mean Impute
        if len(self.numindx) >0 :
            if self.initial_imputer_Mean is None:
                self.initial_imputer_Mean = SimpleImputer(missing_values=self.missing_values,strategy='mean')
                self.initial_imputer_Mean.fit(X[:,self.numindx])
                X_filled[:,self.numindx] = self.initial_imputer_Mean.transform(X[:,self.numindx])
            else:
                X_filled[:,self.numindx] = self.initial_imputer_Mean.transform(X[:,self.numindx])


        #Mode Impute
        if len(self.catindx) >0 :
            if self.initial_imputer_Mode is None:
                self.initial_imputer_Mode = SimpleImputer(missing_values=self.missing_values,strategy='most_frequent')
                self.initial_imputer_Mode.fit(X[:,self.catindx])
                X_filled[:,self.catindx] = self.initial_imputer_Mode.transform(X[:,self.catindx])
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

        data_m = 1-np.isnan(X)

        X_r = self._initial_imputation(X)

        #if numericals
        if len(self.numindx) >0 :
            X_num  = X_r[:,self.numindx]
            X_conc = X_num


        #if categoricals
        if len(self.catindx) >0 :
            #Do one hot encoding to cat variables.
            X_cat = X_r[:,self.catindx]
            #X_cat = pd.DataFrame(X_cat,columns=self.vmaps.keys())
            self.onehot.fit(X_cat)
            X_cat=self.onehot.transform(X_cat)
            X_conc = X_cat



        #If mixed type then concat
        if len(self.catindx) >0 and len(self.numindx) >0:
            X_conc  = np.concatenate((X_num ,X_cat),axis=1)


        train_data = torch.from_numpy(X_conc).float()

        train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=min(self.batch_size, X.shape[0]),shuffle=True)


        cost_list = []
        early_stop = False

        self.dim = X_conc.shape[1]

        self.model = Autoencoder(dim = self.dim,theta = self.theta,dropout=self.drop_out).to(device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), momentum=0.99, lr=self.lr, nesterov=True)


        for epoch in range(self.epochs):

            total_batch = len(train_data)//min(self.batch_size, X.shape[0])

            for i, batch_data in enumerate(train_loader):

                batch_data = batch_data.to(device)
                reconst_data = self.model(batch_data)

                cost = self.loss(reconst_data, batch_data)

                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                #if (i+1) % (total_batch//2) == 0:
                #if epoch % 100 ==0 and i == 1:
                #   print('Epoch [%d/%d], lter [%d/%d], Loss: %.6f'%(epoch+1, self.epochs, i+1, total_batch, cost.item()))

                # early stopping rule 1 : MSE < 1e-06
                if cost.item() < 1e-06 :
                    early_stop = True
                    break

                cost_list.append(cost.item())

            if early_stop :
                break

        #Evaluate
        self.model.eval()
        filled_data = self.model(train_data.to(device))
        filled_data_train = filled_data.cpu().detach().numpy()


        #if mixed slice.
        if len(self.catindx) >0 and len(self.numindx) >0:
            X_num_sliced,X_cat_sliced = filled_data_train[:,:len(self.numindx)] , filled_data_train[:,len(self.numindx):]
            X_cat_sliced=self.onehot.inverse_transform(X_cat_sliced)
            filled_data_train  = np.concatenate((X_num_sliced ,X_cat_sliced),axis=1)
            self.new_names = self.num_names + self.cat_names
        elif len(self.catindx) >0:
            filled_data_train=self.onehot.inverse_transform(filled_data_train)

        #add mask

        X=np.transpose(np.array(filled_data_train)).tolist()


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
        device = torch.device('cpu')
        #check_is_fitted(self)

        #Code by George Paterakis
        #List of Lists -->> np.array with samples as rows and features as columns.
        X=np.transpose(np.array(X))

        #1 if not missing , 0 if missing.
        data_m = 1-np.isnan(X)
        data_nm = np.isnan(X)-0
        #keeps nan
        #gets original values. - > (data_m)*X
        #gets imputed values. - > (1-data_m)*X



        X_filled = self._initial_imputation(X)

        X_orig = X_filled

        #if numericals
        if len(self.numindx) >0 :
            X_num  = X_filled[:,self.numindx]
            X_conc = X_num

        #if categoricals
        if len(self.catindx) >0 :
            #Do one hot encoding to cat variables.
            X_cat = X_filled[:,self.catindx]
            #X_cat = pd.DataFrame(X_cat,columns=self.vmaps.keys())
            X_cat=self.onehot.transform(X_cat)
            X_conc = X_cat



        #If mixed type then concat
        if len(self.catindx) >0 and len(self.numindx) >0:
            X_conc  = np.concatenate((X_num ,X_cat),axis=1)

        X_filled = torch.from_numpy(X_conc).float()
        #Evaluate
        self.model.eval()

        #Transform Test set
        filled_data = self.model(X_filled.to(device))
        filled_data_test = filled_data.cpu().detach().numpy()
        X_r = filled_data_test


        #if mixed slice.
        if len(self.catindx) >0 and len(self.numindx) >0:
            X_num_sliced,X_cat_sliced = filled_data_test[:,:len(self.numindx)] , filled_data_test[:,len(self.numindx):]
            X_cat_sliced=pd.DataFrame(X_cat_sliced)
            X_cat_sliced=self.onehot.inverse_transform(X_cat_sliced)
            X_r  = np.concatenate((X_num_sliced ,X_cat_sliced),axis=1)
            self.new_names = self.num_names + self.cat_names
        elif len(self.catindx) >0:
            X_r=self.onehot.inverse_transform(filled_data_test)

        #add mask
        #Keep the original values and add the. imputations through X_R
        X_r= (data_m*X_orig) + (data_nm*X_r)

        #Code by George Paterakis
        #Turn np.array for transform to List of Lists
        #Samples turn to columns , and columns to rows.

        X=np.transpose(np.array(X_r)).tolist()

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

