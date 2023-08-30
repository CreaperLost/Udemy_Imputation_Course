# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tqdm import tqdm as tq
from algorithms.gain_utils import normalization, renormalization, rounding
from algorithms.gain_utils import xavier_init
from algorithms.gain_utils import binary_sampler, uniform_sampler, sample_batch_index
import torch
import torch.nn.functional as F
import numpy as np






class Gain():
    '''Impute missing values in data_x
    
    Args:
        - data_x: original data with missing values
        - gain_parameters: GAIN network parameters:
        - batch_size: Batch size
        - hint_rate: Hint rate
        - alpha: Hyperparameter
        - iterations: Iterations
        
    Returns:
        - imputed_data: imputed data
    '''
    def __init__(self,parameters: dict, names: list, vmaps: dict) -> None:
        # System parameters
        self.batch_size = parameters.get('batch_size',64)
        self.hint_rate = parameters.get('hint_rate',0.9)
        self.alpha = parameters.get('alpha',1)
        self.iterations = parameters.get('iterations',10000)

        self.theta_G= None
        self.names = names
        self.new_names = names
        self.vmaps = vmaps
        self.new_vmaps = vmaps


        #The indexes for categorical features in feature list.
        """self.catindx = [names.index(i) for i in vmaps.keys()]
        self.numindx = [names.index(i) for i in names if i not in vmaps.keys()]
        self.cat_names = [i for i in vmaps.keys()]
        self.num_names = [i for i in names if i not in vmaps.keys()]"""

        self.catindx = sorted([names.index(i) for i in vmaps.keys()])
        self.numindx = sorted([names.index(i) for i in names if i not in vmaps.keys()])
        self.cat_names = [self.names[i] for i in self.catindx]
        self.num_names = [self.names[i] for i in self.numindx]
  
    # 1. Generator
    def generator(self,new_x,m):
        inputs = torch.cat(dim = 1, tensors = [new_x,m])  # Mask + Data Concatenate
        G_h1 = F.relu(torch.matmul(inputs, self.theta_G[0]) + self.theta_G[3])
        G_h2 = F.relu(torch.matmul(G_h1, self.theta_G[1]) + self.theta_G[4])   
        G_prob = torch.sigmoid(torch.matmul(G_h2, self.theta_G[2]) + self.theta_G[5]) # [0,1] normalized Output
                
        return G_prob

    def fit(self,X):
        #Code by George Paterakis
        #List of Lists -->> np.array with samples as rows and features as columns.
        data_x=np.transpose(np.array(X))

        # Define mask matrix
        data_m = 1-np.isnan(data_x)
        
        # Other parameters
        no, dim = data_x.shape
        
        # Hidden state dimensions
        h_dim = int(dim)
        
        # Normalization
        norm_data, self.norm_parameters = normalization(data_x,None,self.catindx)
        norm_data_x = np.nan_to_num(norm_data, 0)
        

        #Discriminator
        D_W1 = torch.tensor(xavier_init([dim*2, h_dim]),requires_grad=True)     # Data + Hint as inputs
        D_b1 = torch.tensor(np.zeros(shape = [h_dim]),requires_grad=True)

        D_W2 = torch.tensor(xavier_init([h_dim, h_dim]),requires_grad=True)
        D_b2 = torch.tensor(np.zeros(shape = [h_dim]),requires_grad=True)

        D_W3 = torch.tensor(xavier_init([h_dim, h_dim]),requires_grad=True)
        D_b3 = torch.tensor(np.zeros(shape = [h_dim]),requires_grad=True)       # Output is multi-variate

        theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
        

        #Generator
        G_W1 = torch.tensor(xavier_init([dim*2, h_dim]),requires_grad=True)     # Data + Mask as inputs (Random Noises are in Missing Components)
        G_b1 = torch.tensor(np.zeros(shape = [h_dim]),requires_grad=True)

        G_W2 = torch.tensor(xavier_init([h_dim, h_dim]),requires_grad=True)
        G_b2 = torch.tensor(np.zeros(shape = [h_dim]),requires_grad=True)

        G_W3 = torch.tensor(xavier_init([h_dim, h_dim]),requires_grad=True)
        G_b3 = torch.tensor(np.zeros(shape = [h_dim]),requires_grad=True)

        self.theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

        # 2. Discriminator
        def discriminator(new_x, h):
            inputs = torch.cat(dim = 1, tensors = [new_x,h])  # Hint + Data Concatenate
            D_h1 = F.relu(torch.matmul(inputs, D_W1) + D_b1)  
            D_h2 = F.relu(torch.matmul(D_h1, D_W2) + D_b2)
            D_logit = torch.matmul(D_h2, D_W3) + D_b3
            D_prob = torch.sigmoid(D_logit)  # [0,1] Probability Output
            
            return D_prob


        def discriminator_loss(M, New_X, H):
            # Generator
            G_sample = self.generator(New_X,M)
            # Combine with original data
            Hat_New_X = New_X * M + G_sample * (1-M)

            # Discriminator
            D_prob = discriminator(Hat_New_X, H)

            #%% Loss
            D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) + (1-M) * torch.log(1. - D_prob + 1e-8))
            return D_loss

        def generator_loss(X, M, New_X, H):
            #%% Structure
            # Generator
            G_sample = self.generator(New_X,M)

            # Combine with original data
            Hat_New_X = New_X * M + G_sample * (1-M)

            # Discriminator
            D_prob = discriminator(Hat_New_X, H)

            #%% Loss
            G_loss1 = -torch.mean((1-M) * torch.log(D_prob + 1e-8))
            MSE_train_loss = torch.mean((M * New_X - M * G_sample)**2) / torch.mean(M)

            G_loss = G_loss1 + self.alpha * MSE_train_loss 

            #%% MSE Performance metric
            MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
            return G_loss, MSE_train_loss, MSE_test_loss

        
        optimizer_D = torch.optim.Adam(params=theta_D)
        optimizer_G = torch.optim.Adam(params=self.theta_G)
        
        #%% Start Iterations
        for it in tq(range(self.iterations)):    
            
            #%% Inputs
            mb_idx = sample_batch_index(no, min(self.batch_size, no))
            X_mb = norm_data_x[mb_idx,:]  
            
            Z_mb = uniform_sampler(0, 0.01, min(self.batch_size, no), dim) 
            M_mb = data_m[mb_idx,:]  
            # Sample hint vectors
            H_mb_temp = binary_sampler(self.hint_rate, min(self.batch_size, no), dim)
            H_mb = M_mb * H_mb_temp
           
            
            New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce
            
            X_mb = torch.tensor(X_mb).double()
            M_mb = torch.tensor(M_mb).double()
            H_mb = torch.tensor(H_mb).double()
            New_X_mb = torch.tensor(New_X_mb).double()
            
            optimizer_D.zero_grad()
            D_loss_curr = discriminator_loss(M=M_mb, New_X=New_X_mb, H=H_mb)
            D_loss_curr.backward()
            optimizer_D.step()
            
            optimizer_G.zero_grad()
            G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = generator_loss(X=X_mb, M=M_mb, New_X=New_X_mb, H=H_mb)
            
            G_loss_curr.backward()
            optimizer_G.step()    
                
            #%% Intermediate Losses
            if it % 10000 == 0:
                print('Iter: {}'.format(it),end='\t')
                print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())),end='\t')
                print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr.item())))
        return self
               
    def transform(self,X):
        #Code by George Paterakis
        #List of Lists -->> np.array with samples as rows and features as columns.
        data_x=np.transpose(np.array(X))

        # Define mask matrix
        data_m = 1-np.isnan(data_x)
    
        # Other parameters
        no, dim = data_x.shape
    
        # Hidden state dimensions
        h_dim = int(dim)
    
        # Normalization
        norm_data, norm_parameters = normalization(data_x,self.norm_parameters,self.catindx)
        norm_data_x = np.nan_to_num(norm_data, 0)

        ## Return imputed data      
        Z_mb = uniform_sampler(0, 0.01, no, dim) 
        M_mb = data_m
        X_mb = norm_data_x          
                
        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce

        X_mb = torch.tensor(X_mb).double()
        M_mb = torch.tensor(M_mb).double()
        New_X_mb = torch.tensor(New_X_mb).double()
        
        def test_loss(X, M, New_X):
            #Generator
            G_sample = self.generator(New_X,M)

            # MSE Performance metric
            MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
            return MSE_test_loss, G_sample

        MSE_final, Sample = test_loss(X=X_mb, M=M_mb, New_X=New_X_mb)
                        
        imputed_data = M_mb * X_mb + (1-M_mb) * Sample
 
        imputed_data =imputed_data.detach().numpy()
        # Renormalization
        imputed_data = renormalization(imputed_data, self.norm_parameters,self.catindx)  
        
        # Rounding if categoricals
        if len(self.catindx) >0 :
            imputed_data = rounding(imputed_data, self.catindx)  

        imputed_data = np.transpose(np.array(imputed_data)).tolist()

        return imputed_data,self.names,self.vmaps
