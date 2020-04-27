import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 

class MLP_model(nn.Module): #
    def __init__(self,input_feat=33,hidden_dim_1=512,hidden_dim_2=1024,hidden_dim_3=4096,output_unit=1):
        super(MLP_model,self).__init__()
        self.linear_1=nn.Linear(input_feat,hidden_dim_1)
        self.linear_2=nn.Linear(hidden_dim_1,hidden_dim_2)
        self.linear_3=nn.Linear(hidden_dim_2,hidden_dim_3)
        self.linear_4=nn.Linear(hidden_dim_3,hidden_dim_2)
        self.linear_5=nn.Linear(hidden_dim_2,hidden_dim_1)
        self.linear_6=nn.Linear(hidden_dim_1,input_feat)
        self.linear_7=nn.Linear(input_feat,output_unit)
        #self.bn1 = nn.BatchNorm1d(num_features=feat_size)

    def forward(self,x):
        x=self.linear_1(x)
        x=F.relu(x)
        x=self.linear_2(x)
        x=F.relu(x)
        x=self.linear_3(x)
        x=F.relu(x)
        x=self.linear_4(x)
        x=F.relu(x)
        x=self.linear_5(x)
        x=F.relu(x)
        x=self.linear_6(x)
        x=F.relu(x)
        x=self.linear_7(x)
        return(x)

class sales_dataset(Dataset):
    def __init__(self,data,val):
        self.data=data
        self.val=val
    def __len__(self):
        return(len(self.data))
    def __getitem__(self,idx):
        data_item=self.data[idx,:]
        val_item=self.val[idx]
        return(data_item)

def train_MLP(X_train,Y_train,X_valid,Y_valid,model,batch_size=32,device_id=0,epochs=50):

    #numpy data normalization
    X_train=X_train.values #converting to numpy array 
    Y_train=Y_train.values
    X_valid=X_valid.values
    Y_valid=Y_valid.values
    #normalize each column to min-max scale



    #initialize dataset 
    train_sales_ds=sales_dataset(X_train)
    train_sales_dl=DataLoader(dataset=train_sales_ds,batch_size=batch_size,shuffle=True)
    valid_sales_ds=sales_dataset(X_valid)
    valid_sales_dl=DataLoader(dataset=valid_sales_ds,batch_size=batch_size,shuffle=True)
    #initialize optimizer
    optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_id)

    model=model.to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    #for epoch in range(epochs):



if __name__ == "__main__":
    #load data

    data = pd.read_pickle('../data/tot_data_new_v2.pkl')
    print(data.columns)
    test  = pd.read_csv('../data/test.csv').set_index('ID')
    data.fillna(0,inplace=True)
    scaler=MinMaxScaler()
    data=scaler.fit_transform()
    """model=MLP_model()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(model)
    print('Total number of parameters:%d' %(params))"""
    