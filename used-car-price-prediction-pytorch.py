#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


train_data = pd.read_csv("C:\\Users\\Manav Jadaun\\OneDrive\\Desktop\\train-data.csv", na_values = ["null bhp"])
test_data = pd.read_csv("C:\\Users\\Manav Jadaun\\OneDrive\\Desktop\\test-data.csv", na_values = ["null bhp"])


# In[6]:


train_data.head()


# In[7]:


train_data.info()


# In[8]:


plt.figure(figsize = (16,8))
sns.countplot(x = train_data['Location'])


# In[9]:


plt.figure(figsize = (16,8))
sns.countplot(x = train_data['Location'], hue = train_data["Fuel_Type"])


# In[10]:


plt.figure(figsize = (16,8))
sns.boxplot(x="Location", y = "Price", data = train_data)


# In[11]:


train_data.isnull().sum()


# In[12]:


# Deleting New_Price column since it has too many null values

train_data.drop("New_Price", axis = 1, inplace = True)


# In[13]:


train_data.isnull().sum()


# In[14]:


train_data.corr()


# In[15]:


# removing Unnamed: 0 since it has no role

train_data.drop("Unnamed: 0", axis = 1, inplace = True)


# In[16]:


train_data["Seats"].value_counts()


# In[17]:


# since most of the cars have 5 Seats, we will fill the null values in Seats columns as "5"

train_data["Seats"].fillna(train_data["Seats"].value_counts().values[0], inplace = True)


# In[18]:


train_data.isnull().sum()


# In[19]:


def convert_power_data(val):
    if not pd.isnull(val):
        return float(val.split(' ')[0])
    return val

train_data["Power"] = train_data["Power"].apply(lambda val: convert_power_data(val))


# In[20]:


plt.figure(figsize = (16,8))
sns.scatterplot(x = train_data["Location"], y = train_data["Price"])


# In[21]:


plt.figure(figsize = (16,8))
sns.scatterplot(x = train_data["Power"], y = train_data["Price"])


# In[22]:


train_data["Power"].mean()


# In[23]:


train_data["Power"].fillna(train_data["Power"].mean(), inplace = True)


# In[24]:


def convert_engine_data(val):
    if not pd.isnull(val):
        return float(val.split(' ')[0])
    return val

train_data["Engine"] = train_data["Engine"].apply(lambda val: convert_power_data(val))


# In[25]:


plt.figure(figsize = (16,8))
sns.scatterplot(x = train_data["Engine"], y = train_data["Price"])


# In[26]:


train_data["Engine"].fillna(train_data["Engine"].median(), inplace = True)


# In[27]:


def convert_mileage_data(val):
    if not pd.isnull(val):
        return float(val.split(' ')[0])
    return val

train_data["Mileage"] = train_data["Mileage"].apply(lambda val: convert_power_data(val))


# In[28]:


train_data.dropna(inplace = True)


# In[29]:


train_data.info()


# In[30]:


# Dropping Name Column

train_data.drop(["Name", "Location"], axis = 1, inplace = True)


# In[31]:


train_data.head()


# In[32]:


import datetime
train_data['Total Years'] = datetime.datetime.now().year - train_data["Year"]


# In[33]:


train_data.head()


# In[34]:


train_data.drop('Year', axis = 1, inplace = True)


# In[35]:


train_data["Price"] = train_data["Price"] * 100000


# In[36]:


train_data.columns


# # Doing Embedded Encoding for Categorical Data

# In[37]:


cat_features = ['Fuel_Type', 'Transmission', 'Owner_Type']
out_features = 'Price'


# In[38]:


from sklearn.preprocessing import LabelEncoder

lbl_encoders = {}
lbl_encoders["Fuel_Type"] = LabelEncoder()
lbl_encoders["Fuel_Type"].fit_transform(train_data["Fuel_Type"])


# In[39]:


lbl_encoders = {}
for features in cat_features:
    lbl_encoders[features] = LabelEncoder()
    train_data[features] = lbl_encoders[features].fit_transform(train_data[features])


# In[40]:


train_data


# In[41]:


### stacking and converting into tensors

cat_features = np.stack([train_data["Fuel_Type"], train_data["Transmission"], train_data["Owner_Type"]], axis = 1)
cat_features


# In[42]:


import torch
cat_features = torch.tensor(cat_features, dtype = torch.int64)
cat_features


# In[ ]:


cont_features = []
for i in train_data.columns:
    if i in ['Fuel_Type', 'Transmission', 'Owner_Type', 'Price']:
        pass
    else:
        cont_features.append(i)
        
cont_features


# In[ ]:


cont_features = np.stack([train_data[i].values for i in cont_features], axis = 1)
cont_features = torch.tensor(cont_features, dtype = torch.float)
cont_features


# In[ ]:


cont_features.dtype


# In[ ]:


y=torch.tensor(train_data['Price'].values,dtype=torch.float).reshape(-1,1)
y


# In[43]:


cat_features.shape, cont_features.shape, y.shape


# In[44]:


cat_dims = [train_data[i].nunique() for i in ["Fuel_Type", "Transmission", "Owner_Type"]]
cat_dims


# In[45]:


embedding_dim = [(i, min(50, (i+1) // 2)) for i in cat_dims]
embedding_dim


# In[46]:


import torch
import torch.nn as nn

embed_rep = nn.ModuleList([nn.Embedding(i,e) for i,e in embedding_dim])

embed_rep


# In[ ]:


embedding_value = []
for ind, e in enumerate(embed_rep):
    embedding_value.append(e(cat_features[:, ind]))
    
embedding_value


# In[ ]:


z = torch.cat(embedding_value, 1)
z


# In[ ]:


drpout = nn.Dropout(0.4)
z = drpout(z)
z


# # Creating neural network using PyTorch

# In[ ]:


class UsedCarPricePredictionNN(nn.Module):
    def __init__(self, cat_dim, n_cont, layers, out_sz, p=0.5):
        super().__init__()
        embedded_dim = [(i, min(50, (i+1) // 2)) for i in cat_dim]
        self.embd_list = nn.ModuleList([nn.Embedding(inp, out) for inp, out in embedded_dim])
        self.drpout = nn.Dropout(p)
        self.batchnorm = nn.BatchNorm1d(n_cont)
        
        layerslist = []
        n_emb = sum([out for inp, out in embedded_dim])
        n_in = n_emb + n_cont
        
        for i in layers:
            layerslist.append(nn.Linear(n_in, i))
            layerslist.append(nn.ReLU(inplace = True))
            layerslist.append(nn.BatchNorm1d(i))
            layerslist.append(nn.Dropout(p))
            n_in = i
        layerslist.append(nn.Linear(layers[-1], out_sz))
        
        self.layers = nn.Sequential(* layerslist)
        
        
    def forward(self, x_cat, x_cont):
        embeddings = []
        for i, e in enumerate(self.embd_list):
            embeddings.append(e(x_cat[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.drpout(x)
        
        x_cont = self.batchnorm(x_cont)
        
        x = torch.cat([x, x_cont], axis=1)
        
        x = self.layers(x)
        
        return x
        
        


# In[ ]:


torch.manual_seed(100)

model = UsedCarPricePredictionNN(cat_dims, 6, [100, 50], 1, p = 0.4)


# In[ ]:


model


# In[ ]:


loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)


# In[ ]:


cat_features.shape, cont_features.shape, y.shape


# * Train Test Split

# In[ ]:


batch_size=6000
test_size=int(batch_size*0.15)
train_categorical=cat_features[:batch_size-test_size]
test_categorical=cat_features[batch_size-test_size:batch_size]
train_cont=cont_features[:batch_size-test_size]
test_cont=cont_features[batch_size-test_size:batch_size]
y_train=y[:batch_size-test_size]
y_test=y[batch_size-test_size:batch_size]


# In[ ]:


len(train_categorical),len(test_categorical),len(train_cont),len(test_cont),len(y_train),len(y_test)


# # Training for 2400 epochs

# In[ ]:


epochs=2400
final_losses=[]
for i in range(epochs):
    i=i+1
    y_pred=model(train_categorical,train_cont)
    loss=torch.sqrt(loss_function(y_pred,y_train)) ### RMSE
    final_losses.append(loss)
    if i%100==1:
        print("Epoch number: {} and the loss : {}".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# In[ ]:


plt.plot(range(epochs), final_losses)
plt.ylabel('RMSE Loss')
plt.xlabel('epoch');


# # Evaluating

# In[ ]:


y_pred=""
with torch.no_grad():
    y_pred=model(test_categorical,test_cont)
    loss=torch.sqrt(loss_function(y_pred,y_test))
print('RMSE: {}'.format(loss))


# In[ ]:


data_verify=pd.DataFrame(y_test.tolist(),columns=["Test"])


# In[ ]:


data_predicted=pd.DataFrame(y_pred.tolist(),columns=["Prediction"])


# In[ ]:


final_output=pd.concat([data_verify,data_predicted],axis=1)
final_output['Difference']=final_output['Test']-final_output['Prediction']
final_output.head()


# In[ ]:


final_output = final_output / (100000)


# In[ ]:


final_output


# # Saving the model

# In[ ]:


torch.save(model,'SalePrice.pt')


# In[ ]:


torch.save(model.state_dict(),'PriceWeights.pt')


# # Loading model and Predicting a value

# In[ ]:


cat_size=[4,2,4]
model1=UsedCarPricePredictionNN(cat_size,6,[100,50],1,p=0.4)


# In[ ]:


model1.load_state_dict(torch.load('PriceWeights.pt'))


# In[ ]:


model1.eval()


# In[ ]:


catee = np.array([1,1,0])
conti = np.array([4.1000e+04, 1.9670e+01, 1.5820e+03, 1.2620e+02, 5.0000e+00, 6.0000e+00])

catee = torch.tensor(catee.reshape(1,3), dtype = torch.int64)
conti = torch.tensor(conti.reshape(1,6), dtype= torch.float)


# In[ ]:


model1(catee, conti)


# In[ ]:




