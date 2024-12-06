#!/usr/bin/env python
# coding: utf-8

# In[164]:


import gzip
import json
import math
import numpy as np
import random
import sklearn
import string
from collections import defaultdict
from nltk.stem.porter import *
from sklearn import linear_model
import dateutil
from scipy.sparse import lil_matrix 

# In[ ]:


train_dataset, val_dataset, test_dataset = [],[],[]

with open("data/filter_all_t.json") as f:
    l = json.load(f)

# In[166]:


l.keys()

# In[167]:


for d in l["train"]:
    train_dataset.append(d)

for d in l["val"]:
    val_dataset.append(d)

for d in l["test"]:
    test_dataset.append(d)
    
len(train_dataset), len(val_dataset), len(test_dataset)

# In[ ]:


## sample example of train_dataset
train_dataset[0]

# In[339]:


users = {}
business = {}
interactions = []

for d in train_dataset:
    if d["user_id"] not in users.keys(): users[d["user_id"]] = len(users)
    if d["business_id"] not in business.keys(): business[d["business_id"]] = len(business)
    interactions.append((d["user_id"], d["business_id"], d["rating"]))
    
val_interactions = []

for d in val_dataset:
    if d["user_id"] not in users.keys(): users[d["user_id"]] = len(users)
    if d["business_id"] not in business.keys(): business[d["business_id"]] = len(business)
    val_interactions.append((d["user_id"], d["business_id"], d["rating"]))
    
test_interactions = []

for d in test_dataset:
    if d["user_id"] not in users.keys(): users[d["user_id"]] = len(users)
    if d["business_id"] not in business.keys(): business[d["business_id"]] = len(business)
    test_interactions.append((d["user_id"], d["business_id"], d["rating"]))
    
len(users), len(business), len(interactions), len(val_interactions), len(test_interactions)

# In[340]:


random.shuffle(interactions)
mean_rating = sum([r for _,_,r in interactions])/len(interactions)

# In[341]:


import math
import numpy as np
import random
import torch as torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, Dataset


# In[342]:


class Brentford(nn.Module):
    
    def __init__(self, K, user_size,item_size):
        self.K = K
        self.beta_user = torch.from_numpy(np.random.normal(size=user_size)).type(dtype= torch.float32)
        self.beta_item = torch.from_numpy(np.random.normal(size=item_size)).type(dtype= torch.float32)
        # self.gamma_user = torch.from_numpy(np.random.normal(size=(user_size,K))).type(dtype= torch.float32)
        # self.gamma_item = torch.from_numpy(np.random.normal(size=(item_size,K))).type(dtype= torch.float32)
        self.alpha = torch.tensor(np.random.normal(size=1)).type(dtype= torch.float32) ## zero initialization of alpha
        
        
        self.beta_user.requires_grad = True
        self.beta_item.requires_grad = True
        # self.gamma_user.requires_grad = True
        # self.gamma_item.requires_grad = True
        self.alpha.requires_grad = True
    
    def forward(self,bu,bi):
        
        # return self.alpha + bu + bi+ torch.dot(gu, gi)
        return self.alpha + bu + bi
            
            
def train(model,lr,lam,train_set,val_set,users,items):
    
    device = None
    # check availability of gpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print ("Running on: ", device)

    optimizer = torch.optim.SGD([model.alpha, model.beta_user, model.beta_item], lr= lr, weight_decay=1e-4)
    steps = len(train_set)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    losses = []
    val_losses = []
    
    ts = time.time() 
    loss_tot = []
    for e in range(2):
        for idx, (u,i,r) in enumerate(train_set):
            optimizer.zero_grad()
            
            beta_u = model.beta_user[users[u]]
            beta_i = model.beta_item[items[i]]
            # gamma_u = model.gamma_user[users[u],:]
            # gamma_i = model.gamma_item[items[i],:]
            
            y = torch.tensor([float(r)]).to(device)
            # pred = model.forward(beta_u,beta_i,gamma_u,gamma_i) 
            # loss = (pred-y)**2+lam*(beta_u+\
            #         beta_i+ torch.norm(gamma_u)+torch.norm(gamma_i))
            pred = model.forward(beta_u,beta_i).to(device)
            loss = (pred-y)**2+lam*(beta_u**2+beta_i**2)
            loss.backward()
            
            #print(model.beta_user[users[u]],model.beta_item[items[i]],model.gamma_user[users[u],:],model.gamma_item[items[i],:])
            optimizer.step()
            #scheduler.step()
            loss_tot.append(loss)
            if idx % 100 == 0:
                    print("epoch {}, iter {}, loss: {}".format(e, idx, loss.item()))
                    #print(model.beta_user[users[u]],model.beta_item[items[i]],model.gamma_user[users[u],:],model.gamma_item[items[i],:])
                    losses.append(torch.mean(torch.tensor(loss_tot)).item())
                    loss_tot = []
                    val_losses.append(val(model,val_set,users,items))
                    #print("iter: {}, validation loss: {}".format(idx, val_losses[-1]))
    return losses, val_losses

def val(model,val_set,users,items):
    loss_tot = []
    with torch.no_grad():
        for idx, (u,i,r) in enumerate(val_set):
            if u in users.keys() and i in items.keys():
                beta_u = model.beta_user[users[u]]
                beta_i = model.beta_item[items[i]]
                # gamma_u = model.gamma_user[users[u],:]
                # gamma_i = model.gamma_item[items[i],:]
                # pred = model.forward(beta_u,beta_i,gamma_u,gamma_i)
                pred = model.forward(beta_u,beta_i)
            else:
                pred = mean_rating   

            y = torch.tensor([float(r)])
            loss = (pred-y)**2
            loss_tot.append(loss)
            #if idx % 100 == 0:
                #print("iter {}, loss: {}".format(idx, loss.item()))
                #print(model.beta_user[users[u]],model.beta_item[items[i]],model.gamma_user[users[u],:],model.gamma_item[items[i],:])
    losses = torch.mean(torch.tensor(loss_tot)).item()

    return losses

# In[398]:


b_lfm = Brentford(2,len(users),len(business))

# In[ ]:


loss, val_loss = train(b_lfm,0.005,0.05,interactions,val_interactions,users,business)

# In[379]:


from matplotlib import pyplot as plt

## plot training and validation losses
plt.plot(np.arange(len(loss)),loss,c="b")
plt.plot(np.arange(len(val_loss)),val_loss,c="r")
plt.show()

# In[380]:


def test(model,test_set,users,items):
    predictions=[]
    labels = []
    loss_tot = []
    losses = []
    with torch.no_grad():
        for idx, (u,i,r) in enumerate(test_set):
            if u in users.keys() and i in items.keys():
                beta_u = model.beta_user[users[u]]
                beta_i = model.beta_item[items[i]]
                # gamma_u = model.gamma_user[users[u],:]
                # gamma_i = model.gamma_item[items[i],:]
                # pred = model.forward(beta_u,beta_i,gamma_u,gamma_i)
                pred = model.forward(beta_u,beta_i)
            else:
                pred = mean_rating   

            y = torch.tensor([float(r)])
            labels.append((y))
            predictions.append((u,i,pred))
            loss = (pred-y)**2
            
            loss_tot.append(loss)
            if idx % 100 == 0:
                losses.append(torch.mean(torch.tensor(loss_tot)).item())
                loss_tot = []
    return losses, predictions

# In[381]:


## testing the data
test_losses, test_precitions = test(b_lfm,test_interactions,users,business)

# In[382]:


metrics = {}

# In[383]:


def MRR(predictions,labels,users,items,k=3):
    userRanks = defaultdict(list)
    labelRanks = defaultdict(list)
    
    for (u,i,p) in predictions:
        userRanks[u].append((p,i))
        
    for (u,i,p) in labels:
        labelRanks[u].append((p,i))
        
    for u in userRanks:
        userRanks[u].sort(reverse=True)
    
    for u in labelRanks:
        labelRanks[u].sort(reverse=True)
    
    totalMRR = 0
    
    for u, preds in userRanks.items():
        rank = 0
        ps = [p[1] for p in labelRanks[u][:k]]
        for i, (r,b) in enumerate(preds):
            if b in items and b in ps:
                rank = i+1
                break
            
        if rank > 0:
            totalMRR += 1/rank
    
    return totalMRR/len(userRanks) if len(userRanks) > 0 else 0

# In[384]:


mean_reciproal_rank = MRR(test_precitions,test_interactions,users,business)
mean_reciproal_rank
metrics["MRR"] = mean_reciproal_rank

# In[385]:


mean_MSE =  sum(test_losses)/len(test_losses)
mean_MSE
metrics["average MSE"] = mean_MSE

# In[386]:


def NDCG(predictions,labels,users,items,k=3):
    userRanks = defaultdict(list)
    labelRanks = defaultdict(list)
    
    for (u,i,p) in predictions:
        userRanks[u].append((p,i))
        
    for (u,i,p) in labels:
        labelRanks[u].append((p,i))
        
    for u in userRanks:
        userRanks[u].sort(reverse=True)
    
    for u in labelRanks:
        labelRanks[u].sort(reverse=True)
        
    totalDCG = 0
    for u, preds in userRanks.items():
        rank = 0
        rating = 0
        ps = [p[1] for p in labelRanks[u][:k]]
        for i, (r,b) in enumerate(preds):
            if b in items and b in ps:
                rank = i+1
                rating = r
                totalDCG += rating/np.log2(rank+1)
    
    totalIDCG = 0
    for u,preds in labelRanks.items():
        rank = 0
        rating = 0
        for i, (r,b) in enumerate(preds[:k]):
            if b in items:
                rank = i+1
                rating = r
                totalIDCG += rating/np.log2(rank+1)

    return totalDCG, totalDCG/totalIDCG

# In[387]:


DCG3, NDCG3 = NDCG(test_precitions, test_interactions, users, business)
DCG3, NDCG3

metrics["NDCG"] = NDCG3

# In[388]:


def Precision(predictions,labels,users,items,k=3):
    userRanks = defaultdict(list)
    labelRanks = defaultdict(list)
    
    for (u,i,p) in predictions:
        userRanks[u].append((p,i))
        
    for (u,i,p) in labels:
        labelRanks[u].append((p,i))
        
    for u in userRanks:
        userRanks[u].sort(reverse=True)
    
    for u in labelRanks:
        labelRanks[u].sort(reverse=True)
        
    totalPreKU = 0
    for u, preds in userRanks.items():
        relevance= 0
        ps = [p[1] for p in labelRanks[u][:k]]
        for i, (r,b) in enumerate(preds):
            if b in items and b in ps:
                relevance+=1
        totalPreKU += relevance/k

    return totalPreKU/len(userRanks)

def Recall(predictions,labels,users,items,k=3):
    userRanks = defaultdict(list)
    labelRanks = defaultdict(list)
    
    for (u,i,p) in predictions:
        userRanks[u].append((p,i))
        
    for (u,i,p) in labels:
        labelRanks[u].append((p,i))
        
    for u in userRanks:
        userRanks[u].sort(reverse=True)
    
    for u in labelRanks:
        labelRanks[u].sort(reverse=True)
        
    totalRKU = 0
    for u, preds in userRanks.items():
        relevance = 0
        ps = [p[1] for p in labelRanks[u][:k]]
        for i, (r,b) in enumerate(preds):
            if b in items and b in ps:
                relevance+=1
        totalRKU += relevance/len(preds)

    return totalRKU/len(userRanks)
                    

# In[389]:


P3 = Precision(test_precitions,test_interactions,users,business)
P3
metrics["precision"] = P3

# In[390]:


R3 = Recall(test_precitions,test_interactions, users, business)
R3
metrics["recall"] = R3

# In[391]:


metrics

# In[392]:


## Recommender function given a 
def recommendTopNRestaurants(predictions, n=3):
    topN = defaultdict(list)
    for (u,i,p) in predictions:
        topN[u].append((p, i))
    # for uid, ratings in topN.items():
    #     ratings.sort(key=lambda x: x[1], reverse=True)
    #     topN[uid] = ratings[:n]
    for u, ratings in topN.items():
        ratings.sort(reverse=True)
        topN[u] = ratings[:n]
    return topN

# In[393]:


top3 = recommendTopNRestaurants(test_precitions)
for u, recs in top3.items():
    print(f"User {u}: {recs}")

# In[394]:


f = open("Ali_Metrics.txt", 'w')
f.write(str(metrics) + '\n')
f.close()
