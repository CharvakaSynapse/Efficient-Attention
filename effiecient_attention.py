# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 20:50:12 2025

@author: subha_qfp58yg
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

#%%
class EfficientAttention(nn.Module):
    def __init__(self, n_heads, in_channel,key_channel,value_channel):
        super().__init__()
        self.n_heads = n_heads 
        self.in_channel = in_channel
        self.key_channel = key_channel
        self.value_channel = value_channel
        self.key= nn.Conv2d(self.in_channel,self.key_channel,1)
        self.query = nn.Conv2d(self.in_channel,self.key_channel,1)
        self.value = nn.Conv2d(self.in_channel,self.value_channel,1)
        self.reprojection = nn.Conv2d(self.value_channel, self.in_channel,1)
    def forward(self,X):
        n,_,h,w = X.shape
        key_output = self.key(X).reshape(n, self.key_channel,h*w)
        query_output = self.query(X).reshape(n, self.key_channel,h*w)
        value_output = self.value(X).reshape(n, self.value_channel,h*w)
        
        keys = rearrange(key_output, 'n (h c_k) s -> h n c_k s', h= self.n_heads)
        queries = rearrange(query_output, 'n (h c_k) s -> h n c_k s', h= self.n_heads)
        values = rearrange(value_output, 'n (h c_k) s -> h n c_k s', h= self.n_heads)
        ## n_heads in dim=0 so that we can loop over 
        
        attention_list =[]
        for key,query,value in zip(keys,queries,values):
            # key and query shape = batch size(n) x key channel per head(dk_h) x h*w
            key_softmax = F.softmax(key, dim = -1)# softmax over spatial
            query_softmax = F.softmax(query, dim = 1) #over channel
            
            
            
            # key represent as d_k number of feature maps instead n number of
            # feature vectors. For example lets assume ith row in key matrix dk_h_i
            # represents feature of sky in different position , then each 
            # element of this row will represent 
            # a value over all position, the position (pixel) that represents human will have zero value 
            # so does pixel taht represents grass . after softmax when we perform the dot 
            #product with value matrix then we get a context vector and that context vector 
            # represent the expected value of value matrix with respect to dk_h_i. 
            # the context matrix is called global context. 
            # similar way the column vector in Q represents feature value in each position
            # after columwise softmax and dot product with global context will represents
            # expected value of global context with respect to every position (pixel)
            # this aggregate is attention 
            
            context = torch.einsum('nks,nvs -> nkv', key_softmax,value)
            attention = torch.einsum('nks,nkv -> nvs', query_softmax,context)
            ## reverting back to 2D spatial
            attention = rearrange(attention,'n v (h w) -> n v h w', h=h,w=w)
            attention_list.append(attention) 
        
        attention_out = torch.cat(attention_list, dim = 1)
        reprojection = self.reprojection(attention_out)
        out = reprojection + X
        return out
            
            
#%% testing 

x= torch.randn(1,1,3,3)
def test_attn(x):
    e_attn = EfficientAttention(n_heads=2, in_channel =1, key_channel =2, value_channel=2)
         
    assert e_attn(x).shape  == x.shape
    print('Test Successful!')
          
test_attn(x)            
        
        
