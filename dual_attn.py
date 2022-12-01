#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 00:32:05 2022

@author: zhuchen
"""


import torch
import torch.nn as nn
from sparsemax import Sparsemax


class LayerNorm(nn.Module):
    def __init__(self, n_state, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(n_state))
        self.beta = nn.Parameter(torch.zeros(n_state))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = (x - mu).pow(2).mean(-1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return self.gamma * x + self.beta
    

class DualAttnModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, label_dim=1, scale=10, page_scale=10, attn_type='dot'):
        super(DualAttnModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.embeddings.weight.data.uniform_(-0.1, 0.1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
        self.affine = nn.Linear(embed_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.attn_linear = nn.Linear(hidden_dim, hidden_dim)
        self.scale = scale
        self.page_scale = page_scale
        self.V = nn.Parameter(torch.randn(hidden_dim, 1)) # for words
        self.W = nn.Parameter(torch.randn(hidden_dim, 1)) # for web pages
        self.decoder = nn.Linear(hidden_dim, label_dim, bias=False)
        self.attn_type = attn_type
        self.page_sparsemax = Sparsemax(dim=-1)
        
    
    def forward(self, seq_ids, num_pages, seq_lengths=None):
        seq_embs = self.embeddings(seq_ids)
        seq_embs = self.dropout(seq_embs)
        batch_size , max_page, max_len, hidden_dim = seq_embs.size() #batch_size(#comp), #webpages, #words, hidden_dim
        hidden_vecs = seq_embs
        if self.attn_type == 'dot':
            inter_out = hidden_vecs
        else:
            inter_out = torch.tanh(self.attn_linear(hidden_vecs))
            
        #-----<token-level attention>-----
        #inter_out = self.LayerNorm(inter_out)
        scores = torch.matmul(inter_out, self.V).squeeze(-1)
        # (batch_size x max_len x hidden_dim) x (hidden_dim, 1)
        scores = scores/self.scale
        # Mask the padding values
        batch_size, max_page = seq_lengths.size() # batch_size(#comp) x max_page; element: seq_len
        mask = torch.zeros_like(seq_ids)
        for i in range(batch_size):
            for j in range(num_pages[i]):
                mask[i, j, seq_lengths[i, j]:] = 1
        scores = scores.masked_fill(mask.bool(), -999)
        #Softmax, batch_size*max_page*1*max_len
        #attn = self.word_sparsemax(scores).unsqueeze(2)
        attn = self.softmax(scores).unsqueeze(2)
        #weighted sum, batch_size*max_page*hidden_dim
        webpage_vec = torch.einsum('abcd, abde -> abe', attn, hidden_vecs)
        #webpage_vec = self.relu(self.wp_affine(webpage_vec))
        
        #webpage_vec = self.LayerNorm(webpage_vec)
        #-----<page-level attention>-----
        #num_pages = torch.tensor([2, 1]) # batch_size x 1
        page_scores = torch.matmul(webpage_vec, self.W).squeeze(-1) # batch_size x num_page
        page_scores = page_scores/self.page_scale
        page_mask = torch.zeros_like(page_scores)
        for i in range(batch_size):
            page_mask[i, num_pages[i]:] = 1
        page_scores = page_scores.masked_fill(page_mask.bool(), -9999)
        page_attn = self.page_sparsemax(page_scores).unsqueeze(1)
        final_vec = torch.bmm(page_attn, webpage_vec).squeeze(1)
        #finaL_vec = self.LayerNorm(final_vec)
        final_vec = self.dropout(final_vec)
        senti_scores = self.decoder(final_vec)
        probs = self.sigmoid(senti_scores)
        return probs, senti_scores, attn, page_attn, final_vec, page_scores, webpage_vec
    
    
    def load_vector(self, pretrained_vectors, trainable=False):
        '''
        Load pre-savedd word embeddings
        '''
        self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_vectors))
        self.embeddings.weight.requires_grad = trainable
        print('embeddings loaded')
        