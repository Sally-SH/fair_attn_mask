from unittest.mock import _patch_dict
import torch
import torch.nn as nn
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.long

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout_ratio):
        super().__init__()

        self.d_model = d_model # d_model
        self.nhead = nhead # nhead 
        self.head_dim = d_model // nhead # head_dim 

        self.qLinear = nn.Linear(d_model, d_model) # query lineqr
        self.kLinear = nn.Linear(d_model, d_model) # key linear
        self.vLinear = nn.Linear(d_model, d_model) # value linear

        self.oLinear = nn.Linear(d_model, d_model) # output linear

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, query, key, value):

        batch_size = query.shape[0]
        query_len = query.shape[1]
        value_len = key_len = key.shape[1]

        # query: [batch_size, query_len, d_model]
        # key: [batch_size, key_len, d_model]
        # value: [batch_size, value_len, d_model]
 
        Q = self.qLinear(query)
        K = self.kLinear(key)
        V = self.vLinear(value)

        # Q: [batch_size, query_len, d_model]
        # K: [batch_size, key_len, d_model]
        # V: [batch_size, value_len, d_model]

        # d_model = nhead * head_dim
        Q = rearrange(Q, 'b l (h d) -> b h l d', h=self.nhead)
        K = rearrange(K, 'b l (h d) -> b h l d', h=self.nhead)
        V = rearrange(V, 'b l (h d) -> b h l d', h=self.nhead)

        # Q: [batch_size, nhead, query_len, head_dim]
        # K: [batch_size, nhead, key_len, head_dim]
        # V: [batch_size, nhead, value_len, head_dim]

        #=========Scaled Dot-Product Attention=========
        weight = torch.matmul(Q, rearrange(K, 'b h l d -> b h d l')) / np.sqrt(self.head_dim)

        # weight: [batch_size, nhead, query_len, key_len]
        attention = torch.softmax(weight, dim=-1)
        # attention: [batch_size, nhead, query_len, key_len]
        c = torch.matmul(self.dropout(attention), V)
        # c: [batch_size, nhead, query_len, head_dim]

        #=========Scaled Dot-Product Attention=========
        # reshape & concat
        c = rearrange(c, 'b h l d -> b l (h d)')
        # c: [batch_size, query_len, d_model]
        output = self.oLinear(c)
      # output: [batch_size, query_len, d_model]
        return output, attention
    
class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, d_model, ff_dim, dropout_ratio):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):

        # x: [batch_size, seq_len, d_model]
        # The MLP contains two layers with a GELU non-linearity
        x = self.dropout(nn.functional.gelu(self.linear1(x)))
        # x: [batch_size, seq_len, ff_dim]
        x = self.linear2(x)
        # x: [batch_size, seq_len, d_model]
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, ff_dim, dropout_ratio):
        super().__init__()

        self.layerNorm1 = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)
        self.multiHeadAttentionLayer = MultiHeadAttentionLayer(d_model, nhead, dropout_ratio)
        self.positionWiseFeedForward = PositionWiseFeedForwardLayer(d_model, ff_dim, dropout_ratio)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, src):

        # src: [batch_size, src_len, d_model] 
        # Layernorm (LN) is applied before every block
        _src = self.layerNorm1(src)
        _src, attention = self.multiHeadAttentionLayer(_src, _src, _src)

        # residual connections after every block

        # Dropout, when used, is applied after every dense layer except for the the qkv-projections
        # and directly after adding positional- to patch embeddings. 
        src = src + self.dropout(_src)
        _src = self.layerNorm2(src)

        # src: [batch_size, src_len, d_model]
        _src = self.positionWiseFeedForward(_src)
        src = src + self.dropout(_src)
        # src: [batch_size, src_len, d_model]

        return src, attention
    
class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, nhead, ff_dim, dropout_ratio):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, nhead, ff_dim, dropout_ratio) for _ in range(n_layers)])

    def forward(self, src):

        # src: [batch_size, src_len]
        attentions = []
        for layer in self.layers:
            src, attention = layer(src)
            attentions.append(attention)
        # src: [batch_size, src_len, d_model]

        return src, attentions
    
class ImageEmbedding(nn.Module):
  def __init__(self, channel, patch_size, D):
    super().__init__()
    self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(channel*patch_size*patch_size),
            nn.Linear(channel*patch_size*patch_size, D),
            nn.LayerNorm(D),
        )
    # learnable cls_token (only for classification)
    self.cls_token = nn.Parameter(torch.randn(1, 1, D))

  def forward(self, image):

    b, c, w, h = image.shape
    # image: [batch_size, channel, width, height]
    # n = num_w * num_h = (w*h) / (p1*p2)
    embedded_patches = self.to_patch_embedding(image)
    cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
    embedded_patches = torch.cat((cls_tokens, embedded_patches), dim=1)

    return embedded_patches
  

class TokPosEmbedding(nn.Module):
  def __init__(self, c, img_size, p, d_model, dropout_ratio):
    super().__init__()
    self.tokEmbedding = ImageEmbedding(c, p, d_model)
    self.posEmbedding = nn.Embedding(100, d_model) 
    self.d_model = d_model
    self.dropout = nn.Dropout(dropout_ratio)
    num_patches = (img_size[0] // p) * (img_size[1] // p)
    self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

  def forward(self, src):

    # src: [batch_size, width, height, channel]
    src = self.tokEmbedding(src)
    # src: [batch_size, src_len, d_model]

    batch_size = src.shape[0]
    src_len = src.shape[1]

    src += self.pos_embedding[:, :src_len]
    src = self.dropout(src)
    # Dropout, when used, is applied after every dense layer except for the the qkv-projections and directly after adding positional- to patch embeddings

    # src: [batch_size, src_len, d_model]

    return src

class VisionTransformer(nn.Module):
    '''
    ch : channel
    img_size : image size
    patch : patch size
    d_model : embed dim
    n_layers : num encode layers
    n_head : num head
    ff_dim : feed forward dim
    dropout_ratio : dropout rate
    output_dim : output dim
    '''
    def __init__(self, ch, img_size, patch, d_model, n_layers, n_head, ff_dim, dropout_ratio, output_dim):
        super().__init__()

        self.encEmbedding = TokPosEmbedding(ch, img_size, patch, d_model, dropout_ratio)
        self.encoder = Encoder(d_model, n_layers, n_head, ff_dim, dropout_ratio)
        self.layerNorm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, output_dim)
        # self.classifier = nn.Sequential(nn.Linear(d_model, d_model*2),
        #                                      nn.ReLU(),
        #                                      nn.Dropout(0.3),
        #                                      nn.Linear(d_model*2, output_dim))
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):    
        # src: [batch_size, src_len]
        src = self.encEmbedding(src)
        enc_src, attentions = self.encoder(src)
        # enc_src: [batch_size, src_len, d_model]
        # classification head
        cls_token = enc_src[:,0,:] # cls token
        # cls_token: [batch_size, d_model]
        cls_token = self.layerNorm(cls_token)
        output = self.linear(cls_token)
        # output: [batch_size, output_dim]

        return output, attentions
