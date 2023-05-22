# ----------------------------------------------------------------------------------------------
# GSRTR Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved 
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
Transformer with verb classifier
"""
import copy
import torch
import torch.nn.functional as F
from typing import Optional, List
from torch import nn, Tensor

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, 
                dim_feedforward=2048, dropout=0.15, activation="relu",output_dim=211):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_verb_classes = output_dim

        # encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # classifer (for verb prediction)
        self.verb_classifier = nn.Sequential(nn.Linear(d_model, d_model*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.3),
                                             nn.Linear(d_model*2, self.num_verb_classes))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, enc_verb_query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        device = enc_verb_query_embed.device

        # Transformer Encoder (w/ verb classifier)
        enc_verb_query_embed = enc_verb_query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        zero_mask = torch.zeros((bs, 1), dtype=torch.bool, device=device)
        mem_mask = torch.cat([zero_mask, mask], dim=1)
        verb_with_src = torch.cat([enc_verb_query_embed, src], dim=0)
        memory, attentions = self.encoder(verb_with_src, src_key_padding_mask=mem_mask, pos=pos_embed)
        vhs, memory = memory.split([1, h*w], dim=0) 
        vhs = vhs.view(bs, -1)
        verb_pred = self.verb_classifier(vhs).view(bs, self.num_verb_classes)

        return verb_pred, attentions

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        attentions = []

        for layer in self.layers:
            output, attention = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            attentions.append(attention)

        if self.norm is not None:
            output = self.norm(output)

        return output, attentions

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.15, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # tensor[0] is verb token (there is no positional encoding)
        return tensor if pos is None else torch.cat([tensor[:1], (tensor[1:] + pos)], dim=0)

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, attention = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, attention

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def build_transformer(args):
    return Transformer(d_model=args.hidden_dim,
                        dropout=args.dropout,
                        nhead=args.nheads,
                        dim_feedforward=args.dim_feedforward,
                        num_encoder_layers=args.enc_layers,
                        output_dim = args.num_verb)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")