# ----------------------------------------------------------------------------------------------
# GSRTR Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved 
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
GSRTR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, accuracy, map_f1)
from .backbone import build_backbone
from .transformer import build_transformer


class FAIR(nn.Module):
    """ GSRTR model for Grounded Situation Recognition"""
    def __init__(self, backbone, transformer):
        """ Initialize the model.
        Parameters:
            - backbone: torch module of the backbone to be used. See backbone.py
            - transformer: torch module of the transformer architecture. See transformer.py
            - num_noun_classes: the number of noun classes
            - vidx_ridx: verb index to role index
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_verb_queries = 504

        # hidden dimension for queries and image features
        hidden_dim = transformer.d_model

        # query embeddings
        self.enc_verb_query_embed = nn.Embedding(1, hidden_dim)

        # 1x1 Conv
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        

    def forward(self, samples):
        """ 
        Parameters:
               - samples: The forward expects a NestedTensor, which consists of:
                        - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - targets: This has verbs, roles and labels information
               - inference: boolean, used in inference
        Outputs:
               - out: dict of tensors. 'pred_verb', 'pred_noun', 'pred_bbox' and 'pred_bbox_conf' are keys
        """
        MAX_NUM_ROLES = 6
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        # model prediction

        verb_pred, attentions = self.transformer(self.input_proj(src), 
                                                    mask, self.enc_verb_query_embed.weight, 
                                                    pos[-1])      
        # outputs
        return verb_pred, attentions


class LabelSmoothing(nn.Module):
    """ NLL loss with label smoothing """
    def __init__(self, smoothing=0.0):
        """ Constructor for the LabelSmoothing module.
        Parameters:
                - smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class ImSituCriterion(nn.Module):
    """ 
    Loss for GSRTR with SWiG dataset, and GSRTR evaluation.
    """
    def __init__(self, weight_dict):
        """ 
        Create the criterion.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.loss_function_img = LabelSmoothing(0.3)
        self.loss_function_mask = LabelSmoothing(0.3)


    def forward(self, targets, img_outputs, mask_outputs, eval=False):
        """ This performs the loss computation, and evaluation of GSRTR.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             eval: boolean, used in evlauation
        """
        # top-1 & top 5 verb acc and calculate verb loss
        out = {}
        verb_acc_topk = [0,0]
        # losses 
        gt_verbs = targets.max(1, keepdim=False)[1]
        img_verb_pred_logits = img_outputs.squeeze(1)
        verb_acc_topk = accuracy(img_verb_pred_logits, gt_verbs, topk=(1, 5))
        img_verb_loss = self.loss_function_img(img_verb_pred_logits, gt_verbs)
        out['loss_img'] = img_verb_loss
        
        mask_verb_pred_logits = mask_outputs.squeeze(1)
        mask_verb_loss = self.loss_function_mask(mask_verb_pred_logits, gt_verbs)
        out['loss_mask'] = mask_verb_loss

        # All metrics should be calculated per verb and averaged across verbs.
        ## In the dev and test split of SWiG dataset, there are 50 images for each verb (same number of images per verb).
        ### Our implementation is correct to calculate metrics for the dev and test split of SWiG dataset. 
        ### We calculate metrics in this way for simple implementation in distributed data parallel setting. 

        # accuracies (for verb and noun)
        out['verb_acc_top1'] = verb_acc_topk[0]
        out['verb_acc_top5'] = verb_acc_topk[1]
        out['mean_acc'] = torch.stack([v for k, v in out.items() if 'verb_acc' in k]).mean()
        return out
    
class VanillaImSituCriterion(nn.Module):
    """ 
    Loss for GSRTR with SWiG dataset, and GSRTR evaluation.
    """
    def __init__(self, weight_dict):
        """ 
        Create the criterion.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.loss_function_img = LabelSmoothing(0.3)
        self.loss_function_mask = LabelSmoothing(0.3)


    def forward(self, targets, img_outputs, eval=False):
        """ This performs the loss computation, and evaluation of GSRTR.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             eval: boolean, used in evlauation
        """
        # top-1 & top 5 verb acc and calculate verb loss
        out = {}
        verb_acc_topk = [0,0]
        # losses 
        gt_verbs = targets.max(1, keepdim=False)[1]
        img_verb_pred_logits = img_outputs.squeeze(1)
        verb_acc_topk = accuracy(img_verb_pred_logits, gt_verbs, topk=(1, 5))
        img_verb_loss = self.loss_function_img(img_verb_pred_logits, gt_verbs)
        out['loss_img'] = img_verb_loss

        # All metrics should be calculated per verb and averaged across verbs.
        ## In the dev and test split of SWiG dataset, there are 50 images for each verb (same number of images per verb).
        ### Our implementation is correct to calculate metrics for the dev and test split of SWiG dataset. 
        ### We calculate metrics in this way for simple implementation in distributed data parallel setting. 

        # accuracies (for verb and noun)
        out['verb_acc_top1'] = verb_acc_topk[0]
        out['verb_acc_top5'] = verb_acc_topk[1]
        out['mean_acc'] = torch.stack([v for k, v in out.items() if 'verb_acc' in k]).mean()
        return out


def build(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = FAIR(backbone,
                  transformer)
    criterion = None
    
    if args.fair:
        weight_dict = {'loss_img': args.img_loss_coef, 'loss_mask': args.mask_loss_coef}
        criterion = ImSituCriterion(weight_dict=weight_dict)
    else:
        weight_dict = {'loss_img': args.img_loss_coef}
        criterion = VanillaImSituCriterion(weight_dict=weight_dict)

    return model, criterion