from collections import OrderedDict

import numpy as np
import yaml
from model import objectives


from .CrossEmbeddingLayer_local import TexualEmbeddingLayer, TexualEmbeddingLayer1, VisualEmbeddingLayer, VisualEmbeddingLayer1
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights, Transformer, LayerNorm, QuickGELU, \
    Transformer1
import torch
import torch.nn as nn
import torch.nn.functional as F




def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim)
    )

class CFFA(nn.Module):


    def __init__(self, args, num_classes=11003, config=None):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)


        if 'mal' in self.current_task:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)
        if 'mlm' in self.current_task:
            self.vocab_size = 49408
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer1(width=self.embed_dim,
                                                       layers=4,
                                                       heads=self.embed_dim // 64)
            scale = self.cross_modal_transformer.width ** -0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                             ('gelu', QuickGELU()),
                             ('ln', LayerNorm(self.embed_dim)),
                             ('fc', nn.Linear(self.embed_dim, self.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

        if 'mal' in self.current_task:
            loss_type = 'mal'
        elif 'triplet' in self.current_task:
            loss_type = 'triplet'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        else:
            exit()
        self.loss_type = loss_type

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()

    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_local(self, image):
        x,atten_i = self.base_model.encode_image(image)
        i_local = self.visul_emb_layer(x, atten_i)
        return i_local.float()


    def encode_text_local(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_local = self.texual_emb_layer(x, text, atten_t)
        return t_local.float()



    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def forward(self, batch, alpha):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        images = batch['images']


        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_local_feats = self.visul_emb_layer(image_feats, atten_i)

        t_local_feats = self.texual_emb_layer(text_feats, caption_ids, atten_t)
        label_hat = batch['label_hat'].to(i_feats.device)

        loss1 = objectives.compute_mixed(i_feats, t_feats, i_local_feats, t_local_feats, batch['pids'], \
                                         label_hat=label_hat, margin=self.args.margin, tau=self.args.tau, \
                                         loss_type=self.loss_type, logit_scale=self.logit_scale)



        ret.update({'mixed_triplet_loss':loss1})


        # MLM
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats, _ = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, 49408)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            mlm_loss_weight = 1.4
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels) * mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        if 'mal' in self.current_task:
            id_loss_weight = 1.0
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*id_loss_weight})



        return ret


def build_model(args, num_classes=11003, Config = None):
    model = CFFA(args, num_classes, config = Config)
    # covert model to fp16
    convert_weights(model)
    return model
