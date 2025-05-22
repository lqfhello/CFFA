import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    """https://github.com/woodfrog/vse_infty, thanks!"""
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN) from https://github.com/woodfrog/vse_infty, thanks!"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B * N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x


class TexualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=1024, ratio=0.3):
        super(TexualEmbeddingLayer, self).__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.ratio = ratio

    def forward(self, features, text, atten):
        # print(atten) 64 x 77 x 77
        mask = ((text != 0) + 0)
        lengths = mask.sum(1).view(-1) - 2  # -2 for SOS token and EOS token
        k = int((atten.size(1) - 2) * self.ratio)
        bs = features.size(0)
        atten[torch.arange(bs), :, text.argmax(dim=-1)] = -1  # last token
        atten[torch.arange(bs), :, 0] = -1  # first token
        atten = atten[torch.arange(bs), text.argmax(dim=-1), :]  # 64 x 77
        atten = atten * mask

        atten_topK = atten.topk(dim=-1, k=k)[1].unsqueeze(-1).expand(bs, k, features.size(2))  # 64 x k x 512
        features = torch.gather(input=features, dim=1, index=atten_topK)  # 64 x k x 512
        features = l2norm(features, dim=-1)

        lengths = torch.Tensor([lengths[i] if lengths[i] < k else k for i in range(bs)])  # Keep at least K

        cap_emb = self.linear(features.half())
        features = self.mlp(features) + cap_emb
        features = maxk_pool1d_var(features, 1, 1, lengths.to(cap_emb.device))  # max

        return features.float()


class TexualEmbeddingLayer1(nn.Module):
    def __init__(self, input_dim=512, embed_dim=1024, ratio=0.3):
        super(TexualEmbeddingLayer1, self).__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.ratio = ratio

    def account_len(self, np_a):
        lens = 0
        for w in range(np_a.shape[0]):
            if (not np_a[w] == 0):
                lens += 1
            elif (np_a[w] == 0):
                break
        if lens >= 77:
            lens = 76
        return lens

    def forward(self, all_word_embeddings, caption_ids, attention_map):
        token_len_list = []
        np_tokens = np.array(caption_ids.cpu())
        for w in range(np_tokens.shape[0]):
            token_lens = self.account_len(np_tokens[w])
            token_len_list.append(token_lens)
        # attention_map = attention_map.mean(axis=1)
        bs = attention_map.size(0)
        selected_word_embedding = []
        attention_weights_list = []
        selected_indices_list = []
        for i in range(bs):
            attention_cls_part = attention_map[i, token_len_list[i]]
            attention_cls_part = attention_cls_part[1:token_len_list[i]]
            len_attention_cls_part = attention_cls_part.shape[0]

            all_word_embeddings_i = all_word_embeddings[i, :, :].squeeze()
            all_word_embeddings_i = all_word_embeddings_i[1:token_len_list[i], :]

            sort1, indices = torch.sort(attention_cls_part, descending=True)
            max_words = 25
            if (len_attention_cls_part > max_words):
                selected_indices = indices[: max_words]
                attention_cls_part = torch.index_select(attention_cls_part, 0, selected_indices)
                all_word_embeddings_i = torch.index_select(all_word_embeddings_i, 0, selected_indices)

            elif (len_attention_cls_part < max_words):
                yhi = max_words - len_attention_cls_part
                device = "cuda"
                add_yhi = torch.zeros(yhi).to(device)
                attention_cls_part = torch.cat((attention_cls_part, add_yhi), dim=0)

            sort2, indices2 = torch.sort(attention_cls_part, descending=True)
            Rt = 0.4
            select_token = round(Rt * max_words)
            selected_indices2 = indices2[:select_token]

            attention_weights_list.append(attention_cls_part.unsqueeze(0))
            selected_indices_list.append(selected_indices2.unsqueeze(0))

            top_k_embedding = torch.index_select(all_word_embeddings_i, 0, selected_indices2)
            top_k_embedding = top_k_embedding.unsqueeze(0)
            selected_word_embedding.append(top_k_embedding)

        attention_weights_last = torch.cat(attention_weights_list, 0)
        selected_indices_last = torch.cat(selected_indices_list, 0)
        selected_word_embedding = torch.cat(selected_word_embedding, 0)

        features = l2norm(selected_word_embedding, dim=-1)
        mask = ((caption_ids != 0) + 0)
        lengths = mask.sum(1).view(-1) - 2
        k = int((attention_map.size(1) - 2) * Rt)
        lengths = torch.Tensor([lengths[i] if lengths[i] < k else k for i in range(bs)])  # Keep at least K

        cap_emb = self.linear(features.half())
        features = self.mlp(features) + cap_emb
        features = maxk_pool1d_var(features, 1, 1, lengths.to(cap_emb.device))  # max

        # return selected_word_embedding, attention_weights_last, selected_indices_last
        return features.float()


class VisualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=1024, ratio=0.3):
        super(VisualEmbeddingLayer, self).__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.ratio = ratio
        self.fc = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)

    def forward(self, base_features, atten):
        k = int((atten.size(1) - 1) * self.ratio)  # 192

        bs = base_features.size(0)
        atten[torch.arange(bs), :, 0] = -1  # CLS token
        atten_topK = atten[:, 0].topk(dim=-1, k=k)[1]

        atten_topK = atten_topK.unsqueeze(-1).expand(bs, k, base_features.size(2))  # 64 x k x 512
        base_features = torch.gather(input=base_features, dim=1, index=atten_topK)  # 64 x k x 512
        base_features = l2norm(base_features, dim=-1)
        base_features = base_features.half()
        feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device).half()
        feat_lengths[:] = base_features.size(1)

        features = self.fc(base_features)
        features = self.mlp(base_features) + features
        features = maxk_pool1d_var(features, 1, 1, feat_lengths)  # max

        return features.float()


class VisualEmbeddingLayer1(nn.Module):
    def __init__(self, input_dim=512, embed_dim=1024, ratio=0.3):
        super(VisualEmbeddingLayer1, self).__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.ratio = ratio
        self.fc = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)

    def forward(self, all_patch_embeddings, attention_map):
        # attention_map = attention_map.mean(axis=1)

        attention_cls_part = attention_map[:, 0, 1:]
        attention_cls_part = attention_cls_part.squeeze()

        sorted, indices = torch.sort(attention_cls_part, descending=True)
        Rv = 0.3
        select_token = round(Rv * 192)
        selected_indices = indices[:, 0:select_token]
        selected_patch_embedding = []
        for i in range(selected_indices.size(0)):  # bs
            all_patch_embeddings_i = all_patch_embeddings[i, :, :].squeeze()
            top_k_embedding = torch.index_select(all_patch_embeddings_i, 0, selected_indices[i])
            top_k_embedding = top_k_embedding.unsqueeze(0)
            selected_patch_embedding.append(top_k_embedding)
        selected_patch_embedding = torch.cat(selected_patch_embedding, 0)

        base_features = l2norm(selected_patch_embedding, dim=-1)
        base_features = base_features.half()
        feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device).half()
        feat_lengths[:] = base_features.size(1)

        features = self.fc(base_features)
        features = self.mlp(base_features) + features
        features = maxk_pool1d_var(features, 1, 1, feat_lengths)  # max

        # return selected_patch_embedding, selected_indices
        return features.float()