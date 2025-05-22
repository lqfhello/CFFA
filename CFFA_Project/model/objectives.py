import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.comm import floor_round


def compute_triplet_per(scores, pid, margin=0.2, tau=0.02):
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_1 = ((scores / tau).exp() * labels / ((scores / tau).exp() * labels).sum(dim=1, keepdim=True)).detach()
    alpha_2 = ((scores.t() / tau).exp() * labels / ((scores.t() / tau).exp() * labels).sum(dim=1,
                                                                                           keepdim=True)).detach()

    pos_1 = (alpha_1 * scores).sum(1)
    pos_2 = (alpha_2 * scores.t()).sum(1)

    neg_1 = (mask * scores).max(1)[0]
    neg_2 = (mask * scores.t()).max(1)[0]

    cost_1 = (margin + neg_1 - pos_1).clamp(min=0)
    cost_2 = (margin + neg_2 - pos_2).clamp(min=0)
    return cost_1 + cost_2


def compute_InfoNCE_per(scores, logit_scale):
    # cosine similarity as logits
    logits_per_image = logit_scale * scores
    logits_per_text = logits_per_image.t()

    p1 = F.softmax(logits_per_image, dim=1)
    p2 = F.softmax(logits_per_text, dim=1)

    loss = (- p1.diag().log() - p2.diag().log()) / 2
    return loss


def compute_mixed_triplet_per(scores, pid, tau, margin):
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_i2t = ((scores / tau).exp() * labels / ((scores / tau).exp() * labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t() / tau).exp() * labels / ((scores.t() / tau).exp() * labels).sum(dim=1,
                                                                                             keepdim=True)).detach()

    loss = (-  (alpha_i2t * scores).sum(1) + floor_round(1 / batch_size, 3) * ((scores / tau).exp() * mask).sum(
        1).clamp(max=10e35).log() + margin).clamp(min=0) \
           + (-  (alpha_t2i * scores.t()).sum(1) + floor_round(1 / batch_size, 3) * (
                (scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)

    return loss


def compute_mixed(i_feats, t_feats, i_local_feats, t_local_feats, pid, label_hat=None, tau=0.02, margin=0.1,
                  loss_type='mal', logit_scale=50):
    loss1, _ = compute_per_loss(i_feats, t_feats, pid, tau, margin, loss_type, logit_scale)
    loss2, _ = compute_per_loss(i_local_feats, t_local_feats, pid, tau, margin, loss_type, logit_scale)

    loss1 = (label_hat * loss1).sum()
    loss2 = (label_hat * loss2).sum()

    if loss_type in ['mal', 'triplet']:

        return loss1 + loss2
    else:
        # return loss1/label_hat.sum(), loss2/label_hat.sum() # mean
        return loss1 / label_hat.sum()


def compute_per_loss(image_features, text_features, pid, tau=0.02, margin=0.2, loss_type='mal', logit_scale=50):
    # # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()

    if 'mal' in loss_type:
        per_loss = compute_mixed_triplet_per(scores, pid, tau, margin=margin)
    elif 'triplet' in loss_type:
        per_loss = compute_triplet_per(scores, pid, tau=tau, margin=margin)
    elif 'InfoNCE' in loss_type:
        per_loss = compute_InfoNCE_per(scores, logit_scale)
    else:
        exit()

    return per_loss, scores.diag()


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)

    return loss / 2

