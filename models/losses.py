import gin
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from torch import nn

from models.cka import kernel_CKA, linear_CKA


def cross_entropy_loss(logits, targets):
    log_p_y = F.log_softmax(logits, dim=1)
    preds = log_p_y.argmax(1)
    labels = targets.type(torch.long)
    loss = F.nll_loss(log_p_y, labels, reduction="mean")
    acc = torch.eq(preds, labels).float().mean()
    stats_dict = {"loss": loss.item(), "acc": acc.item()}
    pred_dict = {"preds": preds.cpu().numpy(), "labels": labels.cpu().numpy()}
    return loss, stats_dict, pred_dict


# clean_loss or LocalNCC
def clean_loss(
    support_embeddings,
    support_labels,
    query_embeddings,
    query_labels,
    patch_weight=None,
    distance="cos",
):
    n_way = len(query_labels.unique())
    prots = compute_prototypes(
        support_embeddings, support_labels, n_way
    ).unsqueeze(0)
    embeds = query_embeddings.unsqueeze(1)

    if distance == "l2":
        logits = -torch.pow(embeds - prots, 2).sum(
            -1
        )  # shape [n_query, n_way]
    elif distance == "cos":
        logits = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * 10
    elif distance == "lin":
        logits = torch.einsum("izd,zjd->ij", embeds, prots)
    elif distance == "corr":
        logits = F.normalize((embeds * prots).sum(-1), dim=-1, p=2) * 10
    else:
        logits = None

    if patch_weight is not None:
        logits = patch_weight * logits
        return prots, cross_entropy_loss(logits, query_labels)
    else:
        return prots, cross_entropy_loss(logits, query_labels)


def noise_loss(
    prots,
    query_embeddings,
    distance="cos",
):
    embeds = query_embeddings.unsqueeze(1)

    if distance == "l2":
        logits = -torch.pow(embeds - prots, 2).sum(
            -1
        )  # shape [n_query, n_way]
    elif distance == "cos":
        logits = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * 10
    elif distance == "lin":
        logits = torch.einsum("izd,zjd->ij", embeds, prots)
    elif distance == "corr":
        logits = F.normalize((embeds * prots).sum(-1), dim=-1, p=2) * 10
    else:
        logits = None

    logits = F.softmax(logits, dim=-1)
    entropy = -torch.sum(logits * torch.log(logits + 1e-30), dim=-1)
    loss = torch.mean(entropy)
    return loss


def compute_prototypes(embeddings, labels, n_way):
    prots = (
        torch.zeros(n_way, embeddings.shape[-1])
        .type(embeddings.dtype)
        .to(embeddings.device)
    )
    for i in range(n_way):
        if torch.__version__.startswith("1.1"):
            prots[i] = embeddings[(labels == i).nonzero(), :].mean(0)
        else:
            prots[i] = embeddings[
                (labels == i).nonzero(as_tuple=False), :
            ].mean(0)
    return prots


# knn
def knn_loss(
    support_embeddings,
    support_labels,
    query_embeddings,
    query_labels,
    distance="cos",
):
    n_way = len(query_labels.unique())

    prots = support_embeddings
    embeds = query_embeddings.unsqueeze(1)

    if distance == "l2":
        dist = -torch.pow(embeds - prots, 2).sum(-1)  # shape [n_query, n_way]
    elif distance == "cos":
        dist = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * 10
    elif distance == "lin":
        dist = torch.einsum("izd,zjd->ij", embeds, prots)
    _, inds = torch.topk(dist, k=1)

    logits = (
        torch.zeros(embeds.size(0), n_way)
        .to(embeds.device)
        .scatter_(1, support_labels[inds.flatten()].view(-1, 1), 1)
        * 10
    )

    return cross_entropy_loss(logits, query_labels)


class DistillKL(nn.Module):
    """KL divergence for distillation"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction="sum") * (self.T**2) / y_s.shape[0]
        return loss


class AdaptiveCosineNCC(nn.Module):
    def __init__(self):
        super(AdaptiveCosineNCC, self).__init__()
        self.scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)

    def forward(
        self,
        support_embeddings,
        support_labels,
        query_embeddings,
        query_labels,
        return_logits=False,
    ):
        n_way = len(query_labels.unique())

        prots = compute_prototypes(
            support_embeddings, support_labels, n_way
        ).unsqueeze(0)
        embeds = query_embeddings.unsqueeze(1)
        logits = (
            F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * self.scale
        )

        if return_logits:
            return logits

        return cross_entropy_loss(logits, query_labels)


def cal_dist(inputs, inputs_center):
    n = inputs.size(0)
    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs, inputs.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    # Compute pairwise distance, replace by the official when merged
    dist_center = (
        torch.pow(inputs_center, 2).sum(dim=1, keepdim=True).expand(n, n)
    )
    dist_center = dist_center + dist_center.t()
    dist_center.addmm_(1, -2, inputs_center, inputs_center.t())
    dist_center = dist_center.clamp(
        min=1e-12
    ).sqrt()  # for numerical stability
    loss = torch.mean(torch.norm(dist - dist_center, p=2))
    return loss


def cal_dist_cosine(inputs, inputs_center):
    # We would like to perform cosine similarity for pairwise distance
    n = inputs.size(0)
    # Compute pairwise distance, replace by the official when merged
    dist = 1 - F.cosine_similarity(
        inputs.unsqueeze(1), inputs, dim=-1, eps=1e-30
    )
    dist = dist.clamp(min=1e-12)

    # Compute pairwise distance, replace by the official when merged
    dist_center = 1 - F.cosine_similarity(
        inputs_center.unsqueeze(1), inputs_center, dim=-1, eps=1e-30
    )
    dist_center = dist_center.clamp(min=1e-12)
    loss = torch.mean(torch.norm(dist - dist_center, p=2))
    return loss


def distillation_loss(fs, ft, opt="l2", delta=0.5):
    if opt == "l2":
        return (fs - ft).pow(2).sum(1).mean()
    if opt == "l1":
        return (fs - ft).abs().sum(1).mean()
    if opt == "huber":
        l1 = (fs - ft).abs()
        binary_mask_l1 = (
            (l1.sum(1) > delta).type(torch.FloatTensor).unsqueeze(1).cuda()
        )
        binary_mask_l2 = (
            (l1.sum(1) <= delta).type(torch.FloatTensor).unsqueeze(1).cuda()
        )
        loss = (
            (l1.pow(2) * binary_mask_l2 * 0.5).sum(1)
            + (l1 * binary_mask_l1).sum(1) * delta
            - delta**2 * 0.5
        )
        loss = loss.mean()
        return loss
    if opt == "rkd":
        return cal_dist(fs, ft)
    if opt == "cosine":
        return 1 - F.cosine_similarity(fs, ft, dim=-1, eps=1e-30).mean()
    if opt == "rkdcos":
        return cal_dist_cosine(fs, ft)
    if opt == "linearcka":
        return 1 - linear_CKA(fs, ft)
    if opt == "kernelcka":
        return 1 - kernel_CKA(fs, ft)
