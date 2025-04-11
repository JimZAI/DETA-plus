import random

import torch
import torch.nn as nn
from torchvision import transforms

from models.cora import CoRA
from models.losses import prototype_entropy_loss, prototype_loss
from models.memory_bank import BBox, MemoryBank
from models.tools import generate_embeddings


class projector(nn.Module):
    def __init__(self, z_dim=128, feat_dim=512, type="mlp"):
        super(projector, self).__init__()
        if type == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, 512), nn.ReLU(), nn.Linear(512, z_dim)
            )
        else:
            self.proj = nn.Linear(feat_dim, z_dim)

    def forward(self, x):
        x = self.proj(x)
        return x


def resize_transform_tensor(images, size=84):
    transform_ = transforms.Compose([transforms.Resize(size)])
    images_list = []
    [images_list.append(transform_(img)) for img in images]
    return torch.stack(images_list).cuda()


def gen_patches_tensor(
    context_images, shot_nums, size=84, overlap=0.3, K_patch=5
):
    _, _, h, w = context_images.shape
    way = len(shot_nums)
    ch = int(overlap * h)
    cw = int(overlap * w)
    patches = []

    bboxes = []
    max_x0y0 = h // 2 + ch - size - 1
    for _ in range(K_patch):
        bbh1, bbh2 = (h // 4 - ch // 2, 3 * h // 4 + ch // 2)
        bbw1, bbw2 = (w // 4 - cw // 2, 3 * w // 4 + cw // 2)

        patch_x = context_images[:, :, bbh1:bbh2, bbw1:bbw2]

        start_x = random.randint(0, max_x0y0)
        start_y = random.randint(0, max_x0y0)

        patch_xx = patch_x[
            :, :, start_x : start_x + size, start_y : start_y + size
        ]
        patches.append(patch_xx)
        bboxes.append([(bbh1, bbh2, bbw1, bbw2), (start_x, start_y), size])

    point = 0
    patches_img = []
    patch_boxes = []
    for w in range(way):
        patches_class = [
            patches[p][point : point + shot_nums[w]] for p in range(K_patch)
        ]
        point = point + shot_nums[w]
        for s in range(shot_nums[w]):
            for i, pt in enumerate(patches_class):
                patches_img.append(pt[s])
                box, pos, size = bboxes[i]
                # set idx to patches pos
                idx = point + s - shot_nums[w]
                # idx = len(patches_img) - 1
                patch_boxes.append(
                    BBox(
                        box=box,
                        pos=pos,
                        size=size,
                        label=w,
                        # image_idx=s + point - shot_nums[w],
                        image_idx=idx,
                        patch=i,
                    )
                )

    images_gather = torch.stack(patches_img, dim=0)
    return images_gather, patch_boxes


def prepare_labels(context_images, context_labels, K_patch):
    _, counts = torch.unique(context_labels, return_counts=True)

    shot_nums = counts.tolist()
    shot_nums_sum = counts.sum().item()

    labels_all = []
    for i, count in enumerate(shot_nums):
        labels_all.extend([i] * (count * K_patch))

    label_clu_way = torch.tensor(labels_all, dtype=torch.long).cuda()
    return shot_nums, shot_nums_sum, label_clu_way


def ta(
    context_images,
    context_labels,
    model,
    model_name="MOCO",
    max_iter=40,
    lr_finetune=0.001,
    distance="cos",
    is_baseline=False,
    is_weight_patch=False,
    is_weight_sample=False,
    K_patch=5,
    dataset="",
):

    model.eval()
    lr = lr_finetune
    if model_name == "CLIP":
        feat = model.encode_image(context_images[0].unsqueeze(0))
    else:
        feat = model(context_images[0].unsqueeze(0))
    proj = projector(feat_dim=feat.shape[1]).cuda()

    params = []
    backbone_params = [v for k, v in model.named_parameters()]
    params.append({"params": backbone_params})
    proj_params = [v for k, v in proj.named_parameters()]
    params.append({"params": proj_params})

    optimizer = torch.optim.Adadelta(params, lr=lr)

    shot_nums, shot_nums_sum, label_clu_way = prepare_labels(
        context_images, context_labels, K_patch
    )

    balance = 0.3
    START_WEIGHT = 0
    lamb = 0.7
    size_list = [112]
    sample_weight = None
    memory_bank = MemoryBank(shot_nums)
    VARRHO = 0.3
    TOPK = 2
    EPS = 1e-6
    augmented_labels = None

    print(f"VARRHO: {VARRHO}, TOPK: {TOPK}, EPS: {EPS}")

    if is_baseline:
        START_WEIGHT = 10086
    for iter in range(max_iter):
        # start = time.time()

        optimizer.zero_grad()
        model.zero_grad()

        """ For images """
        context_features = generate_embeddings(
            context_images,
            model,
            model_name,
            need_float=False,
        )

        """ For patches """
        if iter >= START_WEIGHT:
            size = random.choice(size_list)
            images_gather, bboxes = gen_patches_tensor(
                context_images,
                shot_nums,
                size=size,
                overlap=0.3,
                K_patch=K_patch,
            )

            if model_name in ["CLIP", "DEIT", "SWIN"]:
                images_gather = resize_transform_tensor(
                    images_gather, size=224
                )

            q_emb = generate_embeddings(
                images_gather.cuda(), model, model_name
            )
            q = proj(q_emb)
            q_norm = nn.functional.normalize(q, dim=1)

            cora_module = CoRA(temperature=0.07)
            _, patch_weight = cora_module(
                q_norm,
                label_clu_way,
                shot_nums=shot_nums,
                is_weight_patch=is_weight_patch,
                q_emb=q_emb,
                K_patch=K_patch,
            )

            # 0. update weight
            for i, bbox in enumerate(bboxes):
                bbox.set_weight(patch_weight[i])

            # 1. filter clean
            clean_region_idxs, _ = torch.where(patch_weight >= VARRHO)
            noise_region_idxs, _ = torch.where(patch_weight < VARRHO)

            # 2. update bank
            memory_bank.store(bboxes, clean_region_idxs)

            # compute sample weight of the current iter
            if is_weight_sample:
                patch_w = patch_weight.squeeze(-1)

                sample_weight_i = torch.stack(
                    [
                        patch_w[s * K_patch : (s + 1) * K_patch].mean()
                        for s in range(shot_nums_sum)
                    ]
                )

                sample_weight = (
                    (lamb * sample_weight + (1 - lamb) * sample_weight_i)
                    if iter != START_WEIGHT
                    else sample_weight_i
                )
                sample_weight = torch.where(
                    sample_weight >= VARRHO,
                    sample_weight,
                    torch.tensor(0.0, device=sample_weight.device),
                )

                context_features = (
                    sample_weight.unsqueeze(-1) * context_features
                )

            context_labels_for_iter = context_labels.clone()
            if iter > 0:
                augmented_features, augmented_labels = (
                    memory_bank.bank_features(
                        context_images,
                        model,
                        model_name,
                        image_type="intraSwap",
                        labels=context_labels,
                        topk=TOPK,  # avoid OOM
                    )
                )
                context_features = torch.cat(
                    [context_features, augmented_features], dim=0
                )
                context_labels_for_iter = torch.cat(
                    [context_labels_for_iter, augmented_labels], dim=0
                )

            patch_weight = torch.where(
                patch_weight >= VARRHO,
                patch_weight,
                torch.tensor(0.0, device=patch_weight.device),
            )
            prots, (loss_clean, stat, _) = prototype_loss(
                context_features,
                context_labels_for_iter,
                q_emb,
                label_clu_way,
                patch_weight=patch_weight,
                distance=distance,
            )
            loss_noise = prototype_entropy_loss(
                prots,
                q_emb[noise_region_idxs],
                distance=distance,
            )

            loss = balance * loss_noise + loss_clean
        else:
            loss, stat, _ = prototype_loss(
                context_features,
                context_labels,
                context_features,
                context_labels,
                patch_weight=None,
                distance=distance,
            )

        loss.backward()
        optimizer.step()

        # end = time.time()
        # print(end-start)

    return sample_weight, memory_bank
