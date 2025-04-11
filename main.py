"""
This code is based on the implementation of TSA.
Reference: https://github.com/VICO-UoE/URL
"""

import os
import random
from copy import deepcopy

import clip
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.transforms as T
from tabulate import tabulate
from torchvision import transforms
from tqdm import tqdm

from config import args
from data.meta_dataset_reader import MetaDatasetEpisodeReader
from models.losses import prototype_loss

# stronger models
from models.ta import ta
from models.tools import generate_embeddings

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def preserve_key(state, remove_prefix: str):
    """Preserve part of model weights based on the
    prefix of the preserved module name.
    """
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if remove_prefix + "." in key:
            newkey = key.replace(remove_prefix + ".", "")
            state[newkey] = state.pop(key)
        else:
            state.pop(key)
    return state


def norm_clip(imgs):
    transform_img = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([0.48145466, 0.4578275, 0.40821073]),
                np.array([0.26862954, 0.26130258, 0.27577711]),
            ),
        ]
    )
    imgs_list = []
    for img in imgs:
        # img = img.squeeze(0)
        to_pil = T.ToPILImage()
        img = (img / 2 + 0.5) * 255
        img = img.to(torch.uint8)
        img = to_pil(img)
        imgs_list.append(transform_img(img))
    img_tensor = torch.stack(imgs_list).cuda()

    return img_tensor


# ALL_METADATASET_NAMES = "ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower traffic_sign mscoco".split(' ')
# TRAIN_METADATASET_NAMES = ALL_METADATASET_NAMES[:8]
# TEST_METADATASET_NAMES = "ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower traffic_sign mscoco" .split(' ')

ALL_METADATASET_NAMES = "dtd".split(" ")
TRAIN_METADATASET_NAMES = ALL_METADATASET_NAMES
TEST_METADATASET_NAMES = "dtd".split(" ")


def initialize_model(model_name):
    if model_name == "CLIP":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("RN50", device=device)
        return model

    if model_name == "MOCO":
        from models.resnet50 import create_model

        model = create_model()
        state = torch.load("models/moco_v2_800ep_pretrain.pth.tar")[
            "state_dict"
        ]
        state = preserve_key(state, "module.encoder_q")
        state_keys = list(state.keys())
        for i, key in enumerate(state_keys):
            if "fc" in key:
                state.pop(key)
        model.load_state_dict(state)

    elif model_name == "DEIT":
        from deit.models import deit_small_patch16_224

        model = deit_small_patch16_224(pretrained=True)

    elif model_name == "SWIN":
        from swin_transformer.models.swin_transformer import SwinTransformer

        model = SwinTransformer()
        state = torch.load("models/swin_tiny_patch4_window7_224.pth")["model"]
        model.load_state_dict(state)

    model.eval()
    model.cuda()
    return model


def introduce_label_bias(context_images, context_labels, ratio):
    """
    Introduce bias in the context images based on the given ratio.

    Args:
        context_images: Tensor of context images
        context_labels: Tensor of context labels
        ratio: Float indicating the proportion of images to be biased (0 to 1)

    Returns:
        Tensor of modified context images
    """
    if ratio <= 0:
        return context_images

    N_shots = 10
    N_bias = int(N_shots * ratio)
    data = []
    N_way = context_labels[-1] + 1

    for nw in range(N_way):
        # Create list of other class labels
        label_others = [t for t in range(N_way) if t != nw]
        random.shuffle(label_others)
        num_others = len(label_others)

        # Get current class data
        data_nw = context_images[nw * N_shots : (nw + 1) * N_shots]

        # Replace some images with other class images
        for tt in range(N_bias):
            ttt = tt if num_others > tt else tt + 1 - num_others
            other_class_label = label_others[ttt]
            start_others = other_class_label * N_shots
            randindx = random.randint(0, N_shots - 1)
            index_others = start_others + randindx
            data_nw[N_shots - 1 - tt] = context_images[index_others]

        data.append(data_nw)

    return torch.cat(data)


def main():
    testsets = TEST_METADATASET_NAMES
    trainsets = TRAIN_METADATASET_NAMES
    test_loader = MetaDatasetEpisodeReader(
        "test", trainsets, trainsets, testsets, test_type=args["test.type"]
    )

    model_name = args["pretrained_model"]
    ratio = args["ratio"]

    if args["ours"]:
        is_baseline = False
    else:
        is_baseline = True
    K_patch = args["n_regions"]
    max_It = args["maxIt"]
    TEST_SIZE = 1
    # TEST_SIZE = 600

    is_weight_patch = True
    is_weight_sample = True
    if model_name == "CLIP":
        lr_finetune = 0.001
    elif model_name == "MOCO":
        lr_finetune = 0.001
    elif model_name == "DEIT":
        lr_finetune = 0.1
    elif model_name == "SWIN":
        lr_finetune = 0.05
    else:
        lr_finetune = 0.0

    model = initialize_model(model_name)

    accs_names = ["NCC"]
    var_accs = dict()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        for dataset in testsets:
            var_accs[dataset] = {name: [] for name in accs_names}

            if is_baseline:
                is_weight_patch = False
                is_weight_sample = False
            for i in tqdm(range(TEST_SIZE)):
                model.zero_grad()
                torch.cuda.empty_cache()
                sample = test_loader.get_test_task(session, dataset)
                context_labels = sample["context_labels"]
                target_labels = sample["target_labels"]
                context_images = sample["context_images"]
                target_images = sample["target_images"]

                context_images = introduce_label_bias(
                    context_images, context_labels, ratio
                )

                cur_model = deepcopy(model)
                if (model_name == "DEIT" and context_labels.shape[0] > 90) or (
                    model_name == "SWIN" and context_labels.shape[0] > 70
                ):  # for saving time/memory
                    K_patch = 1

                sample_weight, memory_bank = ta(
                    context_images,
                    context_labels,
                    cur_model,
                    model_name=model_name,
                    max_iter=max_It,
                    lr_finetune=lr_finetune,
                    distance=args["test.distance"],
                    is_baseline=is_baseline,
                    is_weight_patch=is_weight_patch,
                    is_weight_sample=is_weight_sample,
                    K_patch=K_patch,
                    dataset=dataset,
                )
                
                # Local NCC
                with torch.no_grad():

                    context_features, context_labels = (
                        memory_bank.bank_features(
                            context_images,
                            cur_model,
                            model_name,
                            is_weight_sample,
                        )
                    )
                    target_features = generate_embeddings(
                        target_images, cur_model, model_name, need_float=False
                    )

                _, (_, stats_dict, _) = prototype_loss(
                    context_features,
                    context_labels,
                    target_features,
                    target_labels,
                    patch_weight=None,
                    distance=args["test.distance"],
                )
                var_accs[dataset]["NCC"].append(stats_dict["acc"])

            dataset_acc = np.array(var_accs[dataset]["NCC"]) * 100
            print(f"{dataset}: test_acc {dataset_acc.mean():.2f}%")

    # Print nice results table
    print(f"results of {args['model.name']}")
    rows = []
    sum_all = 0.0
    for dataset_name in testsets:
        row = [dataset_name]
        for model_name in accs_names:
            acc = np.array(var_accs[dataset_name][model_name]) * 100
            mean_acc = acc.mean()
            sum_all = sum_all + mean_acc
            conf = (1.96 * acc.std()) / np.sqrt(len(acc))
            row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
        rows.append(row)
    table = tabulate(
        rows, headers=["model \\ data"] + accs_names, floatfmt=".2f"
    )
    print(table)
    avg = sum_all / 10.0
    print(avg)


def seed_everything(seed=47):
    """Set random seed for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.compat.v1.set_random_seed(seed)


if __name__ == "__main__":
    seed_everything()
    main()
