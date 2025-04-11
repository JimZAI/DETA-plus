import torch.nn as nn


def generate_embeddings(image, model, model_name, need_float=True):
    if model_name == "CLIP":
        q_emb = model.encode_image(image)
    else:
        q_emb = model(image)
    if need_float:
        q_emb = q_emb.float()

    if len(q_emb.shape) == 4:
        avg_pool = nn.AvgPool2d(q_emb.shape[-2:])
        q_emb = avg_pool(q_emb).squeeze(-1).squeeze(-1)
    return q_emb
