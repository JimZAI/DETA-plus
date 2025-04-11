import heapq

import torch

from models.tools import generate_embeddings


class BBox:
    def __init__(self, box, pos, size, label, image_idx, patch):
        self.boxes = box
        self.pos = pos
        self.size = size

        self.label = label
        self.image_idx = image_idx

        self.weight = None
        self.patch = patch

    def set_weight(self, weight):
        self.weight = weight

    def crop(self, images):
        bbh1, bbh2, bbw1, bbw2 = self.boxes
        x, y = self.pos
        idx = self.image_idx

        region_x = images[idx : idx + 1, :, bbh1:bbh2, bbw1:bbw2]
        region_sub_x = region_x[:, :, x : x + self.size, y : y + self.size]

        return region_sub_x

    def intra_swap(self, images, labels):
        crop_image = self.crop(images)
        bbh1, bbh2, bbw1, bbw2 = self.boxes
        x, y = self.pos

        intra_index = torch.where(labels == self.label)[0]
        images = images[intra_index]
        rand_index = torch.randperm(images.size()[0])[0]
        image = images[rand_index : rand_index + 1].clone()
        # image = images[torch.randperm(images.size()[0])].copy()
        image[
            :,
            :,
            bbh1 + x : bbh1 + x + self.size,
            bbw1 + y : bbw1 + y + self.size,
        ] = crop_image

        return image

    def __repr__(self):
        return (
            f"(image_idx: {self.image_idx}, "
            f"label: {self.label}, pos: {self.pos}, "
            f"size: {self.size}, patch: {self.patch})"
        )

    def __lt__(self, other):
        return self.weight < other.weight
        # return self.weight > other.weight
        # return (self.weight or 0) > (other.weight or 0)


class MemoryBank:
    def __init__(self, num_shots):
        self.top_2k = [x * 2 for x in num_shots]
        self.num_classes = len(num_shots)

        self.bank = {i: [] for i in range(self.num_classes)}

    def store(self, bboxes: list[BBox], clean_region_idxs):
        for i in range(self.num_classes):
            s_boxes = [
                bboxes[idx]
                for idx in clean_region_idxs
                if bboxes[idx].label == i
            ]
            s_weight = torch.tensor([box.weight for box in s_boxes])

            limits = self.top_2k[i]

            top2k_indices = s_weight.argsort(dim=0, descending=True)[:limits]

            # for idx in top2k_indices:
            #     box = s_boxes[idx]
            #     if self._update_bank(box, i, limits):
            #         break
            for idx in top2k_indices:
                box = s_boxes[idx]
                self.bank[i].append(box)
            # Sort bank by weight and keep only top 2k
            if len(self.bank[i]) > limits:
                self.bank[i] = sorted(
                    self.bank[i],
                    key=lambda x: x.weight,
                    reverse=True,
                )[:limits]

    def _get_image_by_type(
        self,
        box,
        images,
        image_type,
        labels=None,
    ):
        if image_type == "crop":
            return box.crop(images)
        # elif image_type == "cutmix":
        #     return box.cutmix(images, labels)
        elif image_type == "intraSwap":
            return box.intra_swap(images, labels)
        else:
            raise ValueError(f"Unknown image region type: {image_type}")

    def bank_features(
        self,
        images,
        model,
        model_name,
        is_weighted=False,
        image_type="crop",
        labels=None,
        topk=None,
    ):
        if topk is None:

            context = {
                i: {
                    "feats": torch.concat(
                        [
                            generate_embeddings(
                                self._get_image_by_type(
                                    box, images, image_type, labels
                                ),
                                model,
                                model_name,
                                need_float=False,
                            )
                            for box in self.bank[i]
                        ],
                        dim=0,
                    ),
                    "weights": torch.tensor(
                        [box.weight for box in self.bank[i]],
                        device=images.device,
                    ),
                }
                for i in self.bank
            }
        else:
            context = {
                i: {
                    "feats": torch.concat(
                        [
                            generate_embeddings(
                                self._get_image_by_type(
                                    box, images, image_type, labels
                                ),
                                model,
                                model_name,
                                need_float=False,
                            )
                            for box in self.bank[i][:topk]
                        ],
                        dim=0,
                    ),
                    "weights": torch.tensor(
                        [box.weight for box in self.bank[i][:topk]],
                        device=images.device,
                    ),
                }
                for i in self.bank
            }
        if is_weighted:
            context_features = torch.concat(
                [
                    context[i]["feats"] * context[i]["weights"].unsqueeze(-1)
                    for i in context
                ],
                dim=0,
            )
        else:
            context_features = torch.concat(
                [context[i]["feats"] for i in context], dim=0
            )
        labels = torch.concat(
            [
                torch.full(
                    (context[i]["feats"].shape[0],),
                    i,
                    device=images.device,
                )
                for i in context
            ],
            dim=0,
        )
        return context_features, labels

    def _update_bank(self, box, i, limits):
        if len(self.bank[i]) < limits:
            self.bank[i].append(box)
            if len(self.bank[i]) == limits:
                heapq.heapify(self.bank[i])
        elif box.weight > self.bank[i][0].weight:
            # Replace the smallest element in the heap
            heapq.heapreplace(self.bank[i], box)
        else:
            return True
        return False
