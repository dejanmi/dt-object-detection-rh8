import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F


class Dataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.data_files = list(sorted(filter(lambda x: "npz" in x, os.listdir("./data_collection/dataset"))))

    def __getitem__(self, idx):
        # load images ad masks
        data_path = os.path.join(self.root, "dataset", self.data_files[idx])
        data = np.load(data_path)
        img = data[f"arr_{0}"]
        img = Image.fromarray(img, "RGB")
        boxes = data[f"arr_{1}"]
        classes = data[f"arr_{2}"]
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(classes, dtype=torch.int64)

        image_id = torch.tensor([idx])
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        # img = F.to_tensor(img)
        return img, target

    def __len__(self):
        return len(self.data_files)
