import math
import sys
import time
import torch
import torchvision.models.detection.mask_rcnn
from torchvision.transforms.functional import to_tensor
import utils_engine
import numpy as np
from eval.map_sol import mean_average_precision
import os

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils_engine.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils_engine.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils_engine.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils_engine.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return metric_logger

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def evaluate(model, device):
    dataset_files = list(filter(lambda x: "npz" in x, os.listdir("./data_collection/dataset_test")))
    dataset_files = dataset_files[:50]
    true_boxes = []
    pred_boxes = []
    def make_boxes(id, labels, scores, bboxes):
        temp = []
        for i in range(len(labels)):
            x1 = bboxes[i][0]
            y1 = bboxes[i][1]
            x2 = bboxes[i][2] - x1
            y2 = bboxes[i][3] - y1
            temp.append([id, labels[i], scores[i], x1, y1, x2, y2])
        return temp
    BATCH_SIZE = 2
    BATCH_QTTY = int(len(dataset_files) / BATCH_SIZE)
    def make_batches(list):
        for i in range(0, len(list), BATCH_SIZE):
            yield list[i:i + BATCH_SIZE]
    from tqdm import trange
    batches = list(make_batches(dataset_files[:BATCH_QTTY * BATCH_SIZE]))
    for nb_batch in trange(len(batches)):
        batch = batches[nb_batch]
        for nb_img, file in enumerate(batch):
            with np.load(f'./data_collection/dataset_test/{file}') as data:
                img, boxes, classes = tuple([data[f"arr_{i}"] for i in range(3)])
                batch_or_image = np.array([img])
                if len(batch_or_image.shape) == 3:
                    batch = [batch_or_image]
                else:
                    batch = batch_or_image
                with torch.no_grad():
                    preds = model([to_tensor(img).to(device=device, dtype=torch.float) for img in batch])
                p_boxes = []
                p_classes = []
                p_scores = []
                for pred in preds:
                    p_boxes.append(pred["boxes"].cpu().numpy())
                    p_classes.append(pred["labels"].cpu().numpy())
                    p_scores.append(pred["scores"].cpu().numpy())
                for j in range(len(p_boxes)):
                    pred_boxes += make_boxes(nb_batch + nb_img, p_classes[j], p_scores[j], p_boxes[j])
                true_boxes += make_boxes(nb_batch + nb_img, classes, [1.0] * len(classes), boxes)
    true_boxes = np.array(true_boxes, dtype=float)
    pred_boxes = np.array(pred_boxes, dtype=float)
    # print(mean_average_precision(pred_boxes, true_boxes, box_format="midpoint", num_classes=5))
    print(mean_average_precision(pred_boxes, true_boxes, box_format="midpoint", num_classes=5).item())