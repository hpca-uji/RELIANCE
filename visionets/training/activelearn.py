# General imports
import os
import argparse
import numpy as np
import time
import json
from statistics import mean

import torch
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import AUROC, AveragePrecision, MetricCollection
from timm.layers.norm_act import convert_sync_batchnorm

from visionapi.models import load_model, freeze_extractor
from visionapi.losses import CommonLoss
from visionapi.data import SimpleDataAugmentation, custom_rand_augment_transform
from visionapi.data import setup_simple_dataloader, setup_alpha_dataloader
from visionapi.opts import setup_opt, setup_frozenopt


def load_args():
    """Parameter loading"""
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str)
    parser.add_argument("ori_label_path", type=str)
    parser.add_argument("dst_label_path", type=str)
    parser.add_argument("labelnames_path", type=str)
    parser.add_argument("model_path", type=str)

    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--num_workers", "-j", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--prev_alpha", type=float, default=0.0)
    parser.add_argument("--freeze_extraction", type=int, choices=[0, 1], default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        "--opt",
        type=str,
        default="adam",
        choices=["adam", "adamw", "nadam", "radam", "adamax"],
    )

    parser.add_argument("--tradaug", type=float, nargs=4, default=None)
    parser.add_argument("--randaug", type=str, nargs=3, default=["5", "5", "0.5"])

    parser.add_argument("--name", type=str, default="train")
    parser.add_argument("--num_label", type=int, default=32)

    args = parser.parse_args()

    # Return parse object
    return args


def setup_gpu():
    """Multi-GPU setup"""
    torch.distributed.init_process_group(backend="nccl")
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    return device


def setup_model(args, device):
    """Create and set up model"""
    # Load model into cpu
    model = load_model(args.model_path)
    # Convert BN to SyncBN using timm function
    model = convert_sync_batchnorm(model)

    # Move to specific GPU
    model = model.to(device)
    model.set_hierarchy_device(device)

    # Set up multi-gpu
    if args.freeze_extraction:
        model = DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True
        )
    else:
        model = DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=False
        )

    # Add Metrics
    auroc = AUROC(
        num_labels=args.num_label,
        task="multilabel",
        validate_args=False,
        thresholds=200,
        average=None,
    ).to(device)
    AP = AveragePrecision(
        num_labels=args.num_label,
        task="multilabel",
        validate_args=False,
        thresholds=200,
        average=None,
    ).to(device)
    multimetrics = MetricCollection({"auc": auroc, "mAP": AP})
    model.metric = multimetrics

    return model


def train_epoch(epoch, device, data, sampler, model, opt, loss_fn):
    """Function that trains the model on the given data por one epoch"""
    # Print epoch
    if device == 0:
        print(f"Epoch {epoch+1}")

    # Add shuffle
    sampler.set_epoch(epoch)

    # Activate train mode
    model.train()

    # Initialize train loss
    train_epoch_loss = 0

    for i, (inputs, precompute, label) in enumerate(data):

        # Load to GPU
        inputs = inputs.to(device)
        precompute = precompute.to(device)
        label_loss = label.float().to(device)

        # Reset grad
        opt.zero_grad()

        # Augment and compute predictions
        logits = model(inputs)

        # Compute loss distributed
        loss = loss_fn(logits, precompute, label_loss)
        # Optimize step
        loss.backward()
        opt.step()

        # Agg metrics
        train_epoch_loss += loss.item()

        # Print progress
        if device == 0 and i % 100 == 0:
            print(f"{i:04d} / {len(data)}: {train_epoch_loss/(i+1):.4f}", flush=True)

    # Print final epoch stats
    if device == 0:
        print(f"Epoch {epoch+1} --> train_loss: {train_epoch_loss/len(data):.4f}")


def validation_epoch(epoch, device, data, model, reduce_lr):
    """Function that validates the model on the given data por one epoch"""
    # Activate eval mode
    model.eval()

    with torch.no_grad():  # No grad to avoid overflow
        for i, (inputs, precompute, label) in enumerate(data):

            # Load to GPU
            inputs = inputs.to(device)

            # Forward inputs
            preds = model(inputs)
            model.metric.update(preds, label.int().to(device))

    # Compute metric and reset
    metric = model.metric.compute()
    metricmean = metric["mAP"]
    metricmean = metricmean[metricmean != 0].mean()
    model.metric.reset()

    if epoch != 0:  # Call reduce_lr
        reduce_lr.step(metricmean)

    return metric


def metricprinter(labelnames, padchest_metric, chestx_metric):
    """Function that prints auc and mAP metrics over two datasets and over all label"""
    # Split metric into auc and map
    padchest_auc = padchest_metric["auc"]
    padchest_map = padchest_metric["mAP"]
    chestx_auc = chestx_metric["auc"]
    chestx_map = chestx_metric["mAP"]

    # Foreach label
    title = "Summary"
    print(f"{title:30s} \t PadAUC \t CheAUC \t PadmAP \t ChemAP")
    for i in range(len(labelnames)):
        print(
            f"{labelnames[i]:30s} \t {round(padchest_auc[i].item(),4):.4f} \t {round(chestx_auc[i].item(),4):.4f} \t {round(padchest_map[i].item(),4):.4f} \t {round(chestx_map[i].item(),4):.4f}"
        )

    # Mean computation
    padchest_auc_mean = round(padchest_auc.mean().item(), 4)
    chestx_auc_mean = round(chestx_auc[chestx_auc != 0].mean().item(), 4)
    padchest_map_mean = round(padchest_map.mean().item(), 4)
    chestx_map_mean = round(chestx_map[chestx_map != 0].mean().item(), 4)
    print(
        f"{title:30s} \t {padchest_auc_mean:.4f} \t {chestx_auc_mean:.4f} \t {padchest_map_mean:.4f} \t {chestx_map_mean:.4f}"
    )


def main():

    # Initialization
    args = load_args()
    device = setup_gpu()
    model = setup_model(args=args, device=device)

    if args.randaug is not None:
        magnitude = int(args.randaug[0])
        num_layers = int(args.randaug[1])
        mstd = float(args.randaug[2])
        datransforms = custom_rand_augment_transform(
            magnitude=magnitude, num_layers=num_layers, mstd=mstd
        )
    elif args.tradaug is not None:
        datransforms = SimpleDataAugmentation(
            degrees=args.tradaug[0],
            translate=args.tradaug[1],
            scale=args.tradaug[2],
            shear=args.tradaug[3],
        )
    else:
        datransforms = None

    # Check alpha value for train dataloaders
    if args.prev_alpha > 0:
        # Load datasets to pass to alpha dataloader each epoch
        ori_paths = np.load(
            os.path.join(args.ori_label_path, "tr_paths.npy"), allow_pickle=True
        )
        with open(os.path.join(args.ori_label_path, "tr_label.pt"), "rb") as f:
            ori_label = torch.load(f)
        # Load dst dataset
        dst_paths = np.load(
            os.path.join(args.dst_label_path, "tr_paths.npy"), allow_pickle=True
        )
        with open(os.path.join(args.dst_label_path, "tr_label.pt"), "rb") as f:
            dst_label = torch.load(f)
    else:  # Create dataloader only the first time
        tr_data, tr_sampler = setup_simple_dataloader(
            img_path=args.img_path,
            label_path=args.dst_label_path,
            idf_list=["tr"],
            multigpu=1,
            datransforms=datransforms,
            batch_size=args.batch_size,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers
        )

    # Create validation dataloaders
    ori_vl_data, _ = setup_simple_dataloader(
        img_path=args.img_path,
        label_path=args.ori_label_path,
        idf_list=["vl", "ts"],
        multigpu=1,
        datransforms=None,
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers
    )
    dst_vl_data, _ = setup_simple_dataloader(
        img_path=args.img_path,
        label_path=args.dst_label_path,
        idf_list=["vl", "ts"],
        multigpu=1,
        datransforms=None,
        batch_size=args.batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers
    )

    # Read labelnames
    with open(args.labelnames_path) as f:
        labelnames = json.load(f)

    # Set up frozen opt
    if args.freeze_extraction:
        freeze_extractor(model)
        opt, reduce_lr = setup_frozenopt(
            opt_value=args.opt, pre="resize", lr=args.lr, model=model, device=device
        )
    else:
        opt, reduce_lr = setup_opt(
            opt_value=args.opt, pre="resize", lr=args.lr, model=model, device=device
        )

    # Set up loss fn
    loss_fn = CommonLoss(pw=[None, None], distillation=0, hierarchy="none", loss="bce")
    times = []
    # Normal training
    for epoch in range(args.epochs):

        # Validation before trainining
        padchest_metric = validation_epoch(
            epoch=epoch,
            device=device,
            data=ori_vl_data,
            model=model,
            reduce_lr=reduce_lr,
        )
        chestx_metric = validation_epoch(
            epoch=epoch,
            device=device,
            data=dst_vl_data,
            model=model,
            reduce_lr=reduce_lr,
        )

        # Print metrics
        if device == 0:
            metricprinter(labelnames, padchest_metric, chestx_metric)

        if args.prev_alpha > 0:  # Reset every epoch
            tr_data, tr_sampler = setup_alpha_dataloader(
                img_path=args.img_path,
                ori_paths=ori_paths,
                ori_label=ori_label,
                dst_paths=dst_paths,
                dst_label=dst_label,
                datransforms=datransforms,
                batch_size=args.batch_size,
                prev_alpha=args.prev_alpha,
                pin_memory=args.pin_memory,
                num_workers=args.num_workers
            )
        if device == 0:
            # Print empty line on one device
            print("")
            t1 = time.time()
        train_epoch(
            epoch=epoch,
            device=device,
            data=tr_data,
            sampler=tr_sampler,
            model=model,
            opt=opt,
            loss_fn=loss_fn,
        )
        if device == 0:
            times.append(time.time() - t1)

    if device == 0:
        print(mean(times))


if __name__ == "__main__":
    main()
