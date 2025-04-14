# General imports
import os
import argparse
import numpy as np

import torch
from torch.cuda import set_device
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import AUROC, AveragePrecision, MetricCollection
from timm.layers.norm_act import convert_sync_batchnorm
from torchvision.utils import save_image

from visionapi.models import export_model
from visionapi.models import ChestModel
from visionapi.losses import CommonLoss
from visionapi.data import SimpleDataAugmentation, custom_rand_augment_transform
from visionapi.data import setup_simple_dataloader, setup_complex_dataloader
from visionapi.opts import setup_opt, setup_frozenopt


def load_args():
    """Parameter loading"""
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str)
    parser.add_argument("label_path", type=str)
    parser.add_argument("--pre_path", type=str, default=None)
    parser.add_argument(
        "--semi_id",
        type=str,
        default=None,
        help="Identifier of semi-supervised dataset",
    )
    parser.add_argument("--hierarchy_path", type=str, default="none")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument(
        "--metric_save_path",
        type=str,
        default=None,
        help="Write best metric result to this path",
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--epochs_freeze", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        "--opt",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "nadam", "radam", "adamax"],
    )
    parser.add_argument(
        "--pw_apply",
        type=str,
        default="dual",
        choices=["cls", "distill", "dual"],
        help="Where to apply pw",
    )
    parser.add_argument(
        "--pw_formula",
        type=str,
        default="multilabel",
        choices=["multilabel", "multiclass"],
    )
    parser.add_argument(
        "--loss", type=str, default="bce", choices=["bce", "asym", "mc"]
    )

    parser.add_argument("--tradaug", type=float, nargs=4, default=None)
    parser.add_argument("--randaug", type=str, nargs=3, default=["5", "3", "0.5"])

    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--ptdrop_rate", type=float, default=0)

    parser.add_argument(
        "--pre", type=str, default="resize", choices=["resize", "convrand", "convfixed"]
    )
    parser.add_argument(
        "--base_model_name", type=str, default="deit_base_distilled_patch16_384"
    )
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--layers", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--neurons", type=int, default=0)
    parser.add_argument(
        "--norm_type", type=str, default="none", choices=["bn", "ln", "none"]
    )
    parser.add_argument(
        "--drop_type", type=str, default="alpha", choices=["drop", "alpha"]
    )
    parser.add_argument("--drop_rate", type=float, default=0.0)
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "prelu", "selu"],
    )
    parser.add_argument("--num_label", type=int, default=32)

    parser.add_argument("--name", type=str, default="train")

    args = parser.parse_args()

    # Check argument incompatibilities
    if args.pre_path is None and args.semi_id is not None:
        parser.error("Semi supervised training requires precompute path")
    if args.hierarchy_path == "none" and args.loss == 1:
        parser.error("MCLoss requires hierarchy path")
    if args.pw_apply is None and args.pw_formula is not None:
        parser.error("pw_apply must be provided if pw_formula is provided")
    elif args.pw_apply is not None and args.pw_formula is None:
        parser.error("pw_formula must be provided if pw_apply is provided")

    if "deit" not in args.base_model_name and args.pre_path is not None:
        parser.error("Distillation requires deit distilled model")
    if args.ptdrop_rate > 0:
        if "deit" not in args.base_model_name and "vit" not in args.base_model_name:
            parser.error("Patch drop rate available only with vit and deit models")
    if args.loss == 1 and args.hierarchy_path == "none":
        parser.error("MCLoss requieres hierarchy path")

    # Return parse object
    return args


def setup_gpu():
    """Multi-GPU setup"""
    init_process_group(backend="nccl")
    device = int(os.environ["LOCAL_RANK"])
    set_device(device)
    return device


def setup_model(args, device):
    """Create and set up model"""

    # Load hierarchy matrix
    if args.hierarchy_path != "none":
        H = torch.load(args.hierarchy_path)
        H = H.unsqueeze(0).to(device)
    else:
        H = "none"

    # Instantiate model
    model = ChestModel(
        pre=args.pre,
        base_model_name=args.base_model_name,
        input_size=args.input_size,
        layers=args.layers,
        neurons=args.neurons,
        norm_type=args.norm_type,
        drop_type=args.drop_type,
        drop_rate=args.drop_rate,
        ptdrop_rate=args.ptdrop_rate,
        activation=args.activation,
        num_label=args.num_label,
        hierarchy=H,
    )
    # Convert BN to SyncBN using timm function
    model = convert_sync_batchnorm(model)

    if device == 0:
        print(model)

    # Set up multi-gpu
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[device])

    # Add Metrics
    auroc = AUROC(
        num_labels=args.num_label,
        task="multilabel",
        validate_args=False,
        thresholds=200,
    ).to(device)
    mAP = AveragePrecision(
        num_labels=args.num_label,
        task="multilabel",
        validate_args=False,
        thresholds=200,
    ).to(device)
    multimetrics = MetricCollection({"auc": auroc, "mAP": mAP})
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
        model.metric.update(model.module.process_logits(logits), label_loss.int())

        # Print progress
        if device == 0 and i % 100 == 0:
            print(f"{i:04d} / {len(data)}: {train_epoch_loss/(i+1):.4f}", flush=True)

    # Compute metric (convert to items for legibility) and reset
    metric = {
        key: round(value.item(), 4) for key, value in model.metric.compute().items()
    }
    model.metric.reset()

    # Print final epoch stats
    if device == 0:
        print(
            f"Epoch {epoch+1} --> train_loss: {train_epoch_loss/len(data):.4f}, train_metrics: {metric}"
        )


def validation_epoch(epoch, device, args, best_metric, data, model, reduce_lr):
    """Function that validates the model on the given data por one epoch"""
    # Activate eval mode
    model.eval()

    with torch.no_grad():  # No grad to avoid overflow
        for i, (inputs, _, label) in enumerate(data):

            # Load to GPU
            inputs = inputs.to(device)

            # Forward inputs
            preds = model(inputs)

            # Agg metric
            model.metric.update(preds, label.int().to(device))

    # Compute metric and reset
    metric = {
        key: round(value.item(), 4) for key, value in model.metric.compute().items()
    }
    model.metric.reset()

    # Print results
    if device == 0:
        print(f"Epoch {epoch+1} --> val_metrics: {metric}")

    # Check model saving
    if metric["auc"] > best_metric:
        best_metric = metric["auc"]
        if device == 0 and args.save_path is not None:
            export_model(
                model=model.module, save_path=args.save_path, save_format="torch"
            )

    # Call reduce_lr
    reduce_lr.step(metric["auc"])

    if device == 0:
        # Print empty line on one device
        print("")
    return best_metric


def main():
    """Main function"""

    # Initialization
    args = load_args()
    device = setup_gpu()
    model = setup_model(args=args, device=device)

    # Initialize data augmentation module
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

    # Set up train and validation dataloaders
    tr_data, tr_sampler, pw = setup_complex_dataloader(
        img_path=args.img_path,
        pre_path=args.pre_path,
        label_path=args.label_path,
        semi_id=args.semi_id,
        pw_apply=args.pw_apply,
        pw_formula=args.pw_formula,
        device=device,
        datransforms=datransforms,
        num_repeats=args.num_repeats,
        batch_size=args.batch_size,
    )

    vl_data, _ = setup_simple_dataloader(
        img_path=args.img_path,
        label_path=args.label_path,
        idf_list=["vl"],
        multigpu=1,
        datransforms=None,
        batch_size=args.batch_size,
    )

    # Set up frozen optimizer
    opt, reduce_lr = setup_frozenopt(
        opt_value=args.opt, pre=args.pre, lr=0.001, model=model, device=device
    )

    # Setup distillation training
    if args.pre_path is not None:
        model.module.model.set_distilled_training(True)
        loss_fn = CommonLoss(
            pw=pw, distillation=1, hierarchy=model.module.hierarchy, loss=args.loss
        )
    else:  # Normal training
        loss_fn = CommonLoss(
            pw=pw, distillation=0, hierarchy=model.module.hierarchy, loss=args.loss
        )

    # Setup best metric var
    best_metric = 0

    # Frozen training
    for epoch in range(args.epochs_freeze):
        train_epoch(
            epoch=epoch,
            device=device,
            data=tr_data,
            sampler=tr_sampler,
            model=model,
            opt=opt,
            loss_fn=loss_fn,
        )
        best_metric = validation_epoch(
            epoch=epoch,
            device=device,
            args=args,
            best_metric=best_metric,
            data=vl_data,
            model=model,
            reduce_lr=reduce_lr,
        )

    # Setup new optimizer
    opt, reduce_lr = setup_opt(
        opt_value=args.opt, pre=args.pre, lr=args.lr, model=model, device=device
    )

    # Set up early stop
    earlystop_epochs = 10
    notimprov_epochs = 0

    # Normal training
    for epoch in range(args.epochs):
        train_epoch(
            epoch=epoch,
            device=device,
            data=tr_data,
            sampler=tr_sampler,
            model=model,
            opt=opt,
            loss_fn=loss_fn,
        )
        new_best_metric = validation_epoch(
            epoch=epoch,
            device=device,
            args=args,
            best_metric=best_metric,
            data=vl_data,
            model=model,
            reduce_lr=reduce_lr,
        )

        torch.distributed.barrier()

        # Add to counter o reset it
        if new_best_metric == best_metric:
            notimprov_epochs += 1
        else:
            notimprov_epochs = 0
            best_metric = new_best_metric
        best_metric = new_best_metric

        # Abort if not improvement for some epochs
        if notimprov_epochs == earlystop_epochs:
            break

    # Save result to file if no model is saved
    if args.metric_save_path is not None and device == 0:
        with open(args.metric_save_path, "a") as f:
            f.write(f"{args.save_path}: {best_metric} \n")


if __name__ == "__main__":
    main()
