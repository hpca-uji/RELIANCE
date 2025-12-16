import os
import torch


def load_labels(label_path):
    tr_label = torch.load(os.path.join(label_path, "tr_label.pt"))
    vl_label = torch.load(os.path.join(label_path, "vl_label.pt"))
    ts_label = torch.load(os.path.join(label_path, "ts_label.pt"))
    return {"tr": tr_label, "vl": vl_label, "ts": ts_label}


def load_preds(models_dir, models_idfs, partitions):

    preds = dict()
    for idf in models_idfs:
        inference_path = os.path.join(models_dir, idf, "inferencedata/")
        for part in partitions:
            load_path = os.path.join(inference_path, part + "_preds.pt")
            preds[(idf, part)] = torch.load(load_path)
    return preds
