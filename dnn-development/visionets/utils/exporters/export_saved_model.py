import argparse

import torch

from visionapi.models import ChestModel
from visionapi.models import export_model, load_model


def load_args():
    """Parameter loading"""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("save_format", type=str, choices=["torch","jit","onnx"])
    parser.add_argument("--subformat", type=str, choices=["script","trace"])
    parser.add_argument("--opset", type=int, choices=range(7,17))
    args = parser.parse_args()

    return args

def setup_model(args):
    """Create and set up model"""
    model = load_model(model_path=args.model_path)
    model.eval()
    return model

def main():

    args = load_args()
    model = setup_model(args) 
    export_model(model=model, 
                 save_path=args.save_path,
                 save_format=args.save_format,
                 input_size=(2,1,512,512), 
                 subformat=args.subformat, 
                 opset=args.opset)
    print("Export complete")   

if __name__ == "__main__":
    main()