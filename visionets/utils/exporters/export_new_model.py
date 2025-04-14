import argparse

import torch
from torch import nn

from visionapi.models import ChestModel
from visionapi.models import export_model


def load_args():
    """Parameter loading"""
    parser = argparse.ArgumentParser()
    parser.add_argument("save_path", type=str)
    parser.add_argument("save_format", type=str, choices=["torch","jit","onnx"])
    parser.add_argument("--subformat", type=str, choices=["script","trace"])
    parser.add_argument("--opset", type=int, choices=range(7,17))
    parser.add_argument("--quant", type=int, choices=[0,1], default=0)

    """
    deit_base_distilled_patch16_384
    convnext_xlarge
    swinv2_large_window12to24_192to384
    """

    parser.add_argument("--pre", type=str, default="resize", choices=["resize","convrand","convfixed"])
    parser.add_argument("--base_model_name", type=str, default="swinv2_large_window12to24_192to384")
    parser.add_argument("--input_size", type=int, default=384)
    parser.add_argument("--layers", type=int, default=0, choices=[0,1,2,3])
    parser.add_argument("--neurons", type=int, default=0)
    parser.add_argument("--norm_type", type=str, default="none", choices=["bn", "ln","none"])
    parser.add_argument("--drop_type", type=str, default="drop", choices=["drop", "alpha"])
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu", "prelu", "selu"])
    parser.add_argument("--num_label", type=int, default=32)
    args = parser.parse_args()

    # Return parse object
    return args

def setup_model(args, device):
    """Create and set up model"""
    model = ChestModel(pre=args.pre, 
                       base_model_name=args.base_model_name,
                       input_size=args.input_size,
                       layers=args.layers,
                       neurons=args.neurons,
                       norm_type=args.norm_type,
                       drop_type=args.drop_type,
                       drop_rate=args.drop_rate,
                       activation=args.activation,
                       num_label=args.num_label,
                       hierarchy="none")
    model = model.to(device)
    model.eval()
    return model

def quant_model(model):

    quant_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.float16)
    return quant_model

def main():
    
    args = load_args()
    device = torch.device("cpu")
    model = setup_model(args=args, device=device)
    if args.quant:
        model = quant_model(model)
    export_model(model=model, 
                 save_path=args.save_path,
                 save_format=args.save_format,
                 input_size=(2,1,512,512), 
                 subformat=args.subformat, 
                 opset=args.opset)
    print("Export complete")
            
if __name__ == "__main__":
    main()