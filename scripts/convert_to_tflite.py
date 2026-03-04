"""
Setup
-----
Requires Python 3.10 and the litert-torch package::

    pip install litert-torch

litert-torch pins its own TensorFlow fork (ai-edge-tensorflow).  If a stock
``tensorflow`` package is also installed the two will conflict at import time
with an ``undefined symbol: Wrapped_PyInit_...`` error.  Fix by removing the
stock package::

    pip uninstall tensorflow -y

Usage
-----
# fp32 (no quantization, ~208 MB)
python convert_to_tflite.py -o pix2pix_maps.tflite

# dynamic int8 quantization
python convert_to_tflite.py -o pix2pix_maps_int8.tflite --quantize dynamic_int8

# weight-only int8
python convert_to_tflite.py -o pix2pix_maps_w8.tflite --quantize weight_only_int8

# fp16
python convert_to_tflite.py -o pix2pix_maps_fp16.tflite --quantize fp16
"""

import argparse
import os
import sys

# Ensure the parent package (pytorch-CycleGAN-and-pix2pix) is importable
# regardless of the working directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import litert_torch
from litert_torch.generative.quantize import quant_recipes

from models.networks import UnetGenerator

QUANT_CONFIGS = {
    "none": None,
    "dynamic_int8": quant_recipes.full_dynamic_recipe(),
    "weight_only_int8": quant_recipes.full_weight_only_recipe(),
    "fp16": quant_recipes.full_fp16_recipe(),
}


def main():
    parser = argparse.ArgumentParser(description="Convert pix2pix model to TFLite")
    parser.add_argument("-o", "--output", required=True, help="Output .tflite path")
    parser.add_argument(
        "--quantize",
        choices=QUANT_CONFIGS.keys(),
        default="none",
        help="Quantization mode (default: none)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Path to a local .pth state-dict file (default: load from HuggingFace Hub)",
    )
    args = parser.parse_args()

    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = UnetGenerator(
        input_nc=3, output_nc=3, num_downs=8, ngf=64, use_dropout=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    sample_input = torch.randn(1, 3, 256, 256)
    quant_config = QUANT_CONFIGS[args.quantize]

    print(f"Converting with quantization={args.quantize!r} ...")
    litert_torch.signature("forward", model, (sample_input,)).convert(
        quant_config=quant_config
    ).export(args.output)
    print(f"Exported to ./{args.output}")


if __name__ == "__main__":
    main()
