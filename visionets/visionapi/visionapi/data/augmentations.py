import random
import math
from typing import Union, Optional, List

from torchvision.transforms import Lambda, RandomAffine, RandomHorizontalFlip
import torch.nn as nn
import torch
from timm.data import RandAugment
from PIL import Image, ImageOps, ImageEnhance
import PIL


class SimpleDataAugmentation(nn.Module):
    """simple data augmentation class based on RandomFlip and RandomAffine"""

    def __init__(self, degrees=0.0, translate=0.0, scale=0.0, shear=0.0):
        """Init DataAugmentation class

        Args:
        degress: Random affine degress
        translate: Random translate moves
        scale: Random scale moves
        shear: Random shear degrees
        """
        super().__init__()

        # Check translate param
        if translate == 0.0:
            self.translate = None
        else:
            self.translate = (translate, translate)

        # Check scale param
        if scale == 0.0:
            self.scale = None
        else:
            self.scale = (1 - scale, 1 + scale)

        # Check shear param
        if shear == 0.0:
            self.shear = None
        else:
            self.shear = (-shear, shear, -shear, shear)

        self.degrees = degrees

        # Set up transforms
        self.transforms = nn.Sequential(
            RandomHorizontalFlip(p=0.5),
            RandomAffine(
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
            ),
        )

    @torch.no_grad()
    def forward(self, x):
        return self.transforms(x)


## Constants definition ##
# Maximum level of magnitude
_LEVEL_DENOM = 10.0
# Pil version used
_PIL_VER = tuple([int(x) for x in PIL.__version__.split(".")[:2]])
if hasattr(Image, "Resampling"):
    _RANDOM_INTERPOLATION = (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC)
    _DEFAULT_INTERPOLATION = Image.Resampling.BICUBIC
else:
    _RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)
    _DEFAULT_INTERPOLATION = Image.BICUBIC


# Helper functions
def _interpolation(kwargs):
    interpolation = kwargs.pop("resample", _DEFAULT_INTERPOLATION)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    return interpolation


def _check_args_tf(kwargs):
    if "fillcolor" in kwargs and _PIL_VER < (5, 0):
        kwargs.pop("fillcolor")
    kwargs["resample"] = _interpolation(kwargs)


# Transformation functions and level to args assignations
def shear_x(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def rotate(img, degrees, **kwargs):
    _check_args_tf(kwargs)
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, **kwargs)
    if _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
        )
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs)
    return img.rotate(degrees, resample=kwargs["resample"])


def scale(img, factor, **kwargs):

    return ImageOps.scale(img, 1 + factor)


def hflip(img, **__):
    return ImageOps.mirror(img)


def auto_contrast(img, **__):
    return ImageOps.autocontrast(img)


def invert(img, **__):
    return ImageOps.invert(img)


def equalize(img, **__):
    return ImageOps.equalize(img)


def contrast(img, factor, **__):
    return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__):
    return ImageEnhance.Color(img).enhance(factor)


def brightness(img, factor, **__):
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    return ImageEnhance.Sharpness(img).enhance(factor)


def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v


def _rotate_level_to_arg(level, _hparams):
    # range [-30, 30]
    level = (level / _LEVEL_DENOM) * 30.0
    level = _randomly_negate(level)
    return (level,)


def _enhance_increasing_level_to_arg(level, _hparams):
    # the 'no change' level is 1.0, moving away from that towards 0. or 2.0 increases the enhancement blend
    # range [0.1, 1.9] if level <= _LEVEL_DENOM
    level = (level / _LEVEL_DENOM) * 0.9
    level = max(0.1, 1.0 + _randomly_negate(level))  # keep it >= 0.1
    return (level,)


def _shear_level_to_arg(level, _hparams):
    # range [-0.3, 0.3]
    level = (level / _LEVEL_DENOM) * 0.3
    level = _randomly_negate(level)
    return (level,)


def _translate_rel_level_to_arg(level, hparams):
    # default range [-0.2, 0.2]
    translate_pct = hparams.get("translate_pct", 0.2)
    level = (level / _LEVEL_DENOM) * translate_pct
    level = _randomly_negate(level)
    return (level,)


_RAND_INCREASING_TRANSFORMS = [
    # Pool of transformations to be applied on RandAugment
    "Hflip",
    "AutoContrast",
    "Equalize",
    #'Invert',
    "Rotate",
    #'Scale',
    "ColorIncreasing",
    "ContrastIncreasing",
    "BrightnessIncreasing",
    "SharpnessIncreasing",
    "ShearX",
    "ShearY",
    "TranslateXRel",
    "TranslateYRel",
]

LEVEL_TO_ARG = {
    # Assignation of level to arg function for each transform
    "Hflip": None,
    "AutoContrast": None,
    "Equalize": None,
    "Invert": None,
    "Rotate": _rotate_level_to_arg,
    "Scale": _translate_rel_level_to_arg,
    "ColorIncreasing": _enhance_increasing_level_to_arg,
    "ContrastIncreasing": _enhance_increasing_level_to_arg,
    "BrightnessIncreasing": _enhance_increasing_level_to_arg,
    "SharpnessIncreasing": _enhance_increasing_level_to_arg,
    "ShearX": _shear_level_to_arg,
    "ShearY": _shear_level_to_arg,
    "TranslateXRel": _translate_rel_level_to_arg,
    "TranslateYRel": _translate_rel_level_to_arg,
}

NAME_TO_OP = {
    # Assignation of transformation id to transform function
    "Hflip": hflip,
    "AutoContrast": auto_contrast,
    "Equalize": equalize,
    "Invert": invert,
    "Rotate": rotate,
    "Scale": scale,
    "ColorIncreasing": color,
    "ContrastIncreasing": contrast,
    "BrightnessIncreasing": brightness,
    "SharpnessIncreasing": sharpness,
    "ShearX": shear_x,
    "ShearY": shear_y,
    "TranslateXRel": translate_x_rel,
    "TranslateYRel": translate_y_rel,
}


class AugmentOp:

    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams
        self.name = name
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = dict(
            fillcolor=hparams["img_mean"],
            resample=(
                hparams["interpolation"]
                if "interpolation" in hparams
                else _RANDOM_INTERPOLATION
            ),
        )

        # If magnitude_std is > 0, we introduce some randomness
        # in the usually fixed policy and sample magnitude from a normal distribution
        # with mean `magnitude` and std-dev of `magnitude_std`.
        # NOTE This is my own hack, being tested, not in papers or reference impls.
        # If magnitude_std is inf, we sample magnitude from a uniform distribution
        self.magnitude_std = self.hparams.get("magnitude_std", 0)
        self.magnitude_max = self.hparams.get("magnitude_max", None)

    def __call__(self, img):
        if self.prob < 1.0 and random.random() > self.prob:
            return img
        magnitude = self.magnitude
        if self.magnitude_std > 0:
            # magnitude randomization enabled
            if self.magnitude_std == float("inf"):
                # inf == uniform sampling
                magnitude = random.uniform(0, magnitude)
            elif self.magnitude_std > 0:
                magnitude = random.gauss(magnitude, self.magnitude_std)
        # default upper_bound for the timm RA impl is _LEVEL_DENOM (10)
        # setting magnitude_max overrides this to allow M > 10 (behaviour closer to Google TF RA impl)
        upper_bound = self.magnitude_max or _LEVEL_DENOM
        magnitude = max(0.0, min(magnitude, upper_bound))
        level_args = (
            self.level_fn(magnitude, self.hparams)
            if self.level_fn is not None
            else tuple()
        )
        return self.aug_fn(img, *level_args, **self.kwargs)

    def __repr__(self):
        fs = self.__class__.__name__ + f"(name={self.name}, p={self.prob}"
        fs += f", m={self.magnitude}, mstd={self.magnitude_std}"
        if self.magnitude_max is not None:
            fs += f", mmax={self.magnitude_max}"
        fs += ")"
        return fs


def rand_augment_ops(
    magnitude: Union[int, float] = 10, prob: float = 0.5, hparams=None, transforms=None
):
    return [
        AugmentOp(name, prob=prob, magnitude=magnitude, hparams=hparams)
        for name in transforms
    ]


def custom_rand_augment_transform(magnitude, num_layers, mstd):

    hprams = {"img_mean": 0, "translate_pct": 0.2, "magnitude_std": mstd}
    ra_ops = rand_augment_ops(
        magnitude=magnitude,
        prob=0.5,
        hparams=hprams,
        transforms=_RAND_INCREASING_TRANSFORMS,
    )
    return RandAugment(ops=ra_ops, num_layers=num_layers, choice_weights=None)
