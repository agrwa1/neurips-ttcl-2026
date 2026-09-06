# Third-party notices

This project uses third-party datasets, pretrained model weights, and Python
packages. They remain subject to their own licenses and terms. Raw datasets and
pretrained weight files are not redistributed here; `figure_data/` and `output/`
contain derived numerical results and figures.

## Datasets

### CIFAR-100

CIFAR-100 was created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton from
the 80 Million Tiny Images collection. The [official CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html)
requests citation of:

> Alex Krizhevsky. *Learning Multiple Layers of Features from Tiny Images*. 2009.

The distribution page does not state a general-purpose dataset license. Obtain
the data from the official source, cite it, and determine that the intended use
is permitted. This repository accesses it through `torchvision` and does not
redistribute it.

### ImageNet-R

ImageNet-R was introduced by Dan Hendrycks, Steven Basart, Norman Mu, Saurav
Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike
Guo, Dawn Song, Jacob Steinhardt, and Justin Gilmer in:

> Dan Hendrycks et al. *The Many Faces of Robustness: A Critical Analysis of
> Out-of-Distribution Generalization*. ICCV, 2021.

Use the [official ImageNet-R release](https://github.com/hendrycks/imagenet-r).
Its repository carries the MIT License and a 2020 copyright notice for Dan
Hendrycks. Individual images may have separate copyright holders and are not
relicensed by this project.

### ImageNet

The pretrained backbones were trained on ImageNet-1K and, for ViT-B/16,
ImageNet-21K. ImageNet was created by the ImageNet research team; cite:

> Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei.
> *ImageNet: A Large-Scale Hierarchical Image Database*. CVPR, 2009.

ImageNet does not own the copyright in its indexed images. Access supplied by
ImageNet is governed by its [access agreement](https://image-net.org/accessagreement),
including use only for non-commercial research and education. These restrictions
must be considered when using ImageNet-trained weights.

## Model architectures and pretrained weights

- ViT-B/16 is loaded through [`timm`](https://github.com/huggingface/pytorch-image-models)
  using ImageNet-21K pretraining followed by ImageNet-1K fine-tuning. The model
  implementation is distributed under Apache-2.0; pretrained weights can carry
  additional conditions arising from their training data.
- ResNet-50 is loaded through [`torchvision`](https://github.com/pytorch/vision)
  with weights hosted by PyTorch and trained on ImageNet-1K. Torchvision warns
  that pretrained weights may have terms derived from their training datasets.

Architecture citations:

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. *Deep Residual Learning
  for Image Recognition*. CVPR, 2016.
- Alexey Dosovitskiy et al. *An Image Is Worth 16x16 Words: Transformers for
  Image Recognition at Scale*. ICLR, 2021.

## Software dependencies

Dependencies are installed separately from the locked package indexes and are
not vendored into this repository. Their installed distributions include their
applicable license texts and third-party notices.

| Package | License | Project/license |
| --- | --- | --- |
| Matplotlib | Matplotlib License | [license](https://matplotlib.org/stable/project/license.html) |
| NumPy | BSD-3-Clause | [license](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| pandas | BSD-3-Clause | [license](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| Pillow | MIT-CMU | [license](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| PyYAML | MIT | [license](https://github.com/yaml/pyyaml/blob/main/LICENSE) |
| timm | Apache-2.0 | [license](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE) |
| PyTorch | Multiple permissive licenses; see distribution | [license](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| torchvision | BSD-3-Clause | [license](https://github.com/pytorch/vision/blob/main/LICENSE) |
| pytest (development) | MIT | [license](https://github.com/pytest-dev/pytest/blob/main/LICENSE) |
| Ruff (development) | MIT | [license](https://github.com/astral-sh/ruff/blob/main/LICENSE) |

The exact resolved versions and transitive dependencies are recorded in
`uv.lock`. Anyone redistributing a bundled environment, executable, container,
or other binary distribution must also retain the license and notice files from
all included packages and their transitive components.
