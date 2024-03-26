# FeatUp: A Model-Agnostic Framework for Features at Any Resolution
###  ICLR 2024


[![Website](https://img.shields.io/badge/FeatUp-%F0%9F%8C%90Website-purple?style=flat)](https://aka.ms/featup) [![arXiv](https://img.shields.io/badge/arXiv-2403.10516-b31b1b.svg)](https://arxiv.org/abs/2403.10516) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mhamilton723/FeatUp/blob/main/example_usage.ipynb)
[![Huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FeatUp-orange)](https://huggingface.co/spaces/mhamilton723/FeatUp) 
[![Huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper%20Page-orange)](https://huggingface.co/papers/2403.10516)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/featup-a-model-agnostic-framework-for/feature-upsampling-on-imagenet)](https://paperswithcode.com/sota/feature-upsampling-on-imagenet?p=featup-a-model-agnostic-framework-for)



[Stephanie Fu*](https://stephanie-fu.github.io/),
[Mark Hamilton*](https://mhamilton.net/),
[Laura Brandt](https://people.csail.mit.edu/lebrandt/),
[Axel Feldman](https://feldmann.nyc/),
[Zhoutong Zhang](https://ztzhang.info/),
[William T. Freeman](https://billf.mit.edu/about/bio)
*Equal Contribution.

![FeatUp Overview Graphic](https://mhamilton.net/images/website_hero_small-p-1080.jpg)

*TL;DR*:FeatUp improves the spatial resolution of any model's features by 16-32x without changing their semantics.

https://github.com/mhamilton723/FeatUp/assets/6456637/8fb5aa7f-4514-4a97-aebf-76065163cdfd


## Contents
<!--ts-->
   * [Install](#install)
   * [Using Pretrained Upsamplers](#using-pretrained-upsamplers)
   * [Fitting an Implicit Upsampler](#fitting-an-implicit-upsampler-to-an-image)
   * [Coming Soon](coming-soon)
   * [Citation](#citation)
   * [Contact](#contact)
<!--te-->

## Install

### Pip
For those just looking to quickly use the FeatUp APIs install via:
```shell script
pip install git+https://github.com/mhamilton723/FeatUp
```

### Local Development
To install FeatUp for local development and to get access to the sample images install using the following:
```shell script
git clone https://github.com/mhamilton723/FeatUp.git
cd FeatUp
pip install -e .
```

## Using Pretrained Upsamplers

To see examples of pretrained model usage please see our [Collab notebook](https://colab.research.google.com/github/mhamilton723/FeatUp/blob/main/example_usage.ipynb). We currently supply the following pretrained versions of FeatUp's JBU upsampler:

| Model Name | Checkpoint                                                                                                                         | Checkpoint (No LayerNorm)                                                                                                                  | Torch Hub Repository | Torch Hub Name |
|------------|------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|----------------------|----------------|
| DINO       | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/dino16_jbu_stack_cocostuff.ckpt)   | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/no_norm/dino16_jbu_stack_cocostuff.ckpt)   | mhamilton723/FeatUp  | dino16         |
| DINO v2    | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/dinov2_jbu_stack_cocostuff.ckpt)   | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/no_norm/dinov2_jbu_stack_cocostuff.ckpt)   | mhamilton723/FeatUp  | dinov2         |
| CLIP       | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/clip_jbu_stack_cocostuff.ckpt)     | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/no_norm/clip_jbu_stack_cocostuff.ckpt)     | mhamilton723/FeatUp  | clip           |
| ViT        | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/vit_jbu_stack_cocostuff.ckpt)      | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/no_norm/vit_jbu_stack_cocostuff.ckpt)      | mhamilton723/FeatUp  | vit            |
| ResNet50   | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/resnet50_jbu_stack_cocostuff.ckpt) | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/no_norm/resnet50_jbu_stack_cocostuff.ckpt) | mhamilton723/FeatUp  | resnet50       |

For example, to load the FeatUp JBU upsampler for the DINO backbone without an additional LayerNorm on the spatial features:

```python
upsampler = torch.hub.load("mhamilton723/FeatUp", 'dino16', use_norm=False)
```

To load upsamplers trained on backbones with additional LayerNorm operations which makes training and transfer learning a bit more stable:

```python
upsampler = torch.hub.load("mhamilton723/FeatUp", 'dino16')
```

## Fitting an Implicit Upsampler to an Image

To train an implicit upsampler for a given image and backbone first clone the repository and install it for 
[local development](#local-development). Then run

```python
cd featup
python train_implicit_upsampler.py
```

Parameters for this training operation can be found in the [implicit_upsampler config file](featup/configs/implicit_upsampler.yaml).

## Local Gradio Demo

To run our [HuggingFace Spaces hosted FeatUp demo](https://huggingface.co/spaces/mhamilton723/FeatUp) locally first install FeatUp for local development. Then  run:

```shell
python gradio_app.py
```

Wait a few seconds for the demo to spin up, then navigate to [http://localhost:7860/](http://localhost:7860/) to view the demo.


## Coming Soon:

- Training your own FeatUp joint bilateral upsampler
- Simple API for Implicit FeatUp training


## Citation

```
@inproceedings{
    fu2024featup,
    title={FeatUp: A Model-Agnostic Framework for Features at Any Resolution},
    author={Stephanie Fu and Mark Hamilton and Laura E. Brandt and Axel Feldmann and Zhoutong Zhang and William T. Freeman},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=GkJiNn2QDF}
}
```

## Contact

For feedback, questions, or press inquiries please contact [Stephanie Fu](mailto:fus@mit.edu) and [Mark Hamilton](mailto:markth@mit.edu)
