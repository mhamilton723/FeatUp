# FeatUp: A Model-Agnostic Framework for Features at Any Resolution
### [Project Page](https://aka.ms/featup) | [Paper](https://aka.ms/featup-paper) | [Colab Notebook](https://colab.research.google.com/github/mhamilton723/FeatUp/blob/main/example_usage.ipynb) | ICLR 2024


[Stephanie Fu*](https://stephanie-fu.github.io/),
[Mark Hamilton*](https://mhamilton.net/),
[Zhoutong Zhang](https://ztzhang.info/),
[Laura Brandt](https://people.csail.mit.edu/lebrandt/),
[Axel Feldman](https://feldmann.nyc/),
[William T. Freeman](https://billf.mit.edu/about/bio)

![FeatUp Overview Graphic](https://mhamilton.net/images/website_hero_small-p-1080.jpg)

This is the official implementation of the paper "FeatUp: A Model-Agnostic Framework for Features at Any Resolution". *Equal Contribution.

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

| Model Name | Checkpoint                                                                                                                         | Torch Hub Repository | Torch Hub Name |
|------------|------------------------------------------------------------------------------------------------------------------------------------|----------------------|----------------|
| DINO       | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/dino16_jbu_stack_cocostuff.ckpt)   | `mhamilton723/FeatUp`  | `dino16`        |
| DINO v2    | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/dinov2_jbu_stack_cocostuff.ckpt)   | `mhamilton723/FeatUp`  | `dinov2`         |
| CLIP       | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/clip_jbu_stack_cocostuff.ckpt)     | `mhamilton723/FeatUp`  | `clip`           |
| ViT        | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/vit_jbu_stack_cocostuff.ckpt)      | `mhamilton723/FeatUp`  | `vit`            |
| ResNet50   | [Download](https://marhamilresearch4.blob.core.windows.net/feature-upsampling-public/pretrained/resnet50_jbu_stack_cocostuff.ckpt) | `mhamilton723/FeatUp`  | `resnet50`       |

For example, to load the FeatUp JBU upsampler for the DINO backbone:

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




## Coming Soon:

- Training your own FeatUp joint bilateral upsampler
- Simple API for Implicit FeatUp training
- Pretrained JBU models without layer-norms 


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
