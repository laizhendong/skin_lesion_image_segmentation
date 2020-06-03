# An imporved U-Net++ Network for Skin lesion Image Segmentation

This is an implementation of ["An imporved U-Net++ Network for Skin lesion Image Segmentation"] in Keras deep learning framework (Tensorflow as backend). **U-Net++** (nested U-Net architecture) is proposed for a more precise segmentation, which introduce intermediate layers to skip connections of U-Net, which naturally form multiple new up-sampling paths from different depths, ensembling U-Nets of various receptive fields.

<p align="center">
  <img src="https://github.com/MrGiovanni/Nested-U-Net/blob/master/Figures/fig_U-Net%2B%2B.png" width="700"/>
</p>

# Paper

[U-Net++: A Nested U-Net Architecture for Medical Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1) <br/>
[Zhou Zongwei](https://www.zongweiz.com), [Md Mahfuzur Rahman Siddiquee](https://github.com/mahfuzmohammad), [Nima Tajbakhsh](https://www.linkedin.com/in/nima-tajbakhsh-b5454376/), and [Jianming Liang](https://chs.asu.edu/jianming-liang) <br/>
[Biomedical Informatics, Arizona State University](https://chs.asu.edu/programs/biomedical-informatics) <br/>
Deep Learning in Medical Image Analysis ([DLMIA](https://cs.adelaide.edu.au/~dlmia4/)) 2018. **(Oral)**


# Requirements
Python 3.6, Keras 2.2.2, Tensorflow 1.4.1 and other common packages listed in `requirements.txt`.

# Available architectures
 - [U-Net](https://arxiv.org/abs/1505.04597)
 - [DLA](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Deep_Layer_Aggregation_CVPR_2018_paper.pdf)
 - [U-Net++](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1)**
 - [FPN](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
 - [Linknet](https://arxiv.org/abs/1707.03718)
 - [PSPNet](https://arxiv.org/abs/1612.01105)
 
# Available backbones
| Backbone model      |Name| Weights    |
|---------------------|:--:|:------------:|
| VGG16               |`vgg16`| `imagenet` |
| VGG19               |`vgg19`| `imagenet` |
| ResNet18            |`resnet18`| `imagenet` |
| ResNet34            |`resnet34`| `imagenet` |
| ResNet50            |`resnet50`| `imagenet`<br>`imagenet11k-places365ch` |
| ResNet101           |`resnet101`| `imagenet` |
| ResNet152           |`resnet152`| `imagenet`<br>`imagenet11k` |
| ResNeXt50           |`resnext50`| `imagenet` |
| ResNeXt101          |`resnext101`| `imagenet` |
| DenseNet121         |`densenet121`| `imagenet` |
| DenseNet169         |`densenet169`| `imagenet` |
| DenseNet201         |`densenet201`| `imagenet` |
| Inception V3        |`inceptionv3`| `imagenet` |
| Inception ResNet V2 |`inceptionresnetv2`| `imagenet` |

# Installation

```bash
git clone https://github.com/laizhendong/skin_lesion_image_segmentation.git
pip install -r requirements.txt
git submodule update --init --recursive
```

# Running the scripts

#### Application 1: [Data ISBI 2016](https://www.kaggle.com/c/data-ISBI 2016)
```bash
CUDA_VISIBLE_DEVICES=0 python ISBI2016_application.py --run 1 \
                                                     --arch Xnet \
                                                     --backbone vgg16 \
                                                     --init random \
                                                     --decoder transpose \
                                                     --input_rows 512 \
                                                     --input_cols 512 \
                                                     --input_deps 3 \
                                                     --nb_class 1 \
                                                     --batch_size 2048 \
                                                     --weights None \
                                                     --verbose 1
```
#### Application 2: [Data ISBI 2017](https://www.kaggle.com/c/data-ISBI 2017)

```bash
CUDA_VISIBLE_DEVICES=0 python ISBI2017_application.py --run 1 \
                                                     --arch Xnet \
                                                     --backbone DenseNet169 \
                                                     --init random \
                                                     --decoder transpose \
                                                     --input_rows 512 \
                                                     --input_cols 512 \
                                                     --input_deps 3 \
                                                     --nb_class 1 \
                                                     --batch_size 2048 \
                                                     --weights None \
                                                     --verbose 1
```

# Code examples for your own data

Train a U-Net++ structure (`Xnet` in the code):  
```python
from segmentation_models import Unet, Nestnet, Xnet

# prepare data
x, y = ... # range in [0,1]

# prepare model
model = Xnet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build U-Net++
# model = Unet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build U-Net
# model = NestNet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build DLA

model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])

# train model
model.fit(x, y)
```

# To do
- [x] Add VGG backbone for U-Net++
- [x] Add ResNet backbone for U-Net++
- [x] Add ResNeXt backbone for U-Net++
- [x] Add DenseNet backbone for U-Net++
- [x] Add Inception backbone for U-Net++

# Acknowledgments

This repository has been built upon [qubvel/segmentation_models](https://github.com/qubvel/segmentation_models). We appreciate the effort of Pavel Yakubovskiy for providing well-organized segmentation models to the community. This research has been supported partially by NIH under Award Number R01HL128785, by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant. The content is solely the responsibility of the authors and does not necessarily represent the official views of NIH.


