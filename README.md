# Proximal Splitting Adversarial Attacks for Semantic Segmentation https://arxiv.org/abs/2206.07179

This repository contains the code of the experiments of the paper. It **does not** contain the code of the ALMA $\mathrm{prox}$ attack proposed in the paper, which is directly implemented in the [adversarial-library](https://github.com/jeromerony/adversarial-library) in the `adv_lib.attacks.segmentation` module. Additionally, the ASMA, DAG, PDGD and PDPGD attacks for segmentation are also implemented there.

## Upate 04/2023: accepted for publication at CVPR 2023

#### Citation
```bibtex
@inproceedings{rony2022proximal,
  title={Proximal Splitting Adversarial Attacks for Semantic Segmentation},
  author={Rony, J{\'e}r{\^o}me and Pesquet, Jean-Christophe and {Ben Ayed}, Ismail},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023},
  month={June},
}
```

## Getting the necessary files

Download the checkpoints for MMSegmentation models:
```bash
wget -i checkpoints/checkpoint_urls.txt -P checkpoints/
```
And clone the MMSegmentation library at the root of the dir (to retrieve the configs of the models):
```bash
git clone https://github.com/open-mmlab/mmsegmentation
```

Download and extract Pascal VOC 2012 and Cityscapes validation sets to `data` and `data/cityscapes` respectively.
The structure should be the following:
```
data
|-- VOCdevkit
|   `-- VOC2012
|       |-- Annotations
|       |-- ImageSets
|       |-- JPEGImages
|       |-- SegmentationClass
|       `-- SegmentationObject
|-- cityscapes
|   |-- gtCoarse
|   |   `-- val
|   |-- gtFine
|   |   `-- val
|   |-- leftImg8bit
|   |   `-- val
|   |-- test.txt
|   |-- train.txt
|   `-- val.txt
```

Pascal VOC 2012 can be downloaded at: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

You need an account to download Cityscapes.

## Setting up the environment

The main dependencies are the following:
 - pytorch 1.11.0
 - torchvision 0.12.0
 - sacred
 - adv_lib https://github.com/jeromerony/adversarial-library
 - mmcv-full
 - mmsegmentation

To setup the environment using conda, you can use the seg_attacks_env.yml file with:
```bash
conda env create -f seg_attacks_env.yml
```
This will install pytorch, torchvision, sacred and adv_lib. To complete the setup, activate the environment `conda activate seg_attacks` and install mmcv-full and mmsegmentation with:
```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
pip install mmsegmentation
```

## Running experiments

### Basic usage

We use sacred to manage the experiments, and refer the user to their documentation: https://sacred.readthedocs.io/en/stable/quickstart.html.
To perform $\ell_\infty$ untargeted attacks on Pascal VOC 2012 with a DeepLabV3+ ResNet-50, and save the results in `<SAVE_DIR>`, run:
```bash
python attack_experiment.py -F <SAVE_DIR> with dataset.pascal_voc_2012 model.deeplabv3plus_resnet50 cudnn_flag=benchmark attack.alma_prox_linf
```

Additionally, you can specify a target with either an `.png` containing the targets as colors, or an `int` as the target for every pixel. For instance, on Pascal VOC 2012, to have a background target, simply add the `target=0` arg:
```bash
python attack_experiment.py -F <SAVE_DIR> with dataset.pascal_voc_2012 model.deeplabv3plus_resnet50 cudnn_flag=benchmark attack.alma_prox_linf target=0
```

To run the experiments on a reduced number of samples, you can also add the `dataset.num_images=10` to attack only the first 10 images of the dataset.

### List of experiments

Tu run all the experiments of the paper, we combine the following list of arguments related to datasets and models:
```text
dataset.pascal_voc_2012 model.fcn_hrnetv2_w48
dataset.pascal_voc_2012 model.deeplabv3plus_resnet50
dataset.pascal_voc_2012 model.deeplabv3plus_resnet101
dataset.cityscapes model.fcn_hrnetv2_w48
dataset.cityscapes model.deeplabv3plus_resnet50
dataset.cityscapes model.segformer_mitb0
dataset.cityscapes model.segformer_mitb3
dataset.pascal_voc_2012 model.fcn_hrnetv2_w48 target=0
dataset.pascal_voc_2012 model.deeplabv3plus_resnet50 target=0
dataset.pascal_voc_2012 model.deeplabv3plus_resnet101 target=0
dataset.cityscapes model.fcn_hrnetv2_w48 target=cityscapes_resources/manual_smooth_majority_label.png
dataset.cityscapes model.deeplabv3plus_resnet50 target=cityscapes_resources/manual_smooth_majority_label.png
dataset.cityscapes model.segformer_mitb0 target=cityscapes_resources/manual_smooth_majority_label.png
dataset.cityscapes model.segformer_mitb3 target=cityscapes_resources/manual_smooth_majority_label.png
```

With all the configurations of the attacks:
 - I-FGSM 13x20 : `attack.minimal_mifgsm attack.mu=0`
 - MI-FGSM 13x20 : `attack.minimal_mifgsm`
 - PGD :
   - CE 13x40 : `attack.minimal_pgd_linf`
   - CE 13x4x10 : `attack.minimal_pgd_linf attack.num_steps=10 attack.restarts=4 attack.relative_step_size=0.125`
   - CE 13x40 : `attack.minimal_pgd_linf attack.loss=dlr`
   - CE 13x4x10 : `attack.minimal_pgd_linf attack.loss=dlr attack.num_steps=10 attack.restarts=4 attack.relative_step_size=0.125`
 - DAG :
   - Linf η=0.001 : `attack.dag attack.gamma=0.001`
   - Linf η=0.003 : `attack.dag attack.gamma=0.003`
 - FMN : `attack.fmn_linf`
 - PDPGD : `attack.pdpgd`
 - ALMA prox : `attack.alma_prox_linf`

### Results

The results of the experiments are stored in the `<SAVE_DIR>/<ID>` folders, following the `sacred` default template. In particular, the APSRs and distances of the adversarial examples are stored in the `info.json` file.
