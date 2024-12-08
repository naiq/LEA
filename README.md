# [TMLR2024] LEA: Learning Latent Embedding Alignment Model for fMRI Decoding and Encoding

Xuelin Qian, Yikai Wang, Xinwei Sun, Yanwei Fu, Xiangyang Xue, Jianfeng Feng

<p align="center">
<img src=assets/teaser.jpg />
</p>


## Overview
We introduce LEA, a unified framework that addresses both fMRI decoding and encoding. We train two latent spaces to represent and reconstruct fMRI signals and visual images, respectively. By aligning these two latent spaces, we seamlessly transform between the fMRI signal and visual stimuli. LEA can recover visual stimuli from fMRI signals and predict brain activity from images. LEA outperforms existing methods on multiple fMRI decoding and encoding benchmarks.

<p align="center">
<img src=assets/framework.jpg />
</p>

## Environment Setups
Create and activate conda environment named ```lea``` from our ```requirements.yaml```
```sh
conda env create -f requirements.yaml
conda activate lea
```

## Data Preparation
Please download the GOD and BOLD5000 datasets from [MinD-Vis](https://github.com/zjc062/mind-vis) and put them in the ```\dataset``` folder as organized below.

```
/dataset
┣ 📂 Kamitani
┃   ┣ 📂 npz
┃   ┃   ┗ 📜 sbj_1.npz
┃   ┃   ┗ 📜 sbj_2.npz
┃   ┃   ┗ 📜 sbj_3.npz
┃   ┃   ┗ 📜 sbj_4.npz
┃   ┃   ┗ 📜 sbj_5.npz
┃   ┃   ┗ 📜 images_256.npz
┃   ┃   ┗ 📜 imagenet_class_index.json
┃   ┃   ┗ 📜 imagenet_training_label.csv
┃   ┃   ┗ 📜 imagenet_testing_label.csv

┣ 📂 BOLD5000
┃   ┣ 📂 BOLD5000_GLMsingle_ROI_betas
┃   ┃   ┣ 📂 py
┃   ┣ 📂 BOLD5000_Stimuli
┃   ┃   ┣ 📂 Image_Labels
┃   ┃   ┣ 📂 Scene_Stimuli
┃   ┃   ┣ 📂 Stimuli_Presentation_Lists

```

## Chcekpoints Download
The pre-trained weighs on the [Human Connectome Projects (HCP)]((https://db.humanconnectome.org/data/projects/HCP_1200)) dataset can be downloaded from [MinD-Vis](https://github.com/zjc062/mind-vis) repository. After downloaded, put then into the ```\pretrains``` folder.

For the checkpoints of fMRI reconstruction and image reconstruction, please download them from [Google Drive](xxx) (coming soon) and place them into the  ```\checkpoints``` folder.

All checkpoints should be organized as follows,

```
/checkpoints
┣ 📂 BOLD_sbj1
┃   ┗ 📜 checkpoint.pth
┣ 📂 BOLD_sbj2
┃   ┗ 📜 checkpoint.pth
┣ ...
┣ 📂 GOD_sbj1
┃   ┗ 📜 checkpoint.pth
┣ 📂 GOD_sbj2
┃   ┗ 📜 checkpoint.pth
┣ ...
┣ 📂 ImageDecoder_MaskGIT
┃   ┗ 📜 checkpoint.pth
┃   ┗ 📜 checkpoint_v2.pth

/pretrains
┣ 📂 BOLD
┃   ┗ 📜 fmri_encoder.pth
┣ 📂 GOD
┃   ┗ 📜 fmri_encoder.pth
┣ 📜 MaskGIT_ImageNet256_checkpoint.pth
┣ 📜 MaskGIT_Trans_ImageNet256_checkpoint.pth

```

## Inference

### 1. GOD Dataset
Run ``python LEA_GOD.py`` to reconstruct images from fMRI signals and predict fMRI signals from visual stimuli.

To evalute different individuals, some hyper-parameters and paths in TWO files need to be manually modified. In ```/LEA_GOD.py```,
```python
367 |    ckpt_encoder = 'PATH_to_encoder_ckpt' 
368 |    ckpt_decoder = 'PATH_to_decoder_ckpt'
369 |    args.cfg_file = [
370 |        'PATH_to_encoder_cfg',
371 |        'PATH_to_decoder_cfg'
372 |    ]
```

and in ```/configs/fMRI_TransAE_GOD.yaml```,
```python
19 |  num_voxel: 4466 # sbj_1: 4466, sbj_2: 4404, sbj_3: 4643, sbj_4: 4133, sbj_5: 4370
20 |  roi_patch: [1004, 801, 476, 588, 431, 249, 157, 760] # sbj1: [1004, 801, 476, 588, 431, 249, 157, 760]
21 |     # sbj2: [757, 727, 603, 416, 372, 124, 576, 829]
22 |     # sbj3: [872, 826, 605, 630, 770, 262, 306, 372]
23 |     # sbj4: [719, 652, 676, 597, 494, 236, 308, 451]
24 |     # sbj5: [659, 676, 649, 729, 661, 301, 246, 449]
25 |
26 |  sub: ['sbj_1'] # ['sbj_1', 'sbj_2', 'sbj_3', 'sbj_4', 'sbj_5']
```

The output will be saved in 
The outputs will be stored in the ``\results`` folder of the path where the fMRI reconstruction model is located

### 2. BOLD5000 Dataset
Run ``python LEA_BOLD.py`` for inference, which is similar to the operations on GOD dataset.

## Acknowledgments

Our fMRI autoencoder implementation is based on the [MAE](https://github.com/facebookresearch/mae) and [MinD-Vis](https://github.com/zjc062/mind-vis). Our image autoencoder implementation is based on [MaskGIT](https://github.com/google-research/maskgit). 
We extend our gratitude to the authors for their excellent work and publicly sharing codes!

## Citation
```
@article{qian2024lea,
  title={LEA: Learning Latent Embedding Alignment Model for fMRI Decoding and Encoding},
  author={Xuelin Qian and Yikai Wang and Xinwei Sun and Yanwei Fu and Xiangyang Xue and Jianfeng Feng},
  journal={Transactions on Machine Learning Research},
  year={2024},
  url={https://openreview.net/forum?id=SUMtDJqicd},  
}

@article{qian2023joint,
  title={Joint fMRI Decoding and Encoding with Latent Embedding Alignment},
  author={Qian, Xuelin and Wang, Yikai and Fu, Yanwei and Sun, Xinwei and Xue, Xiangyang and Feng, Jianfeng},
  journal={arXiv preprint arXiv:2303.14730},
  year={2023}
}
```

## Contact
Any questions or discussions are welcome!

Xuelin Qian (<xuelinq92@gmail.com>)
