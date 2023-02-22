# CoordGAN: Self-Supervised Dense Correspondences Emerge from GANs
This repository contains the official implementation for CoordGAN introduced in the following paper: [CoordGAN: Self-Supervised Dense Correspondences Emerge from GANs (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Mu_CoordGAN_Self-Supervised_Dense_Correspondences_Emerge_From_GANs_CVPR_2022_paper.pdf). The code is developed based on the Pytorch framework(1.8.0) with python 3.7

> [**CoordGAN: Self-Supervised Dense Correspondences Emerge from GANs**](https://openaccess.thecvf.com/content/CVPR2022/papers/Mu_CoordGAN_Self-Supervised_Dense_Correspondences_Emerge_From_GANs_CVPR_2022_paper.pdf)  
> [*Jiteng Mu*](https://jitengmu.github.io/),
[*Shalini De Mello*](https://research.nvidia.com/person/shalini-gupta),
[*Zhiding Yu*](https://research.nvidia.com/person/zhiding-yu),
[*Nuno Vasconcelos*](http://www.svcl.ucsd.edu/~nuno/),
[*Xiaolong Wang*](https://xiaolonw.github.io/),
[*Jan Kautz*](https://research.nvidia.com/person/jan-kautz),
[*Sifei Liu*](https://research.nvidia.com/person/sifei-liu)   
> CVPR, 2022

[Project Page](https://jitengmu.github.io/CoordGAN/) / [ArXiv](https://arxiv.org/pdf/2203.16521.pdf) / [Video](https://www.youtube.com/watch?v=FP27huY0Yu0)

<div align="center">
<img src="figs/teaser.gif" width="75%">
</div>


## Prepare datasets and pretrained checkpoints
### Datasets
Please follow the individual instructions to download the datasets and put data in the `data` directory.
| Dataset | Description
| :--- | :----------
|[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) | Please follow the [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) instructions to download the CelebAMask-HQ dataset.
|[Stanford Cars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) | Please follow the [Stanford Cars](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) instructions to download the Stanford Cars training subset.
|[AFHQ-cat](https://github.com/clovaai/stargan-v2) | Please follow the [AFHQ](https://github.com/clovaai/stargan-v2) instructions to download the AFHQ cat training subset.
|[DatasetGAN](https://github.com/nv-tlabs/datasetGAN_release) | [DatasetGAN](https://github.com/nv-tlabs/datasetGAN_release) annoated images are used for semantic label propagation evaluation.

### Pretrained models
Please follow the individual instructions to download the datasets and put data in the `data` directory.
| Checkpoints | Description
| :--- | :----------
|[IR-SE50 Model](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | Follow the repo [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel) to download pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch). This is required for ArcFace score evaluation. 

## Requirements
The project is developed with packages shown in the `reqruirements.txt`. Please install the packages by running
```
pip install -r requirements.txt
```

## Stage 1. Training CoordGAN
We design a structure-texture disentangled GAN such that dense correspondence can be extracted explicitly from the structural component, where the key component is to represent the image structure in a coordinate space that is shared by all images. Specifically, the structure of each generated image is represented as a warped coordinate frame, transformed from a shared canonical 2D coordinate frame. 

<div align="center">
<img src="figs/texture_swap_celebA.gif" width="75%">
</div>

CoordGAN generates images at resolution 128x128. Please run the following command to train CoordGAN,
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train.py --output_dir CHECKPOINT_128 --config configs/celebA_128.yaml
```
where `--output_dir` specifies the output directory, `--config` specifies the config file (`configs/celebA_128.yaml` or `configs/stanfordcar_128.yaml` or `configs/afhqcat_128.yaml`).

### High resolution
For CelebAMask-HQ, we first train CoordGAN with an output size of 128 × 128 and then append two upsampling layers to generate high-resolution images (512x512). With the previous checkpoint, CoordGAN can be further trained with following command, 
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train.py --output_dir celebA_512 --config configs/celebA_512.yaml --ckpt CHECKPOINT_128 --reinit_discriminator True --fix_struc_grad True
```
where `--output_dir` specifies the output directory, `--config` specifies the config file, `--ckpt` specifies the checkpoint of resolution 128x128.

### Eval FID
Please running the following command to evaluate the FID score,
```
python eval_fid.py --ckpt CHECKPOINT --dataset DATASET --size SIZE
```
where `--ckpt` specifies the CoordGAN checkpoint, `--dataset` specifies the category (currently support `celebA` or `stanfordcar` or `afhq-cat`), `--size` specifies the resolution (currently 128 or 512). Each iteration synthesize a pair of images plus a pair of images with swapped texture codes. The `input` folder contains the resized real images, `samples` folder contains the synthesized images, and `samples_swap` contains the texture swapped images.

After generate images, FID score can be obtained with [torch_fidelity](https://github.com/toshas/torch-fidelity). Please install the package and then run,
```
fidelity --gpu 0 --fid --input1 GENERATED_IMAGES --input2 REAL_IMAGES
```
where `GENERATED_IMAGES` should be replaced with the `samples` folder path after running the previous command and `REAL_IMAGES` with the `input` folder path.

## Stage2. Inverting CoordGAN via an Encoder

<div align="center">
<img src="figs/correspondence_celebA.gif" width="75%">
</div>

The CoordGAN can be equipped with an encoder to enable the extraction of dense correspondence from real images. Please run the following command to train an encoder,
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train-encoder.py --output_dir celebA_enc --config configs/celebA_enc.yaml --ckpt ENC_CHECKPOINT
```
where `--output_dir` specifies the output directory, `--config` specifies the config file (`configs/celebA_enc.yaml` or `configs/stanfordcar_enc.yaml`), --ckpt specifies the trained GAN checkpoint in the first stage.

### Eval label propagation
We quantitatively demonstrate the quality of the extracted dense correspondence on the task of semantic label propagation. Given one reference image with semantic labels, its correspondence map is first inferred with the trained encoder. Another correspondence map is then inferred for a query image and
the labels of the reference image can be obtained. This can be done by running,
```
python eval_corr.py --ckpt ENC_CHECKPOINT --segdataset SEGDATASET
```
where `--ckpt` specifies the obtained the ENC_CHECKPOINT checkpoint `--segdataset` specifies the category (currently support `datasetgan-face-34` or `datasetgan-car-20` or `celebA-7`)

## Citation
If you found our work useful, please cite
```
@InProceedings{mu2022coordgan,
              author = {Mu, Jiteng and De Mello, Shalini and Yu, Zhiding
                          and Vasconcelos, Nuno and Wang, Xiaolong and Kautz, Jan and Liu, Sifei},
              title = {CoordGAN: Self-Supervised Dense Correspondences Emerge from GANs},
              booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
              month = {June},
              year = {2022}
}
```

The code is heavely based on the [styleganv2 pytorch implementation](https://github.com/rosinality/stylegan2-pytorch), [CIPS](https://github.com/saic-mdal/CIPS), [Swapping Autoencoder](https://github.com/taesungp/swapping-autoencoder-pytorch)

Nvidia-licensed CUDA kernels (fused_bias_act_kernel.cu, upfirdn2d_kernel.cu) is for non-commercial use only.
