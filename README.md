## Implementation of  "FastMIM: Expediting Masked Image Modeling Pre-training for Vision"

#### Set up
```
- python==3.x
- cuda==10.x
- torch==1.7.0+
- mmcv-full-1.4.4+

# other pytorch/cuda/timm version can also work

# To pip your environment
sh requirement_pip_install.sh

# build your apex (optional)
cd /your_path_to/apex-master/;
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

#### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is:

```
│path/to/imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

#### Pre-training on ImageNet-1K
To train Swin-B on ImageNet-1K on a single node with 8 gpus:

```
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py --model mim_swin_base --data_path /your_path_to/data/imagenet/ --epochs 400 --warmup_epochs 10 --blr 1.5e-4 --weight_decay 0.05 --output_dir /your_path_to/fastmim_pretrain_output/ --batch_size 256 --save_ckpt_freq 50 --num_workers 10 --mask_ratio 0.75 --norm_pix_loss --input_size 128 --rrc_scale 0.2 1.0 --window_size 4 --decoder_embed_dim 256 --decoder_depth 4 --mim_loss HOG --block_size 32
```

#### Finetuning on ImageNet-1K
To fine-tune Swin-B on ImageNet-1K on a single node with 8 gpus:

```
python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py --model swin_base_patch4_window7_224 --data_path /your_path_to/data/imagenet/ --batch_size 128 --epochs 100 --blr 1.0e-3 --layer_decay 0.80 --weight_decay 0.05 --drop_path 0.1 --dist_eval --finetune /your_path_to_ckpt/checkpoint-399.pth --output_dir /your_path_to/fastmim_finetune_output/
```


#### Notice

We build our object detection and sementic segmentation codebase upon mmdet-v2.23 and mmseg-v0.28, however, we also add some features from the updated mmdet version (e.g., simple copy-paste) into our mmdet-v2.23. If you directly download the mmdet-v2.23 from [MMDet](https://github.com/open-mmlab/mmdetection), the code may report some errors.


### Results and Models

Result an ckpt will be updated when I recover from the covid-19. :(

#### Classification on ImageNet-1K (ViT-B/Swin-B/PVTv2-b2/CMT-Sl)

#### Object Detection on COCO (Swin-B based Mask R-CNN)

#### Semantic Segmentation on ADE20K (ViT-B based UpperNet)



### Acknowledgement

The classification task in this repo is based on [MAE](https://github.com/facebookresearch/mae), [SimMIM](https://github.com/microsoft/SimMIM), [SlowFast](https://github.com/facebookresearch/SlowFast) and [timm](https://github.com/rwightman/pytorch-image-models).

The object detection task in this repo is baesd on [MMDet](https://github.com/open-mmlab/mmdetection), [ViDet](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet) and [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).

The semantic segmentation task in this repo is baesd on [MMSeg](https://github.com/open-mmlab/mmsegmentation) and [BEiT](https://github.com/microsoft/unilm/tree/master/beit).

### Citation

If you find this project useful in your research, please consider cite:



### License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)