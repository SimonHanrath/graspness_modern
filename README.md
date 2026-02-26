# GSNet Modern (Work in Progress)

Modernized implementation of "Graspness Discovery in Clutters for Fast and Accurate Grasp Detection" (ICCV 2021).

This is a refactored fork of the [unofficial implementation](https://github.com/rhett-chen/graspness_unofficial) by [Zibo Chen](https://github.com/rhett-chen), with significant changes to improve portability, maintainability, and ease of installation.

[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.pdf)]
[[dataset](https://graspnet.net/)]
[[API](https://github.com/graspnet/graspnetAPI)]

## Key Changes from Original

- **No MinkowskiEngine**: Replaced with [spconv](https://github.com/traveller59/spconv) for sparse convolutions
- **No custom CUDA extensions**: PointNet++ operators reimplemented in pure PyTorch (no compilation required)
- **Docker-based installation**: Single-command setup with broad GPU support (Pascal through Hopper)
- **Multiple backbone options**: ResUNet (default), PointNet++, Point Transformer V3
- **Modern training features**: DDP multi-GPU, mixed precision (AMP), gradient accumulation
- **Integration server**: ZeroMQ-based inference server for robotic applications

## Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU (Compute Capability 6.1+: GTX 1080 through H100)

## Installation

### Docker (Recommended)

Build the container:
```bash
docker build -t graspness_modern .
```

Run the container:
```bash
docker run --rm -it --runtime=nvidia \
    -v $(pwd):/workspace \
    -v /path/to/graspnet:/datasets/graspnet \
    graspness_modern bash
```

### Manual Installation (Advanced)

If you prefer not to use Docker, the main dependencies are:
- Python 3.10+
- PyTorch 2.0+ with CUDA
- spconv-cu12x
- graspnetAPI (included in this repo)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install spconv-cu121 numpy scipy tqdm tensorboard
pip install ./graspnetAPI
```

## Dataset Preparation

Download the GraspNet-1Billion dataset from [graspnet.net](https://graspnet.net/).

### Generate Graspness Labels
Point-level graspness labels are not included in the original dataset:
```bash
python dataset/generate_graspness.py --dataset_root /datasets/graspnet --camera_type realsense
```

### Simplify Grasp Labels (Optional)
Reduce memory footprint by removing redundant data:
```bash
python dataset/simplify_dataset.py --dataset_root /datasets/graspnet
```

## Training

Basic training:
```bash
python train.py \
    --dataset_root /datasets/graspnet \
    --camera realsense \
    --model_name gsnet_resunet \
    --log_dir logs/gsnet_resunet \
    --backbone resunet \
    --batch_size 2 \
    --max_epoch 10
```

### Backbone Options
- `resunet` - ResUNet14 sparse convolutional backbone (default, recommended)
- `resunet_rgb` - ResUNet14 with RGB features
- `pointnet2` - PointNet++ backbone
- `transformer` - Point Transformer V3
- `transformer_pretrained` - Point Transformer V3 with pretrained weights

### Advanced Training Options
```bash
python train.py \
    --dataset_root /datasets/graspnet \
    --camera realsense \
    --model_name gsnet_advanced \
    --log_dir logs/gsnet_advanced \
    --backbone resunet \
    --batch_size 1 \
    --max_epoch 20 \
    --use_amp \
    --num_workers 4 \
    --persistent_workers \
    --lazy_grasp_labels \
    --cosine_lr \
    --grad_clip 1.0
```

### Multi-GPU Training (DDP)
```bash
torchrun --nproc_per_node=2 train.py \
    --dataset_root /datasets/graspnet \
    --camera realsense \
    --model_name gsnet_ddp \
    --log_dir logs/gsnet_ddp \
    --backbone resunet \
    --batch_size 1
```

## Testing and Evaluation

```bash
python model_analysis/test.py \
    --dataset_root /datasets/graspnet \
    --camera realsense \
    --checkpoint_path logs/gsnet_resunet/gsnet_resunet_epoch10.tar \
    --dump_dir dumps/gsnet_resunet \
    --backbone resunet \
    --infer --eval
```

Set `--collision_thresh -1` for faster inference without collision detection.

## Inference Server

For integration with robotic systems, a ZeroMQ-based server is provided:

```bash
python GraspDetectionClient.py \
    --checkpoint_path logs/gsnet_resunet/gsnet_resunet_epoch10.tar \
    --port 5555 \
    --collision_thresh 0.01
```

The server accepts compressed point clouds (NumPy `.npz` format) and returns grasp candidates as JSON. See [GraspDetectionClient.py](GraspDetectionClient.py) for protocol details.

## Model Weights
TODO

## Project Structure

```
├── dataset/                 # Dataset loading and preprocessing
├── models/                  # Model architectures
│   ├── graspnet.py         # Main GSNet model
│   ├── backbone_resunet14.py
│   ├── backbone_pointnet2.py
│   └── pointcept/          # Point Transformer V3
├── utils/                   # Utilities
│   └── pointnet/           # Pure PyTorch PointNet++ ops
├── model_analysis/          # Testing and evaluation scripts
├── graspnetAPI/            # Evaluation API
├── dockerfile              # Docker build file
├── train.py                # Training script
└── GraspDetectionClient.py # ZMQ inference server
```

## Acknowledgement

- Original GSNet implementation: [graspnet-baseline](https://github.com/graspnet/graspnet-baseline)
- Unofficial implementation: [graspness_unofficial](https://github.com/rhett-chen/graspness_unofficial) by Zibo Chen
- Point Transformer V3: [Pointcept](https://github.com/Pointcept/Pointcept)
- Sparse convolutions: [spconv](https://github.com/traveller59/spconv)
