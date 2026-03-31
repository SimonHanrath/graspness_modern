# GSNet Modern

Modernized implementation of "Graspness Discovery in Clutters for Fast and Accurate Grasp Detection" (ICCV 2021).

This is a refactored fork of the [unofficial implementation](https://github.com/rhett-chen/graspness_unofficial) by [Zibo Chen](https://github.com/rhett-chen), with significant changes to improve portability, maintainability, and ease of installation.

[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.pdf)]
[[dataset](https://graspnet.net/)]
[[API](https://github.com/graspnet/graspnetAPI)]

![Panda Grasp Demo](media/panda_grasp.gif)

## Key Changes from Original

- **No MinkowskiEngine**: Replaced with [spconv](https://github.com/traveller59/spconv) for sparse convolutions
- **No custom CUDA extensions**: PointNet++ operators reimplemented in pure PyTorch (no compilation required)
- **Docker-based installation**: Single-command setup with broad GPU support (Pascal through Hopper)
- **Multiple backbone options**: Point Transformer V3 (default), Sonata, ResUNet14/18, PointNet++
- **Modern training features**: DDP multi-GPU, mixed precision (AMP), gradient accumulation, cosine LR with warmup
- **Integration server**: ZeroMQ-based inference server for robotic applications
- **Stable score prediction**: Optional grasp stability prediction to penalize tipping-prone grasps

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
- [graspnetAPI](https://github.com/graspnet/graspnetAPI)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install spconv-cu121 numpy scipy tqdm tensorboard
pip install graspnetAPI
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
    --model_name gsnet_ptv3 \
    --log_dir logs/gsnet_ptv3 \
    --backbone transformer \
    --batch_size 2 \
    --max_epoch 10
```

### Backbone Options
- `transformer` - Point Transformer V3 (default)
- `transformer_pretrained` - Point Transformer V3 with pretrained weights
- `sonata` - Sonata self-supervised PTv3 (CVPR 2025)
- `resunet` - ResUNet14 sparse convolutional backbone
- `resunet18` - ResUNet18 (more layers)
- `resunet_rgb` / `resunet18_rgb` - ResUNet with RGB features
- `pointnet2` - PointNet++ backbone

### Advanced Training Options
```bash
python train.py \
    --dataset_root /datasets/graspnet \
    --camera realsense \
    --model_name gsnet_sonata \
    --log_dir logs/gsnet_sonata \
    --backbone sonata \
    --batch_size 1 \
    --max_epoch 20 \
    --use_amp \
    --num_workers 4 \
    --persistent_workers \
    --lazy_grasp_labels \
    --cosine_lr \
    --grad_clip 1.0 \
    --weight_decay 0.02 \
    --enable_stable_score
```

For pretrained backbones (`transformer_pretrained`, `sonata`), layer-wise learning rate decay is applied by default (`--layer_decay 0.65`).

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

## Inference Server

For integration with robotic systems, a ZeroMQ-based server is provided:

```bash
python zmq_server.py \
    --checkpoint_path logs/gsnet_resunet/gsnet_resunet_epoch10.tar \
    --backbone resunet \
    --port 5588 \
    --graspness_threshold 0.01
```

The server accepts point clouds via ZMQ and returns grasp candidates as JSON:
```json
[
  {
    "translation": [x, y, z],
    "rotation_matrix": [[r00, r01, r02], ...],
    "score": 0.85,
    "width": 0.08,
    "height": 0.02,
    "depth": 0.02
  }
]
```

Additional options: `--max_angle_to_vertical_deg`, `--vertical_axis`, `--enable_stable_score`.

## Project Structure

```
├── dataset/                 # Dataset loading and preprocessing
│   ├── graspnet_dataset.py # PyTorch dataset
│   ├── generate_graspness.py
│   └── generate_graspness_full.py  # With floor points
├── models/                  # Model architectures
│   ├── graspnet.py         # Main GSNet model
│   ├── backbone_resunet14.py
│   ├── backbone_pointnet2.py
│   └── pointcept/          # Point Transformer V3 & Sonata
├── utils/                   # Utilities
│   └── pointnet/           # Pure PyTorch PointNet++ ops
├── dockerfile              # Docker build file
├── train.py                # Training script
└── zmq_server.py           # ZMQ inference server
```

## Acknowledgement

- Original GSNet implementation: [graspnet-baseline](https://github.com/graspnet/graspnet-baseline)
- Unofficial implementation: [graspness_unofficial](https://github.com/rhett-chen/graspness_unofficial) by Zibo Chen
- Point Transformer V3 & Sonata: [Pointcept](https://github.com/Pointcept/Pointcept)
- Sparse convolutions: [spconv](https://github.com/traveller59/spconv)
