# GSNet Modern

Modernized implementation of "Graspness Discovery in Clutters for Fast and Accurate Grasp Detection" (ICCV 2021).

This is a refactored fork of the [graspness_implementation](https://github.com/rhett-chen/graspness_implementation) by [Zibo Chen](https://github.com/rhett-chen), with significant changes to improve portability, maintainability, and ease of installation.

[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.pdf)]
[[dataset](https://graspnet.net/)]
[[API](https://github.com/graspnet/graspnetAPI)]

![Panda Grasp Demo](media/panda_grasp.gif)

## Key Changes from Original

- **No MinkowskiEngine**: Replaced with [spconv](https://github.com/traveller59/spconv) for sparse convolutions
- **No custom CUDA extensions**: PointNet++ operators reimplemented in pure PyTorch (no compilation required)
- **Docker-based installation**: Single-command setup with broad GPU support (Pascal through Hopper)
- **Multiple backbone options**: ResUNet14, Point Transformer V3, Sonata, PointNet++
- **Modern training features**: DDP multi-GPU, mixed precision (AMP), gradient accumulation, cosine LR with warmup
- **Integration server**: ZeroMQ-based inference server for robotic applications
- **Stable score prediction**: Optional grasp stability prediction to penalize tipping-prone grasps inpspired by [Anygrasp](https://github.com/graspnet/anygrasp_sdk) 

## Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU (Compute Capability 6.1+: GTX 1080 through H100)

## Installation

### Docker (Recommended)

1. **Clone the repository:**
```bash
git clone git@github.com:SimonHanrath/graspness_unofficial_but_better.git
cd graspness_unofficial_but_better
```

2. **Build the container:**
```bash
docker build -t graspness_modern .
```

3. **Run the container:**
```bash
docker run --rm -it --runtime=nvidia \
    --shm-size=8g \
    -v $(pwd):/workspace \
    -v /path/to/graspnet:/datasets/graspnet \
    graspness_modern bash
```

**Notes:**
- `--shm-size=8g` is recommended for PyTorch DataLoader with multiple workers
- Replace `/path/to/graspnet` with your local path to the GraspNet-1Billion dataset
- Add `-e NVIDIA_VISIBLE_DEVICES=0` to select a specific GPU
- Add `-p 5555:5588` if using the ZMQ inference server

## Dataset Preparation

### Download GraspNet-1Billion

Download the dataset from [graspnet.net](https://graspnet.net/). You will need to register for an account.

### Generate Graspness Labels

Point-level graspness labels are not included in the original dataset and need to be generated. The generation code is in [dataset/generate_graspness.py](dataset/generate_graspness.py).

```bash
python dataset/generate_graspness.py --dataset_root /datasets/graspnet --camera_type realsense
```

Or use `--camera_type kinect` for Kinect camera data.

### Simplify Dataset (Optional)

The original dataset grasp label files contain redundant data. Simplifying them significantly reduces memory usage. The code is in [dataset/simplify_dataset.py](dataset/simplify_dataset.py).

```bash
python dataset/simplify_dataset.py --dataset_root /datasets/graspnet
```

## Training

For training command examples refer to [command_train.sh](command_train.sh)


## Inference Server

For integration with robotic systems, a ZeroMQ-based server is provided:

Example:

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

## Model Weights

Pretrained model weights are available under [Releases](https://github.com/SimonHanrath/graspness_unofficial_but_better/releases).

## Acknowledgement

- Original GSNet implementation: [graspnet-implementation](https://github.com/rhett-chen/graspness_implementation/tree/main)
- Point Transformer V3 & Sonata: [Pointcept](https://github.com/Pointcept/Pointcept)
- Sparse convolutions: [spconv](https://github.com/traveller59/spconv)
