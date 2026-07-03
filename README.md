# neural_rti

> [!WARNING]
> **Experimental Project:** This repository is a first attempt at implementing Neural RTI compression and is currently under active development. It requires further work, refinement, and testing before it can be considered production-ready.

`neural_rti` is a PyTorch-based pipeline designed for compressing and evaluating Reflectance Transformation Imaging (RTI) datasets, specifically Hemispherical Harmonics (HSH), using Neural RTI.

Instead of storing large HSH/PTM polynomial coefficient maps, this pipeline trains a tiny Multi-Layer Perceptron (MLP) decoder alongside a compact spatial latent grid (latent map image). This representation can then be rendered in real-time in WebGL-based viewers.

## Features

- **HSH Parsing:** Pure Python reader for `.rti` files.
- **Neural Compression:** Trains a custom `DecoderMLP` model and generates a 4-channel latent map (`latent_map.png`) and compact decoder weights (`decoder_weights.json`).
- **Quality Evaluation:** Measures reconstruction quality (MSE, PSNR in dB) for both training light directions and unseen test light directions.

## Installation

Ensure you have Python 3.8+ installed.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

*Note: For GPU training, make sure you have a CUDA-compatible version of PyTorch installed.*

## Usage

### 1. Training

To compress an `.rti` file and output the latent map and decoder weights:

```bash
python train.py --input path/to/your/file.rti --output-dir output --epochs 50 --latent-dim 4
```

#### Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | *Required* | Path to the input `.rti` file. |
| `--output-dir` | `output` | Directory where weights and latent maps will be saved. |
| `--epochs` | `50` | Number of training epochs. |
| `--steps-per-epoch` | `1000` | Random sampling steps per epoch. |
| `--lr` | `0.005` | Learning rate. |
| `--latent-dim` | `4` | Dimension of the latent space (4 channels matches standard RGBA PNG formats). |
| `--resize` | `0` | Resize the image dimensions before training (0 to keep original size). |
| `--num-lights` | `64` | Number of sampled hemispherical light directions for training. |
| `--batch-size` | `262144` | Number of random pixels per batch step. |

### 2. Evaluation

To evaluate the reconstruction quality (MSE and PSNR) against the original HSH file:

```bash
python evaluate.py --input path/to/your/file.rti --weights output/decoder_weights.json --latent output/latent_map.png
```

## Outputs

- `decoder_weights.json`: Contains weight matrices and biases for the trained MLP decoder.
  ```json
  {
    "w1": [[...], ...],  // Weight matrix for Layer 1 (latent_dim + 3 -> 16)
    "b1": [...],         // Biases for Layer 1 (16 values)
    "w2": [[...], ...],  // Weight matrix for Layer 2 (16 -> 16)
    "b2": [...],         // Biases for Layer 2 (16 values)
    "w3": [[...], ...],  // Weight matrix for Layer 3 (16 -> 3)
    "b3": [...]          // Biases for Layer 3 (3 RGB values)
  }
  ```
- `latent_map.png`: A 4-channel RGBA PNG containing the spatial latent variables for each pixel.

## License

This project is licensed under the [MIT License](LICENSE).
