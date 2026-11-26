# CS6886W Assignment 2: MobileNet-v2 on CIFAR-10 with from-scratch compression

## Environment
- Google Colab, NVIDIA T4 GPU, Python [x.x.x]
- torch [x.x.x], torchvision [x.x.x], numpy [x.x.x], matplotlib [x.x.x], scikit-learn [x.x.x], wandb 0.28.1
- Install: `pip install -r requirements.txt`
- Seed: 42 (set in train.py via `set_seed`; compression and sweep scripts call `torch.manual_seed(42)`)
- No compression library is used; all quantization code is in `compress.py`.

## Files
| File | Purpose |
|---|---|
| data.py | CIFAR-10 transforms (augmentation + normalization) |
| model.py | MobileNet-v2 (width 1.0, dropout 0.2, strides adapted for 32x32) |
| train.py | Baseline training (SGD, warmup + cosine, label smoothing) |
| evaluate.py | Test accuracy, parameter count, confusion matrix |
| compress.py | Per-channel weight quantization + bit-packing, activation quantization, size accounting |
| sweep.py | Quantization sweep logged to wandb |

## Reproduce
1. Q1: train the baseline (about [X] hours on a T4)
   python train.py --epochs 100
   (writes mbv2_baseline.pt and history.json)
2. Q1: evaluate
   python evaluate.py --weights mbv2_baseline.pt
3. Q2: compress one configuration
   python compress.py --weights mbv2_baseline.pt --w_bits 4 --a_bits 6
4. Q3: run the sweep (14 runs, logs to wandb project mbv2-quant; run `wandb login` first)
   python sweep.py

## Results
| Setting | Accuracy | Size | Compression |
|---|---|---|---|
| FP32 baseline | 93.86% | 9.084 MB | 1.00x |
| 4-bit W / 6-bit A (best) | 93.24% | 1.451 MB | 6.26x (weights 7.49x, activations 5.33x) |

Full sweep: q3_results.csv and wandb_parallel_coordinates.png
wandb project: https://wandb.ai/sonal-iit-madras/mbv2-quant

## Notes
- Training on GPU with mixed precision is not bit-exact; reruns may differ by a few tenths of a percent.
- Checkpoint: mbv2_baseline.pt (weights only) is included, so steps 3 and 4 run without retraining.
