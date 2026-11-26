# Q4 results (best configuration: 4-bit weights, 6-bit activations)
- Accuracy: 93.24% (baseline 93.86%)
- Weight compression ratio: 7.49x
- Activation compression ratio: 5.33x (sum over 35 ReLU6 outputs, batch size 1, 6 bits + 1 scale per layer)
- Model compression ratio: 6.26x
- Final model size: 1.451 MB (baseline 9.084 MB)
