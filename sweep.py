import argparse, wandb
from compress import run_compression

p = argparse.ArgumentParser()
p.add_argument('--weights', default='mbv2_baseline.pt')
p.add_argument('--project', default='mbv2-quant')
a = p.parse_args()

# (weight_bits, activation_bits): 14 runs
configs = [(8,8),(8,6),(8,4),(6,8),(6,6),(5,8),(4,8),(4,6),(4,4),
           (3,8),(3,6),(2,8),(2,6),(6,4)]

for wb, ab in configs:
    wandb.init(project=a.project, name=f'w{wb}_a{ab}',
               config={'weight_quant_bits': wb, 'activation_quant_bits': ab})
    r = run_compression(a.weights, wb, ab)
    wandb.log({'quantized_acc': r['quantized_acc'],
               'baseline_acc': r['baseline_acc'],
               'compression_ratio': r['compression_ratio'],
               'weight_ratio': r['weight_ratio'],
               'act_ratio': r['act_ratio'],
               'model_size_mb': r['model_size_mb']})
    print(wb, ab, r['quantized_acc'], r['compression_ratio'], r['model_size_mb'], flush=True)
    wandb.finish()
