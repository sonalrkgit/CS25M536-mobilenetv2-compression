import argparse, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, torchvision
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data import test_tf
from model import get_model

p = argparse.ArgumentParser()
p.add_argument('--weights', default='mbv2_baseline.pt')
a = p.parse_args()

te = torchvision.datasets.CIFAR10('./data', False, download=True, transform=test_tf)
tel = DataLoader(te, 256, shuffle=False, num_workers=2)
model = get_model().cuda()
model.load_state_dict(torch.load(a.weights))
model.eval()

preds, labels = [], []
with torch.no_grad():
    for x, y in tel:
        preds += model(x.cuda()).argmax(1).cpu().tolist(); labels += y.tolist()

acc = 100 * sum(p == l for p, l in zip(preds, labels)) / len(labels)
n_params = sum(p.numel() for p in model.parameters())
print(f'Test top-1 accuracy: {acc:.2f}%')
print(f'Parameters: {n_params:,}  FP32 size: {n_params*4/1e6:.2f} MB')

classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
ConfusionMatrixDisplay(confusion_matrix(labels, preds), display_labels=classes).plot(xticks_rotation=45)
plt.savefig('confusion.png', dpi=150, bbox_inches='tight')
