import argparse, json, os, random
import numpy as np, torch, torch.nn as nn, torchvision
from torch.utils.data import DataLoader
from data import train_tf, test_tf
from model import get_model

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

@torch.no_grad()
def evaluate(model, loader, crit, dev):
    model.eval(); loss_sum = correct = n = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out = model(x)
        loss_sum += crit(out, y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return loss_sum / n, 100 * correct / n

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ckpt', default='/content/drive/MyDrive/cs6886w/mbv2.pt')
    a = p.parse_args()
    set_seed(a.seed); dev = 'cuda'
    os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)

    tr = torchvision.datasets.CIFAR10('./data', True,  download=True, transform=train_tf)
    te = torchvision.datasets.CIFAR10('./data', False, download=True, transform=test_tf)
    trl = DataLoader(tr, a.bs, shuffle=True, num_workers=2, pin_memory=True)
    tel = DataLoader(te, 256, shuffle=False, num_workers=2)

    model = get_model().to(dev)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9,
                          weight_decay=a.wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler('cuda')
    hist, start = [], 0

    if os.path.exists(a.ckpt):                      # resume after a disconnect
        ck = torch.load(a.ckpt)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched']); hist = ck['hist']; start = ck['epoch'] + 1
        print('Resumed from epoch', start)

    for ep in range(start, a.epochs):
        model.train(); loss_sum = correct = n = 0
        for x, y in trl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.autocast('cuda'):
                out = model(x); loss = crit(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item(); n += len(y)
        sched.step()
        tl, ta = evaluate(model, tel, crit, dev)
        hist.append(dict(epoch=ep, train_loss=loss_sum/n, train_acc=100*correct/n,
                         test_loss=tl, test_acc=ta))
        print(hist[-1])
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), hist=hist, epoch=ep), a.ckpt)

    json.dump(hist, open('history.json', 'w'))
    print('FINAL test acc:', hist[-1]['test_acc'])

if __name__ == '__main__':
    main()

