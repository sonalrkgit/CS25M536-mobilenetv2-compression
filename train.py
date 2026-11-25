import argparse, json, math, os, random
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
    p.add_argument('--warmup_epochs', type=int, default=3)
    p.add_argument('--clip', type=float, default=5.0)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ckpt', default='/content/drive/MyDrive/cs6886w/mbv2_final.pt')
    a = p.parse_args()
    set_seed(a.seed); dev = 'cuda'
    os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)
    use_amp = not a.no_amp

    tr = torchvision.datasets.CIFAR10('./data', True,  download=True, transform=train_tf)
    te = torchvision.datasets.CIFAR10('./data', False, download=True, transform=test_tf)
    trl = DataLoader(tr, a.bs, shuffle=True, num_workers=2, pin_memory=True)
    tel = DataLoader(te, 256, shuffle=False, num_workers=2)

    model = get_model().to(dev)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    # no weight decay on BatchNorm / bias (1-D) parameters
    decay    = [p_ for p_ in model.parameters() if p_.ndim > 1]
    no_decay = [p_ for p_ in model.parameters() if p_.ndim <= 1]
    opt = torch.optim.SGD([{'params': decay, 'weight_decay': a.wd},
                           {'params': no_decay, 'weight_decay': 0.0}],
                          lr=a.lr, momentum=0.9, nesterov=True)

    # per-iteration linear warmup, then ONE cosine decay to 0 over the whole run
    steps_per_ep = len(trl)
    total = a.epochs * steps_per_ep
    warm = a.warmup_epochs * steps_per_ep
    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, total - warm))   # clamp: LR never rises again
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    hist, start = [], 0
    if os.path.exists(a.ckpt):                      # resume after a disconnect
        ck = torch.load(a.ckpt)
        assert ck.get('epochs') == a.epochs, (
            f"Checkpoint was made with epochs={ck.get('epochs')}, you passed {a.epochs}. "
            "Delete the checkpoint or change --ckpt.")
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched']); hist = ck['hist']; start = ck['epoch'] + 1
        print('Resumed from epoch', start)

    for ep in range(start, a.epochs):
        model.train(); loss_sum = correct = n = 0
        for x, y in trl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=use_amp):
                out = model(x); loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            scaler.step(opt); scaler.update()
            sched.step()                            # step every iteration
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item(); n += len(y)
        tl, ta = evaluate(model, tel, crit, dev)
        hist.append(dict(epoch=ep, lr=opt.param_groups[0]['lr'], train_loss=loss_sum/n,
                         train_acc=100*correct/n, test_loss=tl, test_acc=ta))
        print(hist[-1])
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), hist=hist, epoch=ep,
                        epochs=a.epochs), a.ckpt)

    json.dump(hist, open('history.json', 'w'))
    torch.save(model.state_dict(), 'mbv2_baseline.pt')
    print('FINAL test acc:', hist[-1]['test_acc'])

if __name__ == '__main__':
    main()
