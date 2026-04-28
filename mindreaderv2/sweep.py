"""η × MLP-hidden sweep on cached probe weights + activations."""
import json, sys, os
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/mrv2_gemma27"
print(f"[sweep] OUT={OUT}")

Xtr = np.load(f"{OUT}/acts_train.npy")
ytr = np.load(f"{OUT}/y_train.npy")
Xev = np.load(f"{OUT}/acts_eval.npy")
yev = np.load(f"{OUT}/y_eval.npy")
W   = np.load(f"{OUT}/probe_weights.npy")  # (L, H)
F1s = np.load(f"{OUT}/probe_f1.npy")
L, H = W.shape
print(f"[sweep] L={L} H={H}  N_train={len(ytr)}  N_eval={len(yev)}")

def mask_for(eta):
    m = np.zeros_like(W, dtype=bool)
    for l in range(L):
        w = W[l]; s = w.sum()
        if s <= 0: continue
        wn = w / s
        order = np.argsort(-wn)
        cum = np.cumsum(wn[order])
        k = int(np.searchsorted(cum, eta) + 1)
        m[l, order[:k]] = True
    return m

def aggregate(X, mask, alpha):
    feats = []
    for l in range(L):
        if mask[l].sum() == 0: continue
        feats.append(alpha[l] * X[:, l, mask[l]])
    return np.concatenate(feats, axis=1).astype(np.float32)

fmin, fmax = F1s.min(), F1s.max()
alpha = (F1s - fmin) / max(fmax - fmin, 1e-6)
print(f"[sweep] alpha peak L{int(alpha.argmax())} = {alpha.max():.3f}")

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[sweep] mlp dev={dev}")

def fit_mlp(Z_tr, y_tr, Z_ev, y_ev, hidden=128, epochs=20, lr=1e-3, bs=64, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Z_tr))
    cut = int(0.9*len(Z_tr))
    tr, va = perm[:cut], perm[cut:]
    Xtrt = torch.tensor(Z_tr[tr], device=dev); ytrt = torch.tensor(y_tr[tr], dtype=torch.long, device=dev)
    Xvat = torch.tensor(Z_tr[va], device=dev); yvat = torch.tensor(y_tr[va], dtype=torch.long, device=dev)
    Xevt = torch.tensor(Z_ev,    device=dev)
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(Z_tr.shape[1], hidden), nn.ReLU(),
        nn.Linear(hidden, hidden),        nn.ReLU(),
        nn.Linear(hidden, 2)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in net.parameters())
    best_va, best_state = -1, None
    for ep in range(epochs):
        net.train()
        idx = torch.randperm(len(Xtrt), device=dev)
        for i in range(0, len(idx), bs):
            sel = idx[i:i+bs]
            loss = lossf(net(Xtrt[sel]), ytrt[sel])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            va = torch.softmax(net(Xvat), dim=1)[:,1].cpu().numpy()
        va_auc = roc_auc_score(yvat.cpu().numpy(), va)
        if va_auc > best_va:
            best_va = va_auc
            best_state = {k: v.clone() for k,v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        ev = torch.softmax(net(Xevt), dim=1)[:,1].cpu().numpy()
    auc = roc_auc_score(y_ev, ev)
    f1 = f1_score(y_ev, (ev > 0.5).astype(int))
    return auc, f1, best_va, n_params

ETA = [0.3, 0.5, 0.7, 0.9]
HID = [128, 512, 1024]

print(f"\n{'eta':>5} {'hid':>5} {'feat':>6} {'val_auc':>8} {'auroc':>7} {'f1':>6} {'params_M':>9}")
print("-"*60)
results = []
for eta in ETA:
    m = mask_for(eta)
    keep = int(m.sum())
    Ztr = aggregate(Xtr, m, alpha)
    Zev = aggregate(Xev, m, alpha)
    for h in HID:
        auc, f1, va, n = fit_mlp(Ztr, ytr, Zev, yev, hidden=h)
        print(f"{eta:>5.2f} {h:>5d} {keep:>6d} {va:>8.4f} {auc:>7.4f} {f1:>6.4f} {n/1e6:>9.3f}")
        results.append({"eta":eta,"hidden":h,"feat":keep,"val_auc":va,"auroc":auc,"f1":f1,"params_M":n/1e6})

best = max(results, key=lambda r: r["auroc"])
print(f"\nbest: eta={best['eta']} hidden={best['hidden']} -> AUROC={best['auroc']:.4f} F1={best['f1']:.4f}")
json.dump({"results": results, "best": best},
          open(f"{OUT}/sweep_result.json", "w"), indent=2)
print(f"saved {OUT}/sweep_result.json")
