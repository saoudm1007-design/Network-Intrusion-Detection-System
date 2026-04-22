"""PyTorch MLP classifier with early stopping on validation macro-F1."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from ..utils import get_logger

log = get_logger(__name__)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], n_classes: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = len(y) / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def train_mlp(X_train, y_train, X_val, y_val, *, cfg: dict, seed: int,
              n_classes: int) -> MLP:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = cfg["models"]["mlp"]

    Xt = torch.from_numpy(X_train.astype(np.float32))
    yt = torch.from_numpy(y_train.astype(np.int64))
    Xv = torch.from_numpy(X_val.astype(np.float32)).to(device)
    yv = y_val

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=p["batch_size"],
                        shuffle=True, drop_last=False)

    model = MLP(Xt.shape[1], p["hidden_dims"], n_classes, p["dropout"]).to(device)
    weights = _class_weights(y_train, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.Adam(model.parameters(), lr=p["lr"])

    best_f1 = -1.0
    best_state = None
    patience_left = p["patience"]

    for epoch in range(1, p["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            total_loss += loss.item() * xb.size(0)

        model.eval()
        with torch.no_grad():
            preds = model(Xv).argmax(dim=1).cpu().numpy()
        val_f1 = f1_score(yv, preds, average="macro")
        log.info(f"MLP epoch {epoch:02d}  loss={total_loss / len(Xt):.4f}  val_macroF1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = p["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info(f"Early stopping at epoch {epoch} (best macroF1={best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def mlp_predict(model: MLP, X) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X.astype(np.float32)).to(device)
        return model(xb).argmax(dim=1).cpu().numpy()


def mlp_predict_proba(model: MLP, X) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X.astype(np.float32)).to(device)
        return torch.softmax(model(xb), dim=1).cpu().numpy()
