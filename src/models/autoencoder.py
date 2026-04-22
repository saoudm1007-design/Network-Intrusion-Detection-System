"""Autoencoder trained on BENIGN-only flows; anomaly score = reconstruction error."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..utils import get_logger

log = get_logger(__name__)


class Autoencoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int]):
        super().__init__()
        enc_layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        rev = list(reversed(hidden_dims[:-1])) + [in_dim]
        prev = hidden_dims[-1]
        for i, h in enumerate(rev):
            dec_layers.append(nn.Linear(prev, h))
            if i < len(rev) - 1:
                dec_layers.append(nn.ReLU())
            prev = h
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_benign: np.ndarray, cfg: dict, seed: int) -> tuple[Autoencoder, float]:
    torch.manual_seed(seed)
    p = cfg["models"]["autoencoder"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xt = torch.from_numpy(X_benign.astype(np.float32))
    loader = DataLoader(TensorDataset(Xt, Xt), batch_size=p["batch_size"],
                        shuffle=True, drop_last=False)

    model = Autoencoder(Xt.shape[1], p["hidden_dims"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=p["lr"])
    criterion = nn.MSELoss()

    for epoch in range(1, p["epochs"] + 1):
        model.train()
        total = 0.0
        for xb, _ in loader:
            xb = xb.to(device)
            optim.zero_grad()
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optim.step()
            total += loss.item() * xb.size(0)
        if epoch % 10 == 0 or epoch == 1:
            log.info(f"AE epoch {epoch}  mse={total / len(Xt):.6f}")

    # Compute threshold from training reconstruction errors
    model.eval()
    with torch.no_grad():
        errors = []
        for xb, _ in loader:
            xb = xb.to(device)
            recon = model(xb)
            err = ((recon - xb) ** 2).mean(dim=1).cpu().numpy()
            errors.append(err)
    errors = np.concatenate(errors)
    threshold = float(np.percentile(errors, p["threshold_percentile"]))
    log.info(f"AE reconstruction threshold (p{p['threshold_percentile']}) = {threshold:.6f}")
    return model, threshold


def ae_anomaly_score(model: Autoencoder, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X.astype(np.float32)).to(device)
        recon = model(xb)
        return ((recon - xb) ** 2).mean(dim=1).cpu().numpy()


def ae_predict_binary(model: Autoencoder, X: np.ndarray, threshold: float) -> np.ndarray:
    return (ae_anomaly_score(model, X) > threshold).astype(int)
