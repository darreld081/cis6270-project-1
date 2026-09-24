"""Flow matching for Trametinib single-cell PCA data, mirroring the structure
of ChatterjeeLab/CIS6270 lecture_3/esm2_flow_guidance.py but swapping the
ESM-2 sequence encoder for this project's own TrametinibSingleBranchDataModule.

Usual flow matching convention: Z_0 = noise (standing in for the untreated /
DMSO population), Z_1 = real Trametinib-treated cells. The model learns a
velocity field that transports samples from noise to the treated-cell
distribution, exactly like the reference script's Z_0=noise, Z_1=clean latent
setup -- just with PCA coordinates instead of ESM-2 residue embeddings.

By default the "noise" is not a generic N(0,I) prior -- it's a Gaussian fit
to the real DMSO (untreated) population's mean and covariance, so sampling
starts from something that actually looks like the untreated cell state
rather than an arbitrary standard normal. Pass --noise-source standard to
fall back to plain N(0,I).

Run: python train_flow_matching.py --epochs 200
Install: pip install torch pytorch-lightning lightning scikit-learn scipy pandas
"""
import argparse
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from dataloaders.trametinib_loader import TrametinibSingleBranchDataModule  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN, LEARNING_RATE = 128, 1e-3


# 1. Data loading: the project's own LightningDataModule parses the CSV,
# builds the Trametinib-treated (X1) cluster and real DMSO (X0) population,
# and exposes a k-NN tree over the full data for manifold guidance.
def load_data(args):
    dm = TrametinibSingleBranchDataModule(args)
    return dm


# Fit the Z_0 noise source. "data" fits a Gaussian to the real untreated
# (DMSO) cells so sampled noise matches their location/scale/correlations;
# "standard" reproduces the reference script's plain N(0,I) prior.
def fit_noise_source(dm, dim, kind="data"):
    if kind == "standard":
        return {"mean": torch.zeros(1, dim), "chol": torch.eye(dim)}
    if kind != "data":
        raise ValueError("noise-source must be 'data' or 'standard'")
    x0 = dm.coords_t0
    mean = x0.mean(0, keepdim=True)
    cov = torch.cov(x0.T) + 1e-4 * torch.eye(dim)  # Ridge for numerical stability.
    chol = torch.linalg.cholesky(cov)
    return {"mean": mean, "chol": chol}


def sample_noise(stats, n, dim):
    return stats["mean"].to(DEVICE) + torch.randn(n, dim, device=DEVICE) @ stats["chol"].to(DEVICE).T


# 2. Small velocity-field model: flattening lets the output depend on the
# whole PCA vector at once, same design as the reference FlowModel.
class FlowModel(nn.Module):
    def __init__(self, dim, hidden=HIDDEN):
        super().__init__()
        self.dim = dim
        self.time = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.skip = nn.Linear(32, 1)  # Time-gated linear passthrough of the state.
        self.net = nn.Sequential(
            nn.Linear(dim + 32, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z, t):
        time = self.time(t[:, None])
        inputs = torch.cat([z, time], dim=1)
        return self.skip(time) * z + self.net(inputs)
        # A time-dependent linear part plus a learned nonlinear velocity correction.


# 3. Flow matching: Z_t=(1-t)Z_0+tZ_1 with Z_0~N(noise_mean,noise_cov);
# target velocity Z_1-Z_0.
def _unwrap_train_batch(batch):
    # dm.train_dataloader() nests two CombinedLoaders; iterating it yields
    # (nested_batch_dict, batch_idx, dataloader_idx) regardless of mode.
    x1, _weights = batch["train_samples"]["x1"]
    return x1


def train(dm, noise_stats, model, epochs, lr=LEARNING_RATE):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = dm.train_dataloader()
    for epoch in range(epochs):
        total, n_batches = 0.0, 0
        for batch, _batch_idx, _dataloader_idx in loader:
            z1 = _unwrap_train_batch(batch).to(DEVICE)  # Real Trametinib-treated cells, raw PCA scale.
            z0 = sample_noise(noise_stats, len(z1), model.dim)  # Data-fit (or standard) noise source.
            t = torch.rand(len(z1), device=DEVICE)
            zt = (1 - t[:, None]) * z0 + t[:, None] * z1
            loss = F.mse_loss(model(zt, t), z1 - z0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1
        if (epoch + 1) % 50 == 0 or epoch + 1 == epochs:
            print(f"epoch {epoch+1}: flow matching loss {total / max(n_batches, 1):.4f}")
    model.eval().requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    return model


# 4. Manifold guidance: pull the trajectory toward the k-NN-smoothed data
# manifold the dataloader already builds (dm.tree / dm.dataset), in place of
# the reward-gradient guidance the reference script used for sequence objectives.
def manifold_guidance(dm, z, t, eta):
    if eta == 0:
        return torch.zeros_like(z)
    proj_fn = dm.get_manifold_proj(None)
    projected = proj_fn(z.detach().cpu()).to(z.device)
    kappa = eta * 4 * t[:, None] * (1 - t[:, None])  # Fades guidance at both endpoints.
    return kappa * (projected - z)


# 5. Sampling: Euler-integrate from the fitted noise source (real DMSO
# mean/covariance by default) to t=1, optionally steering with manifold guidance.
@torch.no_grad()
def sample(model, dm, noise_stats, n, steps=100, eta=0.0):
    z = sample_noise(noise_stats, n, model.dim)  # Z_0, representing the untreated cells.
    dt = 1.0 / steps
    for step in range(steps):
        t = torch.full((n,), step * dt, device=DEVICE)
        v = model(z, t)
        if eta != 0:
            v = v + manifold_guidance(dm, z, t, eta) / dt
        z = z + dt * v
    return z


# 6. Evaluation: mean distance of generated cells to each metric-sample
# cluster centroid, mirroring the reference's printed composition proxies.
@torch.no_grad()
def evaluate(dm, generated):
    generated = generated.cpu()
    for i, loader in enumerate(dm.metric_samples_dataloaders):
        cluster = next(iter(loader))
        centroid = cluster.mean(dim=0)
        dist = (generated - centroid).norm(dim=1).mean().item()
        print(f"  mean distance to metric cluster {i}: {dist:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path,
                         default=ROOT / "data" / "Trametinib_5.0uM_pca_and_leidenumap_labels.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=50)
    parser.add_argument("--whiten", action="store_true")
    parser.add_argument("--split-ratios", type=float, nargs="+", default=[0.8, 0.2])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--steps", type=int, default=100, help="Euler integration steps")
    parser.add_argument("--guidance-eta", type=float, default=0.0,
                         help="Manifold guidance strength; 0 disables it")
    parser.add_argument("--noise-source", choices=["data", "standard"], default="data",
                         help="'data' fits Z_0 to the real DMSO mean/covariance; "
                              "'standard' uses a plain N(0,I) prior")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=ROOT / "flow_outputs")
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 1 or args.steps < 1:
        parser.error("epochs, samples, and steps must be positive")

    torch.manual_seed(args.seed)
    if DEVICE.type == "cpu":
        torch.set_num_threads(2)

    dm = load_data(args)
    dim = dm.coords_t1.shape[1]
    noise_stats = fit_noise_source(dm, dim, args.noise_source)
    model = FlowModel(dim).to(DEVICE)
    train(dm, noise_stats, model, args.epochs, args.lr)

    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "baseline": sample(model, dm, noise_stats, args.samples, steps=args.steps, eta=0.0),
        "manifold_guided": sample(model, dm, noise_stats, args.samples, steps=args.steps, eta=args.guidance_eta),
    }
    for name, generated in outputs.items():
        if not torch.isfinite(generated).all():
            raise RuntimeError(f"Nonfinite {name} output; reduce guidance-eta or check training")
        print(f"{name}:")
        evaluate(dm, generated)

    torch.save({
        "model": model.state_dict(),
        "outputs": {k: v.cpu() for k, v in outputs.items()},
        "noise_stats": {k: v.cpu() for k, v in noise_stats.items()},
        "args": vars(args),
    }, args.output / "results.pt")
    print(f"Saved model weights and generated coordinates to {args.output}")


if __name__ == "__main__":
    main()
