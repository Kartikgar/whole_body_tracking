"""Supervised Isaac→Genesis gap model g(s, a) -> Δs with a deep-ensemble uncertainty estimate.

Each member is an MLP with a heteroscedastic Student-t head (location and log-scale per output dimension, fixed
degrees of freedom) trained on standardized inputs/targets. Epistemic uncertainty is the disagreement of member
locations in standardized target units; aleatoric uncertainty is the mean predicted variance.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

LOG_SCALE_MIN, LOG_SCALE_MAX = -7.0, 3.0
# The gap dataset is built in the open-loop task, whose base-action term is `motion_joint_action`; the finetune
# task's delta observation carries the same quantity (raw base action) as `current_action`.
OBS_TERM_ALIASES = {"current_action": "motion_joint_action"}


def gap_model_obs_columns(term_names: list[str], term_dims: list[int], x_names: list[str]) -> list[int]:
    """Column index in an observation group (terms concatenated in order) for each gap-model input feature."""
    columns = [f"{OBS_TERM_ALIASES.get(n, n)}[{i}]" for n, d in zip(term_names, term_dims) for i in range(d)]
    missing = [x for x in x_names if x not in columns]
    if missing:
        raise ValueError(f"Delta observation group lacks gap-model inputs: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    return [columns.index(x) for x in x_names]


class GapMember(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: tuple[int, ...] = (256, 256, 256)):
        super().__init__()
        layers, last = [], in_dim
        for width in hidden:
            layers += [nn.Linear(last, width), nn.ELU()]
            last = width
        self.body = nn.Sequential(*layers)
        self.loc = nn.Linear(last, out_dim)
        self.log_scale = nn.Linear(last, out_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.loc(h), self.log_scale(h).clamp(LOG_SCALE_MIN, LOG_SCALE_MAX)


def student_t_nll(loc: torch.Tensor, log_scale: torch.Tensor, target: torch.Tensor, dof: float) -> torch.Tensor:
    """Per-element negative log-likelihood of a location-scale Student-t (robust to reset-artifact outliers)."""
    z2 = ((target - loc) / log_scale.exp()) ** 2
    const = math.lgamma((dof + 1) / 2) - math.lgamma(dof / 2) - 0.5 * math.log(dof * math.pi)
    return -const + log_scale + 0.5 * (dof + 1) * torch.log1p(z2 / dof)


class GapEnsemble(nn.Module):
    def __init__(self, x_names: list[str], y_names: list[str], num_members: int = 5,
                 hidden: tuple[int, ...] = (256, 256, 256), dof: float = 4.0):
        super().__init__()
        self.x_names, self.y_names, self.hidden, self.dof = list(x_names), list(y_names), tuple(hidden), float(dof)
        self.members = nn.ModuleList(GapMember(len(x_names), len(y_names), hidden) for _ in range(num_members))
        for name, dim in (("x_mean", len(x_names)), ("x_std", len(x_names)), ("y_mean", len(y_names)), ("y_std", len(y_names))):
            self.register_buffer(name, torch.zeros(dim) if "mean" in name else torch.ones(dim))
        self.register_buffer("temperature", torch.ones(len(y_names)))  # post-hoc variance calibration (Step 4)

    def set_normalization(self, x: torch.Tensor, y: torch.Tensor):
        self.x_mean.copy_(x.mean(0)); self.x_std.copy_(x.std(0).clamp_min(1e-6))
        self.y_mean.copy_(y.mean(0)); self.y_std.copy_(y.std(0).clamp_min(1e-6))

    def normalize_x(self, x):
        return (x - self.x_mean) / self.x_std

    def normalize_y(self, y):
        return (y - self.y_mean) / self.y_std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Standardized member locations and log-scales, each [K, N, D]."""
        xn = self.normalize_x(x)
        outs = [m(xn) for m in self.members]
        return torch.stack([o[0] for o in outs]), torch.stack([o[1] for o in outs])

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Mean prediction in physical units plus uncertainty (standardized units unless stated)."""
        loc, log_scale = self(x)
        variance = (2 * log_scale).exp() * (self.dof / (self.dof - 2)) * self.temperature  # Student-t variance
        epistemic = loc.var(0, unbiased=False)
        aleatoric = variance.mean(0)
        mean_n = loc.mean(0)
        return dict(mean=mean_n * self.y_std + self.y_mean, mean_std_units=mean_n,
                    epistemic=epistemic, aleatoric=aleatoric, total=epistemic + aleatoric,
                    u=epistemic.mean(-1).sqrt())

    def config(self) -> dict:
        return dict(x_names=self.x_names, y_names=self.y_names, num_members=len(self.members),
                    hidden=list(self.hidden), dof=self.dof)

    def save(self, path: str | Path, extra: dict | None = None):
        torch.save(dict(config=self.config(), state_dict=self.state_dict(), extra=extra or {}), path)

    @classmethod
    def load(cls, path: str | Path, map_location="cpu", expected_x_names: list[str] | None = None) -> "GapEnsemble":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        cfg = payload["config"]
        if expected_x_names is not None and list(expected_x_names) != cfg["x_names"]:
            raise ValueError("Gap model input feature layout does not match the provided observation layout")
        model = cls(cfg["x_names"], cfg["y_names"], cfg["num_members"], tuple(cfg["hidden"]), cfg["dof"])
        model.load_state_dict(payload["state_dict"])
        return model.eval()
