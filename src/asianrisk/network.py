"""A compact MLP for price / strike, trained with or without delta supervision.

Inputs are the unit-cube coordinates of (log_moneyness, vol, sqrt(T)). The net
returns y = price / K, and its slope with respect to the first input gives delta:
    delta = dy/dS at K = 1 = dy/dx * dx/dS = dy/du / (x_high - x_low) * exp(-x)
because x = ln(S / K) and u = (x - x_low) / (x_high - x_low).

The slope is propagated forward by hand next to the activations. That costs two
forward passes rather than one forward and one backward, needs no autograd at
inference, and stays differentiable with respect to the weights in training.
Softplus keeps the slope smooth, unlike ReLU.
"""

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from asianrisk.domain import Domain

PRICE_SCALE = 1e-4  # one basis point of strike, the training unit for price


@dataclass(frozen=True)
class TrainConfig:
    width: int = 64
    depth: int = 3
    epochs: int = 300
    batch_size: int = 4096
    learning_rate: float = 3e-3
    delta_weight: float = 1.0  # 0 gives price-only training
    delta_scale: float = 1e-3  # delta error that counts the same as one price unit
    seed: int = 0


class PriceNet(nn.Module):
    """Inputs are centered to [-1, 1] and the output is scaled to the target's spread.

    Without the output scale, targets of thousands of basis points leave the
    last layer far from its starting scale and training stalls near 50 bp.
    """

    def __init__(self, width: int, depth: int, output_scale: float = 1.0):
        super().__init__()
        sizes = [3] + [width] * depth
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.output = nn.Linear(width, 1)
        self.register_buffer("output_scale", torch.tensor(output_scale, dtype=torch.float64))

    def forward(self, unit_inputs):
        """Return y and dy/du_0 together."""
        value = 2.0 * unit_inputs - 1.0
        slope = torch.zeros_like(unit_inputs)
        slope[:, 0] = 2.0
        for layer in self.hidden:
            activation = layer(value)
            # d softplus(a) / da = sigmoid(a). A tangent passes through weights only.
            slope = torch.sigmoid(activation) * (slope @ layer.weight.T)
            value = nn.functional.softplus(activation)
        scale = self.output_scale
        return scale * self.output(value)[:, 0], scale * (slope @ self.output.weight.T)[:, 0]


@dataclass
class NeuralSurface:
    """y(points) = base(points) + PRICE_SCALE * net(unit coordinates)."""

    domain: Domain
    net: PriceNet
    base: object = None  # callable(points) -> (price, delta), or None
    threads: int | None = None

    def _delta_factor(self, points):
        low, high = self.domain.log_moneyness
        return PRICE_SCALE / (high - low) * np.exp(-points[:, 0])

    @torch.no_grad()
    def __call__(self, points):
        unit = torch.as_tensor(self.domain.to_unit(points), dtype=torch.float64)
        value, slope = self.net(unit)
        price = PRICE_SCALE * value.numpy()
        delta = self._delta_factor(points) * slope.numpy()
        if self.base is not None:
            base_price, base_delta = self.base(points)
            price, delta = price + base_price, delta + base_delta
        return price, delta


def train_surface(domain, points, price, delta, config: TrainConfig, base=None):
    """Fit one network. Returns the surface and a training record."""
    torch.manual_seed(config.seed)
    started = time.perf_counter()
    base_price, base_delta = base(points) if base is not None else (0.0, 0.0)
    output_scale = float(np.std(price - base_price)) / PRICE_SCALE
    net = PriceNet(config.width, config.depth, output_scale).to(torch.float64)

    unit = torch.as_tensor(domain.to_unit(points), dtype=torch.float64)
    price_target = torch.as_tensor((price - base_price) / PRICE_SCALE, dtype=torch.float64)
    delta_residual = delta - base_delta
    low, high = domain.log_moneyness
    # Train on delta error in delta units, converted from the slope at each point.
    to_delta = torch.as_tensor(
        PRICE_SCALE / (high - low) * np.exp(-points[:, 0]), dtype=torch.float64
    )
    delta_target = torch.as_tensor(delta_residual, dtype=torch.float64)

    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    steps_per_epoch = max(len(points) // config.batch_size, 1)
    total_steps = config.epochs * steps_per_epoch
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=1e-6)
    generator = torch.Generator().manual_seed(config.seed)

    history = []
    for epoch in range(config.epochs):
        order = torch.randperm(len(points), generator=generator)
        for batch in order[: steps_per_epoch * config.batch_size].split(config.batch_size):
            value, slope = net(unit[batch])
            loss = torch.mean((value - price_target[batch]) ** 2)
            if config.delta_weight > 0:
                delta_error = to_delta[batch] * slope - delta_target[batch]
                loss = loss + config.delta_weight * torch.mean(
                    (delta_error / config.delta_scale) ** 2
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
        history.append(loss.item())

    record = {
        **asdict(config),
        "n_points": len(points),
        "parameters": sum(p.numel() for p in net.parameters()),
        "wall_seconds": time.perf_counter() - started,
        "final_batch_loss": history[-1],
    }
    return NeuralSurface(domain, net.eval(), base), record
