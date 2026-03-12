import torch
import math
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional
from torch_geometric.graphgym.config import cfg


class ExpNormalSmearing(nn.Module):
    def __init__(
        self,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        num_rbf=50,
        trainable=True,
        dtype=torch.float32,
    ):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.dtype = dtype
        self.cutoff_fn = CosineCutoff(0, cutoff_upper, newEnvelope=False)  # Only the "cosine function" is used here
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)  # scaling factor

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(
            torch.scalar_tensor(
                -self.cutoff_upper + self.cutoff_lower, dtype=self.dtype
            )
        )
        means = torch.linspace(start_value, 1, self.num_rbf, dtype=self.dtype)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf,
            dtype=self.dtype,
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


class CosineCutoff(nn.Module):  # Implementation of distance weight decay function
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, newEnvelope=True):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.newEnvelope = newEnvelope

    def forward(self, distances: Tensor) -> Tensor:
        if self.newEnvelope:
            if cfg.envelope_type == 'simply':
                cutoffs = torch.where(distances <= 1.0, torch.ones_like(distances), 1.0 - 0.2 * distances)

            elif cfg.envelope_type == 'cubic':
                r0 = 1.0
                r1 = 5.0
                s = ((distances - r0) / (r1 - r0)).clamp(min=0.0, max=1.0)  # normalized in [0,1]

                c = 0.6
                t = s.pow(c)
                mid = 1.0 - 3.0 * t**2 + 2.0 * t**3  # power Cubic smoothstep

                cutoffs = torch.where(distances <= r0,  # condition, true or false
                                  torch.ones_like(distances, dtype=torch.float),  # x, if true
                                  torch.where(distances >= r1, torch.zeros_like(distances), mid))  # y, if false
            else:
                raise NotImplementedError("Envelope type not implemented.")
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)

        # remove contributions beyond the cutoff radius
        cutoffs = cutoffs * (distances < self.cutoff_upper)
        return cutoffs


class RBF2Electronegativity(nn.Module):
    def __init__(
        self,
        cutoff_upper=2.14,
        type='newRBF',
        trainable=True,
        dtype=torch.float32,
    ):
        super(RBF2Electronegativity, self).__init__()
        self.cutoff_upper = cutoff_upper
        self.type = type
        if type == 'newRBF':
            self.num_rbf = 20
        else:
            self.num_rbf = 64
        self.trainable = trainable
        self.dtype = dtype
        self.cutoff_fn = CosineCutoff_eNeg(cutoff_upper)
        self.alpha = 2.14 / (cutoff_upper - 0)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        if self.type == 'newRBF' and self.num_rbf == 20 or self.type == 'newRBF04' or self.type == 'newRBF05':
            start_value = torch.exp(
                torch.scalar_tensor(
                    -self.cutoff_upper + 0, dtype=self.dtype
                )
            )
            means = torch.linspace(start_value, 1, self.num_rbf, dtype=self.dtype)
            betas = torch.tensor(
                [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf,
                dtype=self.dtype,
            )
        elif self.type == 'newRBF02' or self.type == 'newRBF03':
            means = torch.linspace(0, self.cutoff_upper, self.num_rbf, dtype=self.dtype)
            lengthscale = np.diff(means).mean()
            betas = torch.tensor(
                1 / (lengthscale ** 2),
                dtype=self.dtype,
            )
        else:
            raise NotImplementedError(f"Unknown RBF type: {self.type}")
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, eNeg):
        if self.type == 'newRBF':
            return self.cutoff_fn(eNeg) * torch.exp(-self.betas * (torch.exp(self.alpha * (-eNeg)) - self.means) ** 2)
        elif self.type == 'newRBF02':
            return self.cutoff_fn(eNeg) * torch.exp(-self.betas * (self.alpha * (eNeg) - self.means) ** 2)
        elif self.type == 'newRBF03':
            return torch.exp(-self.betas * (self.alpha * (eNeg) - self.means) ** 2)
        elif self.type == 'newRBF04':
            return torch.exp(-self.betas * (torch.exp(self.alpha * (-eNeg)) - self.means) ** 2)
        elif self.type == 'newRBF05':
            return self.cutoff_fn(eNeg) * torch.exp(-self.betas * (torch.exp(self.alpha * (-eNeg)) - self.means) ** 2)
        else:
            raise NotImplementedError(f"Unknown RBF type: {self.type}")


class CosineCutoff_eNeg(nn.Module):
    def __init__(self, cutoff_upper=2.14):
        super(CosineCutoff_eNeg, self).__init__()
        self.cutoff_upper = cutoff_upper
    def forward(self, eNeg: Tensor) -> Tensor:
        cutoffs = 0.5 * (1.0 - torch.cos(eNeg * math.pi / self.cutoff_upper))
        return cutoffs
