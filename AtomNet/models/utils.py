import torch
import math
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional
from torch_geometric.graphgym.config import cfg

"""
This module is primarily used to convert the input distance (dist) into a radial basis function (RBF) based on 
an exponential normal distribution.
"""


class ExpNormalSmearing(nn.Module):
    def __init__(
        self,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        num_rbf=50,
        trainable=True,  # 实际输入：False
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

        """
        调用 _initial_params 方法生成初始的 means（均值）和 betas（宽度参数），用于后续的径向基函数计算。
            如果 trainable=True：
            将 means 和 betas 注册为可训练参数（nn.Parameter），允许在训练中更新。
            如果 trainable=False：
            将 means 和 betas 注册为缓冲区（register_buffer），这些值不会被优化，但会随模型保存和加载。
        """
        means, betas = self._initial_params()
        if trainable:  # 可训练参数
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:  # 固定值
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(  # 计算一个起始值，用于初始化 means
            torch.scalar_tensor(
                -self.cutoff_upper + self.cutoff_lower, dtype=self.dtype
            )
        )
        # 使用 torch.linspace 在 start_value 到 1 之间生成 num_rbf 个均匀分布的值。这些值表示径向基函数的中心点。
        means = torch.linspace(start_value, 1, self.num_rbf, dtype=self.dtype)
        # betas 用于控制径向基函数的宽度。
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf,
            dtype=self.dtype,
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):  # 输出形状通常为 [batch_size, num_rbf]，表示每个输入距离被扩展为 num_rbf 个特征。
        # 将输入距离张量增加一个维度（如从 [batch_size] 变为 [batch_size, 1]），以便与 means 和 betas（形状为 [num_rbf]）广播计算。
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


class CosineCutoff(nn.Module):  # 距离权重衰减函数实现
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, newEnvelope=True):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.newEnvelope = newEnvelope

    def forward(self, distances: Tensor) -> Tensor:
        if self.newEnvelope:
            """ 分段函数
            ⭐ Instruction: 在使用 布尔索引（mask）+ 原地赋值（in-place assignment） 来修改 cutoffs，这会让 PyTorch 认为 cutoffs 
            不是通过可微操作从 distances 得到的。故，cutoffs.backward() 不会将梯度传回到 distances。
            
            # 初始化输出张量，与输入 distances 形状相同
            cutoffs = torch.zeros_like(distances, dtype=torch.float)

            # 条件 1: x <= 1.0, f(x) = 1
            mask1 = distances <= 1.0
            cutoffs[mask1] = 1.0

            # 条件 2: x > 1.0, f(x) = 1.0 - 0.2x
            mask2 = distances > 1.0
            cutoffs[mask2] = 1.0 - 0.2 * distances[mask2] """

            if cfg.envelope_type == 'simply':
                # 新的写法，避免 in-place 操作（原地修改）破坏 autograd 计算图（梯度无法回传）。
                cutoffs = torch.where(distances <= 1.0, torch.ones_like(distances), 1.0 - 0.2 * distances)

            elif cfg.envelope_type == 'cubic':
                """ Cubic smoothstep: 平滑的多项式 """
                r0 = 1.0
                r1 = 5.0
                s = ((distances - r0) / (r1 - r0)).clamp(min=0.0, max=1.0)  # normalized in [0,1]
                # cubic: 1 - 3 s^2 + 2 s^3  (decreasing from 1->0)

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
        trainable=True,  # 实际输入：False
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
        self.alpha = 2.14 / (cutoff_upper - 0)  # 缩放因子

        """
        调用 _initial_params 方法生成初始的 means（均值）和 betas（宽度参数），用于后续的径向基函数计算。
            如果 trainable=True：
            将 means 和 betas 注册为可训练参数（nn.Parameter），允许在训练中更新。
            如果 trainable=False：
            将 means 和 betas 注册为缓冲区（register_buffer），这些值不会被优化，但会随模型保存和加载。
        """
        means, betas = self._initial_params()
        if trainable:  # 可训练参数
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:  # 固定值
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        if self.type == 'newRBF' and self.num_rbf == 20 or self.type == 'newRBF04' or self.type == 'newRBF05':
            """ newRBF && newRBF04 && newRBF05 """
            # initialize means and betas according to the default values in PhysNet
            # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
            start_value = torch.exp(  # 计算一个起始值，用于初始化 means
                torch.scalar_tensor(
                    -self.cutoff_upper + 0, dtype=self.dtype
                )
            )
            # 使用 torch.linspace 在 start_value 到 1 之间生成 num_rbf 个均匀分布的值。这些值表示径向基函数的中心点。
            means = torch.linspace(start_value, 1, self.num_rbf, dtype=self.dtype)
            # betas 用于控制径向基函数的宽度。
            betas = torch.tensor(
                [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf,
                dtype=self.dtype,
            )
        elif self.type == 'newRBF02' or self.type == 'newRBF03':
            """ newRBF02 && newRBF03 """
            means = torch.linspace(0, self.cutoff_upper, self.num_rbf, dtype=self.dtype)
            lengthscale = np.diff(means).mean()
            # betas 用于控制径向基函数的宽度。
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

    def forward(self, eNeg):  # 输出形状通常为 [batch_size, num_rbf]，表示 每个电负性差值的绝对值 被扩展为 num_rbf 个特征。
        # eNeg = eNeg.unsqueeze(-1)  # eNeg 本身的形状就是 [batch_size, 1]
        if self.type == 'newRBF':
            return self.cutoff_fn(eNeg) * torch.exp(-self.betas * (torch.exp(self.alpha * (-eNeg)) - self.means) ** 2)  # newRBF
        elif self.type == 'newRBF02':
            return self.cutoff_fn(eNeg) * torch.exp(-self.betas * (self.alpha * (eNeg) - self.means) ** 2)  # newRBF02
        elif self.type == 'newRBF03':
            return torch.exp(-self.betas * (self.alpha * (eNeg) - self.means) ** 2)  # newRBF03
        elif self.type == 'newRBF04':
            return torch.exp(-self.betas * (torch.exp(self.alpha * (-eNeg)) - self.means) ** 2)  # newRBF04
        elif self.type == 'newRBF05':  # 与 newRBF 类似，但特征维度不同(newRBF-20维, newRBF05-64维)
            return self.cutoff_fn(eNeg) * torch.exp(-self.betas * (torch.exp(self.alpha * (-eNeg)) - self.means) ** 2)  # newRBF05
        else:
            raise NotImplementedError(f"Unknown RBF type: {self.type}")


class CosineCutoff_eNeg(nn.Module):
    def __init__(self, cutoff_upper=2.14):
        super(CosineCutoff_eNeg, self).__init__()
        self.cutoff_upper = cutoff_upper
    def forward(self, eNeg: Tensor) -> Tensor:
        cutoffs = 0.5 * (1.0 - torch.cos(eNeg * math.pi / self.cutoff_upper))
        return cutoffs
