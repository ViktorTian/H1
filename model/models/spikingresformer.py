import math
import torch
import torch.nn as nn
from .submodules.layers import Conv3x3, Conv1x1, LIFWithTiming, LIF, PLIF, BN, Linear, SpikingMatmul
from spikingjelly.activation_based import layer
from typing import Any, List, Mapping, Optional
from timm.models.registry import register_model


class AdaptiveSSDPModule(nn.Module):
    """
    - epoch < warmup_epochs: 只记录同步度 S 与 loss，不改权重，返回 0 Δw
    - 首次跨过 warm-up 自动拟合 gate_k, μ_S, σ_S
    - 之后 Δw 乘 gate_b = 1 + k (S-μ)/σ
    """
    def __init__(self, input_dim, output_dim, device,
                 warmup_epochs: int = 100,
                 A_plus: float = 1.5e-3,
                 A_baseline: float = 5e-4,
                 sigma: float = 1.0):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.device     = device

        # 可训练核参数
        self.A_plus     = nn.Parameter(torch.tensor(A_plus , device=device))
        self.A_baseline = nn.Parameter(torch.tensor(A_baseline, device=device))
        self.sigma      = nn.Parameter(torch.tensor(sigma , device=device))

        # warm-up
        self.warmup_epochs = warmup_epochs
        self.record_S: List[torch.Tensor] = []
        self.record_L: List[torch.Tensor] = []

        # 拟合后门控参数
        self.gate_k  : Optional[torch.Tensor] = None
        self.S_mean  : Optional[torch.Tensor] = None
        self.S_std   : Optional[torch.Tensor] = None

    # -------- util 函数 --------
    @staticmethod
    def _batch_sync(pre_spike: torch.Tensor, post_spike: torch.Tensor) -> torch.Tensor:
        # pre:[B,Cin] post:[B,Cout] → S_b = mean(sync) over Cout×Cin
        sync = post_spike.unsqueeze(-1) * pre_spike.unsqueeze(1)  # [B,Cout,Cin]
        return sync.mean(dim=(1,2))                               # [B]

    def compute_pot_dep(self,
                        pre_spike: torch.Tensor,
                        post_spike: torch.Tensor,
                        delta_t: torch.Tensor):
        """返回 (pot-dep, batch_sync_S)"""
        sync  = post_spike.unsqueeze(-1) * pre_spike.unsqueeze(1)     # [B,Cout,Cin]
        gauss = torch.exp(-(delta_t**2) / (2 * self.sigma**2))
        pot   = self.A_plus     * sync        * gauss
        dep   = self.A_baseline * (1 - sync)  * gauss
        return pot - dep, sync.mean(dim=(1,2))                        # ([B,Cout,Cin], [B])

    def _fit_gate(self):
        # 若记录为空，使用默认门控=1
        if len(self.record_S) == 0 or len(self.record_L) == 0:
            self.gate_k  = torch.tensor(0.0, device=self.device)
            self.S_mean  = torch.tensor(0.0, device=self.device)
            self.S_std   = torch.tensor(1.0, device=self.device)
            return
        S = torch.cat(self.record_S)            # [N]
        L = torch.cat(self.record_L)            # [N]
        self.S_mean = S.mean()
        self.S_std  = S.std(unbiased=False) + 1e-8
        S_n = (S - self.S_mean) / self.S_std
        L_n = (L - L.mean()) / (L.std(unbiased=False) + 1e-8)
        self.gate_k = - (S_n * L_n).mean().detach()   # 负相关 → 正门控
        self.record_S.clear(); self.record_L.clear()

    # -------- forward --------
    def forward(self,
                pre_spike: torch.Tensor,
                post_spike: torch.Tensor,
                delta_t: torch.Tensor,
                loss: Optional[torch.Tensor] = None,
                epoch: Optional[int]        = None) -> torch.Tensor:
        """
        返回 Δw:[Cout,Cin]；warm-up 阶段返回全 0
        """
        pd, S = self.compute_pot_dep(pre_spike, post_spike, delta_t)   # [B,Cout,Cin], [B]

        # -------- warm-up ----------
        if epoch is not None and epoch < self.warmup_epochs:
            if loss is not None:
                with torch.no_grad():
                    self.record_S.append(S.detach().cpu())
                    self.record_L.append(loss.detach().repeat(S.size(0)).cpu())
            # 返回 0 Δw（形状需匹配 weight）
            return torch.zeros(self.output_dim, self.input_dim,
                               device=self.device, dtype=pd.dtype)

        # -------- 首次拟合 gate ----------
        if self.gate_k is None:
            self._fit_gate()

        gate = 1.0 + self.gate_k * (S - self.S_mean) / self.S_std
        gate = gate.clamp(0.0, 2.0).view(-1,1,1)                       # [B,1,1]
        delta_w = (pd * gate).mean(dim=0).clamp(-1.0, 1.0)             # [Cout,Cin]
        return delta_w


class GWFFN(nn.Module):
    def __init__(self, in_channels, num_conv=1, ratio=4, group_size=64, activation=LIF):
        super().__init__()
        inner_channels = in_channels * ratio
        self.up = nn.Sequential(
            activation(),
            Conv1x1(in_channels, inner_channels),
            BN(inner_channels),
        )
        self.conv = nn.ModuleList()
        for _ in range(num_conv):
            self.conv.append(
                nn.Sequential(
                    activation(),
                    Conv3x3(inner_channels, inner_channels, groups=inner_channels // group_size),
                    BN(inner_channels),
                ))
        self.down = nn.Sequential(
            activation(),
            Conv1x1(inner_channels, in_channels),
            BN(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_feat_out = x.clone()
        x = self.up(x)
        x_feat_in = x.clone()
        for m in self.conv:
            x = m(x)
        x = x + x_feat_in
        x = self.down(x)
        x = x + x_feat_out
        return x


# =========== 3.1 新增 DSSAWithSSDP 类, 用于替换最后 stage 的 DSSA ===========
class DSSA(nn.Module):
    def __init__(self, dim, num_heads, lenth, patch_size, activation=LIF):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.lenth = lenth
        self.register_buffer('firing_rate_x', torch.zeros(1, 1, num_heads, 1, 1))
        self.register_buffer('firing_rate_attn', torch.zeros(1, 1, num_heads, 1, 1))
        self.init_firing_rate_x = False
        self.init_firing_rate_attn = False
        self.momentum = 0.999

        self.activation_in = activation()
        self.W = layer.Conv2d(dim, dim * 2, patch_size, patch_size, bias=False, step_mode='m')
        self.norm = BN(dim * 2)
        self.matmul1 = SpikingMatmul('r')
        self.matmul2 = SpikingMatmul('r')
        self.activation_attn = activation()
        self.activation_out = activation()

        self.Wproj = Conv1x1(dim, dim)
        self.norm_proj = BN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T,B,C,H,W = x.shape
        x_feat = x.clone()
        x = self.activation_in(x)

        y = self.W(x)  # [T, B, 2*C, H', W'] 其中H'=H/patch_size, W'=W/patch_size
        y = self.norm(y)
        y = y.reshape(T, B, self.num_heads, 2*(C//self.num_heads), -1)
        y1, y2 = y[:, :, :, :C//self.num_heads, :], y[:, :, :, C//self.num_heads:, :]
        x = x.reshape(T,B,self.num_heads,C//self.num_heads,-1)

        if self.training:
            firing_rate_x = x.detach().mean((0,1,3,4),keepdim=True)
            if not self.init_firing_rate_x and torch.all(self.firing_rate_x==0):
                self.firing_rate_x = firing_rate_x
            self.init_firing_rate_x = True
            self.firing_rate_x = self.firing_rate_x*self.momentum + firing_rate_x*(1-self.momentum)

        scale1 = 1./torch.sqrt(self.firing_rate_x*(self.dim//self.num_heads))
        attn = self.matmul1(y1.transpose(-1,-2), x)
        attn = attn * scale1
        attn = self.activation_attn(attn)

        if self.training:
            firing_rate_attn = attn.detach().mean((0,1,3,4),keepdim=True)
            if not self.init_firing_rate_attn and torch.all(self.firing_rate_attn==0):
                self.firing_rate_attn = firing_rate_attn
            self.init_firing_rate_attn = True
            self.firing_rate_attn = self.firing_rate_attn*self.momentum + firing_rate_attn*(1-self.momentum)

        scale2 = 1./torch.sqrt(self.firing_rate_attn*self.lenth)
        out = self.matmul2(y2, attn)
        out = out*scale2
        out = out.reshape(T,B,C,H,W)
        out = self.activation_out(out)

        out = self.Wproj(out)
        out = self.norm_proj(out)
        out = out + x_feat
        return out


class DSSAWithSSDP(DSSA):
    """
    用于在 forward() 中记录 x_in, x_out，以便训练时做 SSDP。
    这里只演示把 SSDP 应用到 self.Wproj 的权重上(形状[C, C, 1, 1])，相对容易映射到 [C_out, C_in]。
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 记录输入（在你要应用 SSDP 的权重对应的输入，就是 x）
        self.x_in = x.detach().clone()  # [T,B,C,H,W]
        # 原本的 forward 计算
        out = super().forward(x)
        # 记录输出
        self.x_out = out.clone()  # [T,B,C,H,W]
        return out


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, activation=LIF) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, out_channels, stride=stride)
        self.norm = BN(out_channels)
        self.activation = activation()

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.activation(x)
        x = self.conv(x)
        x = self.norm(x)
        return x


# SpikingResformer主体
class SpikingResformer(nn.Module):
    def __init__(
        self,
        layers: List[List[str]],
        planes: List[int],
        num_heads: List[int],
        patch_sizes: List[int],
        img_size=224,
        T=4,
        in_channels=3,
        num_classes=1000,
        prologue=None,
        group_size=64,
        activation=LIF,
        # start_epoch=0,
        device='cuda',
        **kwargs,
    ):
        super().__init__()
        self.T = T
        self.skip = ['prologue.0', 'classifier']
        assert len(planes)==len(layers)==len(num_heads)==len(patch_sizes)
        self.current_epoch=0

        if prologue is None:
            self.prologue=nn.Sequential(
                layer.Conv2d(in_channels, planes[0], 7, 2, 3, bias=False, step_mode='m'),
                BN(planes[0]),
                layer.MaxPool2d(kernel_size=3, stride=2, padding=1, step_mode='m'),
            )
            img_size=img_size//4
        else:
            self.prologue=prologue

        self.layers=nn.Sequential()
        for idx in range(len(planes)):
            sub_layers=nn.Sequential()
            if idx!=0:
                sub_layers.append(
                    DownsampleLayer(planes[idx-1], planes[idx], stride=2, activation=activation)
                )
                img_size=img_size//2

            # ----------> MODIFICATION：在最后一个stage用DSSAWithSSDP替换DSSA
            for name in layers[idx]:
                if name=='DSSA':
                    if idx == len(planes)-1:
                        # 最后一层stage，用 DSSAWithSSDP
                        sub_layers.append(
                            DSSAWithSSDP(planes[idx], num_heads[idx],
                                         (img_size//patch_sizes[idx])**2,
                                         patch_sizes[idx],
                                         activation=activation)
                        )
                    else:
                        # 其余stage使用原DSSA
                        sub_layers.append(
                            DSSA(planes[idx], num_heads[idx],
                                 (img_size//patch_sizes[idx])**2,
                                 patch_sizes[idx],
                                 activation=activation)
                        )
                elif name=='GWFFN':
                    sub_layers.append(GWFFN(planes[idx],
                                            group_size=group_size,
                                            activation=activation))
                else:
                    raise ValueError(name)
            self.layers.append(sub_layers)

        self.avgpool=layer.AdaptiveAvgPool2d((1,1),step_mode='m')
        self.classifier=Linear(planes[-1],num_classes)

        # SSDPModule用于classifier
        self.ssdp = AdaptiveSSDPModule(
            input_dim=planes[-1],
            output_dim=num_classes,
            device=device)  # e.g. 前 1/3 epochs warm-up

        # ----------> MODIFICATION：额外实例化一个ssdp_dssa，用于最后一个stage的DSSA
        self.ssdp_dssa = AdaptiveSSDPModule(
            input_dim=planes[-1],
            output_dim=planes[-1],
            device=device)

        # self.start_epoch=start_epoch
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m,(nn.Linear,nn.Conv2d)):
                nn.init.trunc_normal_(m.weight,std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def transfer(self,state_dict:Mapping[str,Any]):
        _state_dict={k:v for k,v in state_dict.items() if 'classifier' not in k}
        return self.load_state_dict(_state_dict,strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            assert x.dim() == 5
        else:
            x = x.transpose(0, 1)
        x = self.prologue(x)
        x = self.layers(x)
        x = self.avgpool(x)  # [T,B,C,1,1]
        T, B, C, H, W = x.shape

        x = x.view(T, B, C)
        pre_spike_seq = (x > 0).float()

        output_all = []
        for t in range(T):
            output_t = self.classifier(x[t])  # [B,C_out]
            output_all.append(output_t)
        output_all = torch.stack(output_all, dim=0)  # [T,B,C_out]

        post_spike_seq = (output_all > 0).float()

        # 找到首次发放时间
        pre_spike_exist = (pre_spike_seq > 0)
        post_spike_exist = (post_spike_seq > 0)
        no_spike_pre = pre_spike_exist.sum(0) == 0
        no_spike_post = post_spike_exist.sum(0) == 0
        t_pre = pre_spike_exist.float().cumsum(0).argmax(0).float()
        t_post = post_spike_exist.float().cumsum(0).argmax(0).float()
        t_pre[no_spike_pre] = float(self.T)
        t_post[no_spike_post] = float(self.T)

        # ---- 必须写回的 3 个属性 ----
        self.t_pre = t_pre.detach()  # [B,C_in]
        self.t_post = t_post.detach()  # [B,C_out]
        self.out = x.mean(dim=0).detach()  # [B,C_in] — 用于 SSDP

        output = output_all.mean(dim=0)  # [B,C_out]
        return output

    def no_weight_decay(self):
        ret=set()
        for name,module in self.named_modules():
            if isinstance(module,PLIF):
                ret.add(name+'.w')
        return ret

@register_model
def spikingresformer_ti(**kwargs):
    return SpikingResformer(
        [
            ['DSSA', 'GWFFN'] * 1,
            ['DSSA', 'GWFFN'] * 2,
            ['DSSA', 'GWFFN'] * 3, ],
        [64, 192, 384],
        [1, 3, 6],
        [4, 2, 1],
        in_channels=3,
        **kwargs,
    )

@register_model
def spikingresformer_s(**kwargs):
    return SpikingResformer(
        [
            ['DSSA', 'GWFFN'] * 1,
            ['DSSA', 'GWFFN'] * 2,
            ['DSSA', 'GWFFN'] * 3, ],
        [64, 256, 512],
        [1, 4, 8],
        [4, 2, 1],
        in_channels=3,
        **kwargs,
    )

@register_model
def spikingresformer_m(**kwargs):
    return SpikingResformer(
        [
            ['DSSA', 'GWFFN'] * 1,
            ['DSSA', 'GWFFN'] * 2,
            ['DSSA', 'GWFFN'] * 3, ],
        [64, 384, 768],
        [1, 6, 12],
        [4, 2, 1],
        in_channels=3,
        **kwargs,
    )

@register_model
def spikingresformer_l(**kwargs):
    return SpikingResformer(
        [
            ['DSSA', 'GWFFN'] * 1,
            ['DSSA', 'GWFFN'] * 2,
            ['DSSA', 'GWFFN'] * 3, ],
        [128, 512, 1024],
        [2, 8, 16],
        [4, 2, 1],
        in_channels=3,
        **kwargs,
    )

@register_model
def spikingresformer_dvsg(**kwargs):
    return SpikingResformer(
        [
            ['DSSA', 'GWFFN'] * 1,
            ['DSSA', 'GWFFN'] * 2,
            ['DSSA', 'GWFFN'] * 3, ],
        [32, 96, 192],
        [1, 3, 6],
        [4, 2, 1],
        in_channels=3,
        prologue=nn.Sequential(
            layer.Conv2d(3, 32, 3, 1, 1, bias=False, step_mode='m'),
            BN(32),
        ),
        group_size=32,
        activation=PLIF,
        **kwargs,
    )

@register_model
def spikingresformer_cifar(**kwargs):
    return SpikingResformer(
        [
            ['DSSA', 'GWFFN'] * 1,
            ['DSSA', 'GWFFN'] * 2,
            ['DSSA', 'GWFFN'] * 3, ],
        [64, 192, 384],
        [1, 3, 6],
        [4, 2, 1],
        in_channels=3,
        prologue=nn.Sequential(
            layer.Conv2d(3, 64, 3, 1, 1, bias=False, step_mode='m'),
            BN(64),
        ),
        **kwargs,
    )
