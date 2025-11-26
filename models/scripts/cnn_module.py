import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.fasta_utils import REV_COMP

class RCFirstConv1d(nn.Module):
    
    def __init__(self, out_channels, kernel_size=15, dilation=1, bias=True, dropout=0.1):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(5, out_channels, kernel_size, padding=pad, dilation=dilation, bias=bias)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        y1 = self.conv(x)             # (B, C, L)
        
        x_rc = x.flip(-1)[:, REV_COMP.to(x.device), :]
        y2 = self.conv(x_rc)          # (B, C, L)
        y2 = y2.flip(-1)
        
        y = torch.max(y1, y2)
        y = self.batch_norm(y)
        y = F.relu(y)
        y = self.dropout(y)
        
        return y 

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=9, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=pad, dilation=dilation)
        self.batch_norm = nn.BatchNorm1d(c_out)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(F.relu(self.batch_norm(self.conv(x))))

class MaskedMaxPool1d(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x, mask):  # x:(B, C, L), mask:(B, L)
        m = mask.unsqueeze(1)
        x_masked = x.masked_fill(~m, float('-inf'))
        y = F.max_pool1d(x_masked, self.kernel_size, self.stride)
        m_pooled = F.max_pool1d(m.float(), self.kernel_size, self.stride) > 0
        y = y.masked_fill(~m_pooled, 0.0)
        return y, m_pooled.squeeze(1)

class RCInputInvariantCNN(nn.Module):
    def __init__(self, width=64, motif_kernel = (9, 15, 21, ), context_kernel = 9, context_dilation = (1, 2), dropout=0.1):
        super().__init__()

        self.motif_conv_blocks = nn.ModuleList(
            RCFirstConv1d(width, kernel_size=ker, dilation=1, dropout=0.1)
            for ker in motif_kernel
        )
        dim = width * len(motif_kernel)

        self.context_blocks = nn.ModuleList()
        c_in = dim
        for dil in context_dilation:
            self.context_blocks.append(
                ConvBlock(c_in, width, kernel_size=context_kernel, dilation=dil, dropout=dropout)
            )
            c_in = width

        self.pool = MaskedMaxPool1d(kernel_size=2, stride=2)
        head_in_features = c_in

        self.head = nn.Sequential(
            nn.Linear(head_in_features, width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(width, 1),
        )

    @staticmethod
    def masked_avg_pool(z, mask):
        if mask is None:
            return z.mean(-1)
        m = mask.unsqueeze(1).float()
        return (z * m).sum(-1) / m.sum(-1).clamp_min(1.0)

    def forward(self, x, mask):
        motif_outputs = [motif_conv(x) for motif_conv in self.motif_conv_blocks]
        z = torch.cat(motif_outputs, dim=1)

        for block in self.context_blocks:
            z = block(z)
            z, mask = self.pool(z, mask)

        z_pooled_avg = self.masked_avg_pool(z, mask)
        out = self.head(z_pooled_avg).squeeze(-1)
        
        # branch_outputs = []
        # for block in self.context_blocks:
            # zb = block(z)
            # zb_pooled, mask_pooled = MaskedMaxPool1d(kernel_size=2, stride=2)(zb, mask)
            # zb_pooled_avg = self.masked_avg_pool(zb_pooled, mask_pooled)
            
            # z_pooled, mask_pooled = MaskedMaxPool1d(kernel_size=2, stride=2)(z, mask)
            # z_pooled_avg = self.masked_avg_pool(z_pooled, mask_pooled)
            # branch_outputs.append(zb_pooled_avg + z_pooled_avg)
        
        # z_concat = torch.cat(branch_outputs, dim=-1)
        # out = self.head(z_concat).squeeze(-1)
        return out