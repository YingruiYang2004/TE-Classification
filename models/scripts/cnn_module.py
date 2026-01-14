import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.fasta_utils import REV_COMP

class RCFirstConv1d(nn.Module):
    
    def __init__(self, out_channels, kernel_size, dilation=1, bias=True, dropout=0.1):
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
        y = F.gelu(y)
        y = self.dropout(y)
        
        return y 

class MaskedMaxPool1d(nn.Module):
    def __init__(self, kernel_size, stride=2):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x, mask):  # x:(B, C, L), mask:(B, L)
        m = mask.unsqueeze(1)
        x_masked = x.masked_fill(~m, float('-inf'))
        y = F.max_pool1d(x_masked, self.kernel_size, self.stride)
        m_pooled = F.max_pool1d(m.float(), self.kernel_size, self.stride) > 0
        y = y.masked_fill(~m_pooled, 0.0)
        return y, m_pooled.squeeze(1)

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=pad, dilation=dilation)
        self.batch_norm = nn.BatchNorm1d(c_out)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(F.gelu(self.batch_norm(self.conv(x))))

class MaskedAttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, x, mask=None):  # x: (B, C, L)
        x_t = x.transpose(1, 2)  # (B, L, C)
        attn_scores = self.score(torch.tanh(self.query(x_t))).squeeze(-1)  # (B, L)
        attn = torch.softmax(attn_scores, dim=-1)
        if mask is not None:
            mask_f = mask.float()
            attn = attn * mask_f
            norm = mask_f.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            attn = attn / norm
        pooled = torch.bmm(attn.unsqueeze(1), x_t).squeeze(1)
        return pooled

class RCInputInvariantCNN(nn.Module):
    def __init__(
            self, 
            motif_width = 64, motif_kernel = (9, 15, ), 
            context_width = (128, 256, ), context_kernel = (7, 5, ), context_dilation = (1, 2, )
        ):
        super().__init__()

        self.motif_conv_blocks = nn.ModuleList(
            RCFirstConv1d(out_channels=motif_width, kernel_size=ker, dilation=1, dropout=0.2)
            for ker in motif_kernel
        )
        dim = motif_width * len(motif_kernel)

        self.context_blocks = nn.ModuleList()
        c_in = dim
        for width, ker, dil in zip(context_width, context_kernel, context_dilation):
            self.context_blocks.append(
                ConvBlock(c_in, c_out=width, kernel_size=ker, dilation=dil, dropout=0.2)
            )
            c_in = width

        self.pool = MaskedMaxPool1d(kernel_size=2, stride=2)
        head_in_features = c_in
        self.attention_pool = MaskedAttentionPooling(head_in_features)

        self.head = nn.Sequential(
            nn.Linear(head_in_features, width),
            nn.GELU(),
            nn.Dropout(0.4),
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
            try:
                z_new = block(z)
                if z_new.shape == z.shape:
                    z = z_new + z
                else:
                    z = z_new
            except RuntimeError:
                z = block(z)
            z, mask = self.pool(z, mask)

        z_pooled = self.attention_pool(z, mask)
        out = self.head(z_pooled).squeeze(-1)
        
        return out
    
if __name__ == "__main__":
    with torch.no_grad():
        device = 'cpu'
        model = RCInputInvariantCNN().to(device)
        X = torch.randn(3, 5, 64, device=device)
        mask = torch.zeros(3, 64, dtype=torch.bool, device=device)
        mask[0, :120] = True
        mask[1, :100] = True
        mask[2, :80] = True
        out = model(X, mask)
        print('Device:', device)
        print('Output shape:', out.shape)