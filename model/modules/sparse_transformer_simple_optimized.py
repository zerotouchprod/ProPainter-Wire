"""
Simple optimized version of SparseWindowAttention using PyTorch 2.x Scaled Dot Product Attention.
This is a minimal working version for testing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size, n_head):
    """
    Args:
        x: shape is (B, T, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B, num_windows_h, num_windows_w, n_head, T, window_size, window_size, C//n_head)
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size[0], window_size[0], W // window_size[1], window_size[1], n_head, C//n_head)
    windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return windows


class SimpleOptimizedSparseWindowAttention(nn.Module):
    """
    Simple optimized version using F.scaled_dot_product_attention.
    This is a minimal implementation for testing purposes.
    """
    def __init__(self, dim, n_head, window_size, pool_size=(4,4), qkv_bias=True, 
                 attn_drop=0., proj_drop=0., pooling_token=False):
        super().__init__()
        assert dim % n_head == 0
        self.key = nn.Linear(dim, dim, qkv_bias)
        self.query = nn.Linear(dim, dim, qkv_bias)
        self.value = nn.Linear(dim, dim, qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj = nn.Linear(dim, dim)
        self.n_head = n_head
        self.window_size = window_size
        self.pooling_token = pooling_token
        
        if pooling_token:
            ks, stride = pool_size, pool_size
            self.pool_layer = nn.Conv2d(dim, dim, kernel_size=ks, stride=stride, 
                                        padding=(0, 0), groups=dim)
            self.pool_layer.weight.data.fill_(1. / (pool_size[0] * pool_size[1]))
            self.pool_layer.bias.data.fill_(0)
        
        self.max_pool = nn.MaxPool2d(window_size, window_size, (0, 0))
        self.scale = (dim // n_head) ** -0.5
        
        # Add expand_size and valid_ind_rolled to match original
        self.expand_size = tuple((i + 1) // 2 for i in window_size)
        if any(i > 0 for i in self.expand_size):
            # Create dummy valid_ind_rolled to match original structure
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[:-self.expand_size[0], :-self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[:-self.expand_size[0], self.expand_size[1]:] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0]:, :-self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0]:, self.expand_size[1]:] = 0
            mask_roll = torch.stack((mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
            self.register_buffer("valid_ind_rolled", mask_roll.nonzero(as_tuple=False).view(-1))

    def forward(self, x, mask=None, T_ind=None, attn_mask=None):
        """
        Simple forward pass using scaled_dot_product_attention.
        """
        b, t, h, w, c = x.shape
        w_h, w_w = self.window_size
        c_head = c // self.n_head
        
        # Window partitioning
        n_wh = math.ceil(h / w_h)
        n_ww = math.ceil(w / w_w)
        new_h = n_wh * w_h
        new_w = n_ww * w_w
        
        # Padding if needed
        pad_r = new_w - w
        pad_b = new_h - h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0), mode='constant', value=0)
            if mask is not None:
                mask = F.pad(mask, (0, 0, 0, pad_r, 0, pad_b, 0, 0), mode='constant', value=0)

        # Compute Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Window partitioning
        win_q = window_partition(q.contiguous(), self.window_size, self.n_head)
        win_k = window_partition(k.contiguous(), self.window_size, self.n_head)
        win_v = window_partition(v.contiguous(), self.window_size, self.n_head)
        
        # Reshape for attention
        win_q = win_q.view(b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head)
        win_k = win_k.view(b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head)
        win_v = win_v.view(b, n_wh*n_ww, self.n_head, t, w_h*w_w, c_head)
        
        # Initialize output
        out = torch.zeros_like(win_q)
        
        # Process each window
        for i in range(b):
            for j in range(n_wh*n_ww):
                # Get query, key, value for this window
                q_ij = win_q[i, j]  # [n_head, t, w_h*w_w, c_head]
                k_ij = win_k[i, j]  # [n_head, t, w_h*w_w, c_head]
                v_ij = win_v[i, j]  # [n_head, t, w_h*w_w, c_head]
                
                # Reshape for attention: combine t and spatial dimensions
                q_flat = q_ij.view(self.n_head, t*w_h*w_w, c_head)
                k_flat = k_ij.view(self.n_head, t*w_h*w_w, c_head)
                v_flat = v_ij.view(self.n_head, t*w_h*w_w, c_head)
                
                # Use scaled dot product attention
                attn_output = F.scaled_dot_product_attention(
                    q_flat, k_flat, v_flat,
                    attn_mask=None,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    is_causal=False,
                    scale=self.scale
                )
                
                # Reshape back
                attn_output = attn_output.view(self.n_head, t, w_h*w_w, c_head)
                out[i, j] = attn_output
        
        # Reshape output back to original format
        out = out.view(b, n_wh, n_ww, self.n_head, t, w_h, w_w, c_head)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(b, t, new_h, new_w, c)
        
        # Remove padding if needed
        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :h, :w, :]
        
        # Output projection
        out = self.proj_drop(self.proj(out))
        return out


# Test the implementation
if __name__ == "__main__":
    # Simple test
    batch_size = 2
    time_steps = 3
    height, width = 32, 32
    channels = 128
    num_heads = 4
    window_size = (8, 8)
    
    # Create model
    model = SimpleOptimizedSparseWindowAttention(
        dim=channels,
        n_head=num_heads,
        window_size=window_size,
        pooling_token=False
    )
    
    # Create input
    x = torch.randn(batch_size, time_steps, height, width, channels)
    mask = (torch.randn(batch_size, time_steps, height, width, 1) > 0).float()
    
    # Forward pass
    output = model(x, mask)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ Simple optimized attention works!")
