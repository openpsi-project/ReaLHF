from typing import Optional, Union

import torch
import torch.nn as nn


class LayerNormLinear(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_norm_epsilon: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16
        self.ln = nn.LayerNorm(input_dim, eps=layer_norm_epsilon, dtype=dtype, device=device)
        self.linear = nn.Linear(input_dim, output_dim, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.ln(x))


class LayerNormMLP(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        resid_pdrop: float,
        activation_function: str,
        layer_norm_epsilon: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16
        self.ln = nn.LayerNorm(hidden_dim, eps=layer_norm_epsilon, dtype=dtype, device=device)
        self.c_fc = nn.Linear(hidden_dim, intermediate_dim, dtype=dtype, device=device)
        self.c_proj = nn.Linear(intermediate_dim, hidden_dim, dtype=dtype, device=device)
        if activation_function == "gelu":
            self.act = nn.functional.gelu
        else:
            raise NotImplementedError("Only \"gelu\" activation function is available.")
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        import pickle
        hidden_states = self.ln(hidden_states)
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return self.dropout(hidden_states)