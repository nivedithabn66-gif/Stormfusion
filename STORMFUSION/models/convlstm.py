"""ConvLSTM PyTorch Module for STORMFUSION (Step 8A).

Implements ConvLSTMCell and multi-layer ConvLSTM modules for spatio-temporal modeling of satellite feature maps.

Gates:
  - Input gate i_t = sigmoid(W_xi * x_t + W_hi * h_{t-1} + b_i)
  - Forget gate f_t = sigmoid(W_xf * x_t + W_hf * h_{t-1} + b_f)
  - Cell gate g_t = tanh(W_xc * x_t + W_hc * h_{t-1} + b_c)
  - Output gate o_t = sigmoid(W_xo * x_t + W_ho * h_{t-1} + b_o)
  - Cell state c_t = f_t * c_{t-1} + i_t * g_t
  - Hidden state h_t = o_t * tanh(c_t)
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell performing spatio-temporal state transitions."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super(ConvLSTMCell, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Combined convolution for all 4 gates (input, forget, cell, output)
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True,
        )

    def forward(
        self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None, c_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single timestep.

        Args:
            x: Input tensor [B, C_in, H, W]
            h_prev: Previous hidden state [B, C_hidden, H, W]
            c_prev: Previous cell state [B, C_hidden, H, W]

        Returns:
            (h_next, c_next) tensors of shape [B, C_hidden, H, W]
        """
        batch_size, _, height, width = x.size()

        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
        if c_prev is None:
            c_prev = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)

        combined = torch.cat([x, h_prev], dim=1)  # [B, C_in + C_hidden, H, W]
        conv_out = self.conv(combined)            # [B, 4 * C_hidden, H, W]

        cc_i, cc_f, cc_g, cc_o = torch.split(conv_out, self.hidden_channels, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)
        o = torch.sigmoid(cc_o)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM module supporting batch-first sequence tensors."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        num_layers: int = 1,
        batch_first: bool = True,
    ):
        super(ConvLSTM, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        cell_list = []
        for i in range(num_layers):
            cur_in_channels = in_channels if i == 0 else hidden_channels
            cell_list.append(ConvLSTMCell(cur_in_channels, hidden_channels, kernel_size))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self, x: torch.Tensor, hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass across all timesteps.

        Args:
            x: Input sequence tensor of shape [B, T, C_in, H, W] if batch_first else [T, B, C_in, H, W]
            hidden_state: Optional initial list of (h, c) tuples for each layer

        Returns:
            layer_output: Sequence output tensor of shape [B, T, C_hidden, H, W]
            (h_last, c_last): Final hidden and cell state tensors of shape [B, C_hidden, H, W]
        """
        if not self.batch_first:
            # Transpose to batch-first [B, T, C, H, W]
            x = x.permute(1, 0, 2, 3, 4)

        batch_size, seq_len, _, height, width = x.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, height, width, device=x.device, dtype=x.dtype)

        layer_output_list = []
        last_states = []

        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], h, c)
                output_inner.append(h)

            # Stack outputs along sequence dimension -> [B, T, C_hidden, H, W]
            cur_layer_input = torch.stack(output_inner, dim=1)
            layer_output_list.append(cur_layer_input)
            last_states.append((h, c))

        final_output_seq = layer_output_list[-1]
        final_h, final_c = last_states[-1]

        return final_output_seq, (final_h, final_c)

    def _init_hidden(
        self, batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Initializes zero hidden and cell states for all layers."""
        init_states = []
        for i in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device, dtype=dtype)
            c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device, dtype=dtype)
            init_states.append((h, c))
        return init_states
