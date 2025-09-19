import torch
import torch.nn as nn
from typing import Tuple, Optional


class SpaceLSTMBlock(nn.Module):
    """
    A spatial LSTM block that processes 3D spatial data by reshaping it into sequences
    and applying LSTM along a grouped spatial dimension. It includes residual connection
    and layer normalization.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Args:
            input_size: Dimension of input features.
            hidden_size: Hidden size of the LSTM.
            output_size: Dimension of output features.
        """
        super(SpaceLSTMBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1

        # LSTM processes the reshaped spatial sequence
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        # Layer norm after LSTM output
        self.norm = nn.LayerNorm(hidden_size)
        # Final projection with skip connection (input_size + hidden_size)
        self.fc = nn.Linear(hidden_size + input_size, output_size, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, Z: int, H: int, W: int) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, Z*H*W, C)
            Z, H, W: Spatial dimensions (depth, height, width)

        Returns:
            Output tensor of shape (B, Z*H*W, output_size)
        """
        skip = x  # Residual connection

        # Reshape: (B, Z, H, W, C)
        x = x.reshape(x.shape[0], Z, H, W, -1)
        # Group along depth: split Z into 2 groups of Z//2
        x = x.reshape(x.shape[0], 2, Z // 2, H, W, -1)
        # Flatten batch and grouping dim: (2*B, (Z//2)*H*W, C)
        x = x.reshape(x.shape[0] * 2, (Z // 2 * H * W), -1)

        # Initialize hidden states (moved inside forward for correctness)
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Apply LSTM
        x, _ = self.lstm(x, (h0, c0))  # (2*B, seq_len, hidden_size)

        # Reshape back: (B, 2, Z//2, H, W, hidden_size)
        x = x.reshape(x.shape[0] // 2, 2, Z // 2, H, W, self.hidden_size)
        # Merge group dim back: (B, Z, H, W, hidden_size)
        x = x.reshape(x.shape[0], Z, H, W, self.hidden_size)
        # Flatten spatial dims: (B, Z*H*W, hidden_size)
        x = x.reshape(x.shape[0], Z * H * W, self.hidden_size)

        # Normalize LSTM output
        x = self.norm(x)
        # Concatenate with original input for residual
        out = self.fc(self.relu(torch.cat([skip, x], dim=2)))  # (B, Z*H*W, output_size)
        return out


class SpaceLSTM(nn.Module):
    """
    Stacked SpaceLSTMBlocks with final residual connection and projection.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int):
        """
        Args:
            input_size: Input feature dimension.
            hidden_size: Hidden size for each LSTM block.
            output_size: Output feature dimension.
            num_layers: Number of stacked SpaceLSTMBlock layers.
        """
        super(SpaceLSTM, self).__init__()
        self.layers = nn.ModuleList([
            SpaceLSTMBlock(input_size, hidden_size, output_size) for _ in range(num_layers)
        ])
        # Final projection and normalization
        self.fc = nn.Linear(input_size, output_size, bias=False)
        self.norm = nn.LayerNorm(input_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, Z: int, H: int, W: int) -> torch.Tensor:
        """
        Forward pass through multiple SpaceLSTM blocks.

        Args:
            x: Input tensor (B, Z*H*W, C)
            Z, H, W: Spatial dimensions

        Returns:
            Processed tensor (B, Z*H*W, output_size)
        """
        skip = x
        for layer in self.layers:
            x = layer(x, Z, H, W)
        x = self.norm(x)
        out = self.fc(self.relu(x + skip))  # Final residual + projection
        return out


class SpaceEmbedding(nn.Module):
    """
    Embeds atmospheric and surface data into patch-based tokens using 3D and 2D convolutions.
    """

    def __init__(self, output_channel: int):
        """
        Args:
            output_channel: Number of output channels (embedding dimension).
        """
        super(SpaceEmbedding, self).__init__()
        self.output_channel = output_channel
        # 3D conv for atmospheric data (8 channels)
        self.conv_atm = nn.Conv3d(8, output_channel, kernel_size=(2, 4, 4), stride=(2, 4, 4), padding=0)
        # 2D conv for surface data (6 channels)
        self.conv_surf = nn.Conv2d(6, output_channel, kernel_size=4, stride=4, padding=0)
        # Pad top level (z=0) of atmosphere to align with surface
        self.pad = nn.ConstantPad3d((0, 0, 0, 0, 0, 1), 0)  # Add one zero layer at top
        self.norm = nn.BatchNorm3d(output_channel)

    def forward(
        self,
        x_atm: torch.Tensor,
        x_surf: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_atm: Atmospheric input (B, 8, 13, 64, 64)
            x_surf: Surface input (B, 6, 1, 64, 64)

        Returns:
            Embedded tokens (B, total_patches, output_channel)
        """
        # Pad atmosphere data to match surface height
        x_atm = self.pad(x_atm)  # Now (B, 8, 14, 64, 64)

        # Reshape surface data for 2D conv
        B = x_surf.shape[0]
        x_surf = x_surf.reshape(B, 6, x_surf.shape[3], x_surf.shape[4])  # (B, 6, 64, 64)

        # Apply convolutions
        out_atm = self.conv_atm(x_atm)  # (B, C, 7, 16, 16)
        out_surf = self.conv_surf(x_surf)  # (B, C, 16, 16)

        # Reshape surface to match 3D structure: (B, C, 1, 16, 16)
        out_surf = out_surf.reshape(B, self.output_channel, 1, out_surf.shape[2], out_surf.shape[3])

        # Concatenate along depth (Z) dimension: (B, C, 8, 16, 16)
        out = torch.cat([out_atm, out_surf], dim=2)
        out = self.norm(out)

        # Permute and flatten: (B, 8*16*16, C) = (B, 2048, C)
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        out = out.reshape(out.shape[0], -1, out.shape[-1])
        return out


class RespaceRecovery(nn.Module):
    """
    Reconstructs original atmospheric and surface data from embedded tokens
    using transpose convolutions.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: Input feature dimension (e.g., 128).
        """
        super(RespaceRecovery, self).__init__()
        # Transpose conv to upsample 3D features back
        self.conv = nn.ConvTranspose3d(dim, 5, kernel_size=(2, 4, 4), stride=(2, 4, 4), padding=0)
        # Transpose conv for surface features
        self.conv_surface = nn.ConvTranspose2d(dim, 4, kernel_size=4, stride=4, padding=0)
        # Linear layers to fuse residual connections
        self.linear1 = nn.Linear(5 * 2, 5, bias=False)  # For atmospheric channels
        self.linear2 = nn.Linear(4 * 2, 4, bias=False)  # For surface channels

    def forward(
        self,
        x: torch.Tensor,
        x_atm: torch.Tensor,
        x_surf: torch.Tensor,
        Z: int,
        H: int,
        W: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Embedded tokens (B, Z*H*W, dim)
            x_atm: Original atmospheric input (for residual)
            x_surf: Original surface input (for residual)
            Z, H, W: Spatial dimensions

        Returns:
            Reconstructed atmospheric and surface tensors.
        """
        # Reshape to 3D: (B, dim, Z, H, W)
        x = x.transpose(1, 2)
        x = x.reshape(x.shape[0], x.shape[1], Z, H, W)

        # Split: last slice is surface, rest is atmosphere
        atm_features = x[:, :, :-1, :, :].contiguous()  # (B, dim, Z-1, H, W)
        surf_features = x[:, :, -1, :, :].contiguous()  # (B, dim, H, W)

        # Reconstruct atmospheric data
        output = self.conv(atm_features)  # (B, 5, 12, 64, 64)
        output = output[:, :, :-1, :, :].contiguous()  # Trim extra layer

        # Add original atmospheric data as residual (first 5 channels)
        residual_atm = x_atm[:, :5, :, :, :].contiguous()
        output = torch.cat([output, residual_atm], dim=1)  # (B, 10, 12, 64, 64)
        output = output.permute(0, 2, 3, 4, 1).contiguous()  # (B, 12, 64, 64, 10)
        output = self.linear1(output)  # (B, 12, 64, 64, 5)
        output = output.permute(0, 4, 1, 2, 3).contiguous()  # (B, 5, 12, 64, 64)

        # Reconstruct surface data
        output_surface = self.conv_surface(surf_features)  # (B, 4, 64, 64)
        output_surface = output_surface.unsqueeze(2)  # (B, 4, 1, 64, 64)

        # Add original surface data as residual (first 4 channels)
        residual_surf = x_surf[:, :4, :, :, :].contiguous()
        output_surface = torch.cat([output_surface, residual_surf], dim=1)  # (B, 8, 1, 64, 64)
        output_surface = output_surface.permute(0, 2, 3, 4, 1).contiguous()  # (B, 1, 64, 64, 8)
        output_surface = self.linear2(output_surface)  # (B, 1, 64, 64, 4)
        output_surface = output_surface.permute(0, 4, 1, 2, 3).contiguous()  # (B, 4, 1, 64, 64)

        return output, output_surface


class SpaceDownSample(nn.Module):
    """
    Reduces spatial resolution by grouping patches and increasing feature dimension.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: Input feature dimension.
        """
        super(SpaceDownSample, self).__init__()
        self.linear = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, Z: int, H: int, W: int) -> torch.Tensor:
        """
        Downsample from (Z, H, W) to (Z//2, H//2, W//2) with doubled feature dim.

        Args:
            x: Input tensor (B, Z*H*W, dim)
            Z, H, W: Input spatial dimensions

        Returns:
            Downsampled tensor (B, (Z//2)*(H//2)*(W//2), 2*dim)
        """
        x = x.reshape(x.shape[0], Z, H, W, x.shape[-1])  # (B, Z, H, W, dim)

        # Group 2x2x2 patches into one
        x = x.reshape(x.shape[0], Z // 2, 2, H // 2, 2, W // 2, 2, x.shape[-1])
        x = x.permute(0, 2, 1, 3, 5, 4, 6, 7).contiguous()  # Rearrange
        x = x.reshape(x.shape[0] * 2, (Z // 2) * (H // 2) * (W // 2), 4 * x.shape[-1])

        x = self.norm(x)
        x = self.linear(x)  # (2*B, new_seq_len, 2*dim)
        return x


class SpaceUpSample(nn.Module):
    """
    Increases spatial resolution by splitting features into finer patches.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
        """
        super(SpaceUpSample, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim * 4, bias=False)
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, Z: int, H: int, W: int) -> torch.Tensor:
        """
        Upsample from (Z, H, W) to (2*Z, 2*H, 2*W).

        Args:
            x: Input tensor (B, Z*H*W, input_dim)
            Z, H, W: Input spatial dimensions

        Returns:
            Upsampled tensor (B, (2*Z)*(2*H)*(2*W), output_dim)
        """
        x = self.linear1(x)  # Expand feature dim

        # Reshape to 3D grid with grouped structure
        x = x.reshape(x.shape[0] // 2, 2, Z, H, W, 2, 2, -1 // 4)
        x = x.permute(0, 2, 1, 3, 5, 4, 6, 7).contiguous()
        x = x.reshape(x.shape[0], Z * 2, H * 2, W * 2, -1)

        x = x.reshape(x.shape[0], -1, x.shape[-1])  # Flatten
        x = self.norm(x)
        x = self.linear2(x)  # Final mixing
        return x


class SCLNet(nn.Module):
    """
    Main SCL (Spatial Contextual Learning) network for atmospheric and surface modeling.
    """

    def __init__(self):
        super(SCLNet, self).__init__()
        # Embedding
        self.embed = SpaceEmbedding(64)
        # Encoder
        self.layer1 = SpaceLSTM(64, 32, 64, 6)
        self.downsample = SpaceDownSample(64)
        self.layer2 = SpaceLSTM(128, 64, 128, 6)
        self.layer3 = SpaceLSTM(128, 64, 128, 6)
        # Decoder
        self.upsample = SpaceUpSample(128, 64)
        self.layer4 = SpaceLSTM(64, 32, 64, 6)
        # Recovery
        self.recovery = RespaceRecovery(128)

    def forward(
        self,
        atm_input: torch.Tensor,
        surf_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            atm_input: Atmospheric data (B, 8, 13, 64, 64)
            surf_input: Surface data (B, 6, 1, 64, 64)

        Returns:
            Reconstructed atmospheric and surface outputs.
        """
        # Embed input
        x = self.embed(atm_input, surf_input)
        skip = x

        # Encoder
        x = self.layer1(x, 8, 16, 16)
        x = self.downsample(x, 8, 16, 16)  # -> (4,8,8)
        x = self.layer2(x, 4, 8, 8)
        x = self.layer3(x, 4, 8, 8)

        # Decoder
        x = self.upsample(x, 4, 8, 8)  # -> (8,16,16)
        x = self.layer4(x, 8, 16, 16)

        # Skip connection
        x = torch.cat([skip, x], dim=2)

        # Recover outputs
        output_atm, output_surf = self.recovery(x, atm_input, surf_input, 8, 16, 16)
        return output_atm, output_surf

def generate_SCL() -> SCLNet:
    """Factory function to generate an SCL model instance."""
    return SCLNet()