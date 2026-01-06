"""
HD U-Net (Hierarchical Dense U-Net) for 3D Medical Image Segmentation.

This module implements HD U-Net with:
- Dense blocks with bottleneck layers for efficient feature reuse
- Multi-scale feature aggregation through hierarchical skip connections
- Deep supervision for improved gradient flow
- Attention gates for enhanced skip connections (optional)

Optimized for RTX 3060 12GB with gradient checkpointing support.

References:
- DenseNet: https://arxiv.org/abs/1608.06993
- HD U-Net: https://arxiv.org/abs/1904.02236
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DenseLayer(nn.Module):
    """Dense layer with bottleneck design (BN-ReLU-Conv1x1-BN-ReLU-Conv3x3).
    
    Args:
        in_channels: Number of input channels
        growth_rate: Number of output channels (k in DenseNet paper)
        bn_size: Multiplicative factor for bottleneck layer (default: 4)
        dropout_rate: Dropout rate after each dense layer
    """
    
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        
        # Bottleneck: 1x1x1 conv to reduce channels
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(
            in_channels,
            bn_size * growth_rate,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        
        # 3x3x3 conv
        self.bn2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dense concatenation."""
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        # Concatenate input with output (dense connection)
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    """Dense block containing multiple dense layers.
    
    Args:
        in_channels: Number of input channels
        growth_rate: Growth rate (k) - channels added per layer
        num_layers: Number of dense layers in the block
        bn_size: Bottleneck size multiplier
        dropout_rate: Dropout rate
        use_checkpoint: Enable gradient checkpointing for memory efficiency
    """
    
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        num_layers: int,
        bn_size: int = 4,
        dropout_rate: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        layers = []
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels=in_channels + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout_rate=dropout_rate,
            )
            layers.append(layer)
        
        self.layers = nn.ModuleList(layers)
        self.out_channels = in_channels + num_layers * growth_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all dense layers."""
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x


class TransitionDown(nn.Module):
    """Transition layer for downsampling (encoder path).
    
    Performs: BN-ReLU-Conv1x1-Dropout-AvgPool
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dropout_rate: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else None
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with compression and pooling."""
        out = self.conv(self.relu(self.bn(x)))
        if self.dropout is not None:
            out = self.dropout(out)
        return self.pool(out)


class TransitionUp(nn.Module):
    """Transition layer for upsampling (decoder path).
    
    Uses transposed convolution for learnable upsampling.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with upsampling."""
        return self.relu(self.bn(self.conv_transpose(x)))


class AttentionGate(nn.Module):
    """Attention gate for skip connections.
    
    Enhances relevant features from encoder while suppressing
    irrelevant ones based on decoder context.
    
    Args:
        gate_channels: Number of channels in gating signal (from decoder)
        in_channels: Number of channels in skip connection (from encoder)
        inter_channels: Number of intermediate channels
    """
    
    def __init__(
        self,
        gate_channels: int,
        in_channels: int,
        inter_channels: Optional[int] = None,
    ):
        super().__init__()
        
        if inter_channels is None:
            inter_channels = in_channels // 2
        
        # Gating signal transformation
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_channels),
        )
        
        # Skip connection transformation
        self.W_x = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_channels),
        )
        
        # Attention coefficient
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention to skip connection.
        
        Args:
            x: Skip connection features from encoder
            g: Gating signal from decoder
            
        Returns:
            Attention-weighted features
        """
        # Transform gating signal
        g_conv = self.W_g(g)
        
        # Transform skip connection
        x_conv = self.W_x(x)
        
        # Align spatial dimensions if needed
        if g_conv.shape[2:] != x_conv.shape[2:]:
            g_conv = F.interpolate(
                g_conv,
                size=x_conv.shape[2:],
                mode='trilinear',
                align_corners=False,
            )
        
        # Compute attention
        psi = self.relu(g_conv + x_conv)
        psi = self.psi(psi)
        
        return x * psi


class HDUNet(nn.Module):
    """Hierarchical Dense U-Net for 3D medical image segmentation.
    
    Architecture features:
    - Dense blocks in both encoder and decoder
    - Hierarchical skip connections with optional attention gates
    - Deep supervision for better gradient flow
    - Efficient bottleneck design
    
    Args:
        in_channels: Number of input channels (e.g., 11 for CT + 10 ROIs)
        out_channels: Number of output channels (1 for dose prediction)
        init_features: Initial number of features after first conv (default: 48)
        growth_rate: Growth rate for dense blocks (default: 16)
        layers_per_block: Number of layers in each dense block (default: [4, 4, 4, 4])
        bn_size: Bottleneck size multiplier (default: 4)
        dropout_rate: Dropout rate (default: 0.2)
        compression: Compression factor for transition layers (default: 0.5)
        use_attention: Use attention gates in skip connections
        deep_supervision: Enable deep supervision
        use_checkpoint: Enable gradient checkpointing for memory efficiency
    """
    
    def __init__(
        self,
        in_channels: int = 11,
        out_channels: int = 1,
        init_features: int = 48,
        growth_rate: int = 16,
        layers_per_block: List[int] = None,
        bn_size: int = 4,
        dropout_rate: float = 0.2,
        compression: float = 0.5,
        use_attention: bool = True,
        deep_supervision: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        if layers_per_block is None:
            layers_per_block = [4, 4, 4, 4]
        
        self.deep_supervision = deep_supervision
        self.use_attention = use_attention
        self.num_levels = len(layers_per_block)
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, init_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(init_features, init_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True),
        )
        
        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        self.encoder_transitions = nn.ModuleList()
        self.skip_channels = []
        
        num_features = init_features
        
        for i, num_layers in enumerate(layers_per_block):
            # Dense block
            block = DenseBlock(
                in_channels=num_features,
                growth_rate=growth_rate,
                num_layers=num_layers,
                bn_size=bn_size,
                dropout_rate=dropout_rate,
                use_checkpoint=use_checkpoint,
            )
            self.encoder_blocks.append(block)
            num_features = block.out_channels
            self.skip_channels.append(num_features)
            
            # Transition down (except for last block)
            if i < len(layers_per_block) - 1:
                out_features = int(num_features * compression)
                transition = TransitionDown(
                    in_channels=num_features,
                    out_channels=out_features,
                    dropout_rate=dropout_rate,
                )
                self.encoder_transitions.append(transition)
                num_features = out_features
        
        # Bottleneck
        self.bottleneck = DenseBlock(
            in_channels=num_features,
            growth_rate=growth_rate,
            num_layers=layers_per_block[-1],
            bn_size=bn_size,
            dropout_rate=dropout_rate,
            use_checkpoint=use_checkpoint,
        )
        num_features = self.bottleneck.out_channels
        
        # Decoder path
        self.decoder_transitions = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if use_attention else None
        
        # Reverse skip channels for decoder
        skip_channels_reversed = self.skip_channels[::-1]
        
        for i in range(len(layers_per_block)):
            skip_ch = skip_channels_reversed[i]
            
            # Transition up
            transition = TransitionUp(
                in_channels=num_features,
                out_channels=num_features // 2,
            )
            self.decoder_transitions.append(transition)
            
            # Attention gate
            if use_attention:
                attn = AttentionGate(
                    gate_channels=num_features // 2,
                    in_channels=skip_ch,
                )
                self.attention_gates.append(attn)
            
            # Concatenated features: upsampled + skip
            concat_features = num_features // 2 + skip_ch
            
            # Dense block
            block = DenseBlock(
                in_channels=concat_features,
                growth_rate=growth_rate,
                num_layers=layers_per_block[-(i + 1)],
                bn_size=bn_size,
                dropout_rate=dropout_rate,
                use_checkpoint=use_checkpoint,
            )
            self.decoder_blocks.append(block)
            num_features = block.out_channels
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv3d(num_features, out_channels, kernel_size=1),
        )
        
        # Deep supervision heads
        if deep_supervision:
            self.deep_supervision_heads = nn.ModuleList()
            # Create supervision heads for decoder levels (except last)
            decoder_channels = []
            temp_features = self.bottleneck.out_channels
            for i in range(len(layers_per_block) - 1):
                skip_ch = skip_channels_reversed[i]
                concat_features = temp_features // 2 + skip_ch
                block_out = concat_features + layers_per_block[-(i + 1)] * growth_rate
                decoder_channels.append(block_out)
                temp_features = block_out
            
            for ch in decoder_channels:
                head = nn.Conv3d(ch, out_channels, kernel_size=1)
                self.deep_supervision_heads.append(head)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through HD U-Net.
        
        Args:
            x: Input tensor of shape (B, C, H, W, D)
            
        Returns:
            Output tensor of shape (B, out_channels, H, W, D)
            If deep_supervision and training: tuple of (main_output, [aux_outputs])
        """
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder path with skip connections
        skip_features = []
        
        for i, (block, transition) in enumerate(
            zip(self.encoder_blocks[:-1], self.encoder_transitions)
        ):
            x = block(x)
            skip_features.append(x)
            x = transition(x)
        
        # Last encoder block (no transition)
        x = self.encoder_blocks[-1](x)
        skip_features.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        deep_outputs = [] if self.deep_supervision and self.training else None
        skip_features_reversed = skip_features[::-1]
        
        for i, (transition, block) in enumerate(
            zip(self.decoder_transitions, self.decoder_blocks)
        ):
            # Upsample
            x = transition(x)
            
            # Get skip connection
            skip = skip_features_reversed[i]
            
            # Apply attention gate
            if self.use_attention:
                skip = self.attention_gates[i](skip, x)
            
            # Align spatial dimensions if needed
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode='trilinear',
                    align_corners=False,
                )
            
            # Concatenate
            x = torch.cat([x, skip], dim=1)
            
            # Dense block
            x = block(x)
            
            # Deep supervision output (except for last level)
            if self.deep_supervision and self.training and i < len(self.decoder_blocks) - 1:
                ds_out = self.deep_supervision_heads[i](x)
                # Upsample to full resolution
                ds_out = F.interpolate(
                    ds_out,
                    scale_factor=2 ** (len(self.decoder_blocks) - i - 1),
                    mode='trilinear',
                    align_corners=False,
                )
                deep_outputs.append(ds_out)
        
        # Final convolution
        output = self.final_conv(x)
        
        if self.deep_supervision and self.training:
            return output, deep_outputs
        
        return output


class HDUNetLite(nn.Module):
    """Lightweight version of HD U-Net for memory-constrained GPUs (12GB).
    
    Heavily optimized for 128³ 3D medical volumes:
    - Minimal layers per block (2)
    - Very low growth rate (8)
    - Small initial features (24)
    - High compression (0.5)
    - Gradient checkpointing enabled by default
    - No attention gates (saves memory)
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        init_features: Initial features (default: 24)
        growth_rate: Growth rate (default: 8)
        dropout_rate: Dropout rate (default: 0.2)
        use_attention: Use attention gates (default: False for memory)
        use_checkpoint: Enable gradient checkpointing (default: True)
        deep_supervision: Ignored for lite version
    """
    
    def __init__(
        self,
        in_channels: int = 11,
        out_channels: int = 1,
        init_features: int = 24,
        growth_rate: int = 8,
        dropout_rate: float = 0.2,
        use_attention: bool = False,
        use_checkpoint: bool = True,  # Enable by default for memory
        deep_supervision: bool = False,  # Accept but ignore for lite version
    ):
        super().__init__()
        
        # Ultra-lightweight configuration for 12GB GPUs
        self.model = HDUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            growth_rate=growth_rate,
            layers_per_block=[2, 2, 2, 2],  # Minimal layers
            bn_size=2,  # Smaller bottleneck
            dropout_rate=dropout_rate,
            compression=0.5,
            use_attention=False,  # Always off for lite
            deep_supervision=False,  # Always disabled for lite version
            use_checkpoint=use_checkpoint,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


def get_hd_unet(
    variant: str = "standard",
    in_channels: int = 11,
    out_channels: int = 1,
    use_checkpoint: bool = False,
    **kwargs,
) -> nn.Module:
    """Factory function to create HD U-Net variants.
    
    Args:
        variant: Model variant ('lite', 'standard', 'large')
        in_channels: Number of input channels
        out_channels: Number of output channels
        use_checkpoint: Enable gradient checkpointing
        **kwargs: Additional arguments passed to the model
        
    Returns:
        HD U-Net model instance
    """
    if variant == "lite":
        # Filter kwargs for lite model
        lite_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['init_features', 'growth_rate', 'dropout_rate', 
                                'use_attention', 'deep_supervision']}
        return HDUNetLite(
            in_channels=in_channels,
            out_channels=out_channels,
            use_checkpoint=True,  # Always use checkpointing for lite
            **lite_kwargs,
        )
    elif variant == "standard":
        return HDUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=32,
            growth_rate=12,
            layers_per_block=[3, 3, 3, 3],
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
    elif variant == "large":
        return HDUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=64,
            growth_rate=24,
            layers_per_block=[4, 6, 8, 8],
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}. Choose from: lite, standard, large")


# Test code
if __name__ == "__main__":
    # Test HD U-Net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Test standard model
    model = get_hd_unet("standard", in_channels=11, out_channels=1)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(1, 11, 128, 128, 128).to(device)
    
    model.train()
    output = model(x)
    if isinstance(output, tuple):
        main_out, aux_outs = output
        print(f"Training output shape: {main_out.shape}")
        print(f"Number of auxiliary outputs: {len(aux_outs)}")
    else:
        print(f"Output shape: {output.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"Eval output shape: {output.shape}")
