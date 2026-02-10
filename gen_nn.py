import sys
import matplotlib.pyplot as plt

# Assuming PlotNeuralNet is in your PYTHONPATH or installed.
# You might need to adjust the import path based on your local installation.
# If using the repo directly: sys.path.append('../PlotNeuralNet')
from pyrenn import *

def draw_architecture():
    # Create figure
    f = plt.figure(figsize=(16, 10))
    ax = f.add_subplot(111)
    
    # ========================================================================
    # INPUTS
    # ========================================================================
    
    # 1. Raw Input (Top Branch - Frequency Domain)
    # Input: (Batch, 7, Time)
    start_raw = add_starting_node(ax, text="Raw Input\nx$_{raw}$", start=(0, 8), width=2.5, height=1.5, color='royalblue')
    
    # 2. Filtered Input (Middle Branch - Time Domain)
    # Input: (Batch, 7, Time)
    start_filt = add_starting_node(ax, text="Filtered Input\nx$_{filtered}$", start=(0, 4), width=2.5, height=1.5, color='forestgreen')
    
    # 3. Auxiliary Input (Bottom Branch)
    # Input: (Batch, 7)
    start_aux = add_starting_node(ax, text="Aux Features\n(7)", start=(0, 0), width=2.5, height=1.5, color='darkorange')

    # ========================================================================
    # FREQUENCY BRANCH (Top)
    # ========================================================================
    
    # FFT Operation
    to_conv_(ax, start_raw, text="FFT", start=(3, 8), width=1.5, height=1.5, depth=10, color='slateblue')
    
    # ImprovedFrequencyEncoder - Simplified representation
    # Multi-scale Convs (3,5,7) -> Concat -> BN
    freq_conv = to_conv_(ax, text="Multi-scale\nConvs", start=(5, 8), width=1.5, height=1.5, depth=20, color='slateblue')
    
    # Band Attention & Pooling
    # Conv2 -> BN -> Attention -> AdaptivePool
    freq_attn = to_conv_(ax, freq_conv, text="Band Attention\n& Pool", start=(7.5, 8), width=1.5, height=1.5, depth=20, color='mediumpurple')
    
    # Output Projection (Linear layers inside Freq Encoder)
    # Output dim = 64
    freq_feat = to_ff_(ax, freq_attn, text="Freq\nFeatures\n(64)", start=(10, 8), width=1.5, height=1.5, depth=10, color='purple')

    # ========================================================================
    # TIME BRANCH (Middle)
    # ========================================================================
    
    # EmbeddedTCN
    # Channels: 7 -> 32 -> 64 -> 64 -> 32
    tcn1 = to_conv_(ax, start_filt, text="TCN Block 1\n(32)", start=(3, 4), width=1.5, height=1.5, depth=20, color='mediumseagreen')
    tcn2 = to_conv_(ax, tcn1, text="TCN Block 2\n(64)", start=(5, 4), width=1.5, height=1.5, depth=20, color='mediumseagreen')
    tcn3 = to_conv_(ax, tcn2, text="TCN Block 3\n(64)", start=(7.5, 4), width=1.5, height=1.5, depth=20, color='mediumseagreen')
    tcn4 = to_conv_(ax, tcn3, text="TCN Block 4\n(32)", start=(10, 4), width=1.5, height=1.5, depth=20, color='mediumseagreen')
    
    # AdaptiveAvgPool1d(1) -> Flatten
    # Output dim = 32
    time_feat = to_pooling_(ax, tcn4, text="Pool\n(32)", start=(11.5, 4), width=1.5, height=1.5, depth=10, color='teal')

    # ========================================================================
    # AUX BRANCH (Bottom)
    # ========================================================================
    
    # Aux Encoder
    # Linear(7, 32) -> ReLU -> BN -> Dropout
    aux_1 = to_ff_(ax, start_aux, text="FC\n(32)", start=(4, 0), width=1.5, height=1.5, depth=10, color='darkorange')
    
    # Linear(32, 16) -> ReLU
    # Output dim = 16
    aux_feat = to_ff_(ax, aux_1, text="FC\n(16)", start=(7.5, 0), width=1.5, height=1.5, depth=10, color='chocolate')

    # ========================================================================
    # FUSION BLOCK
    # ========================================================================
    
    # Attention-based Fusion
    # Takes Time Feats (32) and Freq Feats (64) -> Combined (96) -> Attention -> Fusion MLP
    # Placing it in the middle vertically to collect both streams
    fusion_block = to_ff_(ax, text="Attention\nFusion\n(96 -> 64)", start=(13.5, 6), width=2.5, height=2.5, depth=20, color='firebrick')
    
    # Draw connections from streams to Fusion
    # Frequency (Top) -> Fusion (Middle-Right)
    add_connection(ax, freq_feat, fusion_block)
    # Time (Middle) -> Fusion (Middle-Right)
    add_connection(ax, time_feat, fusion_block)
    
    # ========================================================================
    # HEADS
    # ========================================================================
    
    # 1. Detection Head (Top)
    # Input: Fused Features (64)
    det_1 = to_ff_(ax, fusion_block, text="FC\n(32)", start=(16.5, 8), width=1.5, height=1.5, depth=10, color='gray')
    det_out = to_ff_(ax, det_1, text="Detection\nOutput\n(2)", start=(18.5, 8), width=1.5, height=1.5, depth=10, color='black')
    
    # 2. Species Head (Bottom)
    # Input: Concat(Fused Features (64), Aux Features (16)) -> Total 80
    # We need to connect Aux Features to this path
    # Visual trick: Place a merge node or just connect directly
    
    sp_concat = to_ff_(ax, text="Concat\n(80)", start=(16.5, 2), width=2, height=1.5, depth=10, color='gray')
    
    # Connections for Species
    # From Fusion Block down to Species Concat
    add_connection(ax, fusion_block, sp_concat)
    # From Aux Features (Bottom) up to Species Concat
    add_connection(ax, aux_feat, sp_concat)
    
    # Species MLP
    sp_1 = to_ff_(ax, sp_concat, text="FC\n(48)", start=(19, 2), width=1.5, height=1.5, depth=10, color='gray')
    sp_out = to_ff_(ax, sp_1, text="Species\nOutput\n(2)", start=(21, 2), width=1.5, height=1.5, depth=10, color='black')

    # Adjust plot limits
    ax.set_xlim(0, 23)
    ax.set_ylim(-1, 10)
    ax.axis('off')

    # Save to .tex
    plt.savefig('dual_stream_tcn.tex')
    print("Generated dual_stream_tcn.tex")

if __name__ == '__main__':
    draw_architecture()