
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Updated constants and exponents for the Chinchilla scaling law
n_coefficient = 406.4
d_coefficient = 410.7
n_exponent = -0.34
d_exponent = -0.28
irreducible_error = 1.69

# Using the same model sizes and data volumes for consistency
model_sizes = np.array([10**3, 10**4, 10**5, 10**6])
data_volumes = np.array([10**3, 10**4, 10**5, 10**6])

# Using the architecture details from the paper to estimate parameter counts for each model variant.
# Note: These are rough estimates based on standard ViT configurations.

# Approximate parameters count for each ViT variant:
# ViT-Tiny (T): 12 layers, 192 dimensionality
# ViT-Small (S): 12 layers, 384 dimensionality
# ViT-Base (B): 12 layers, 768 dimensionality
# ViT-Large (L): 24 layers, 1024 dimensionality

# Rough estimation of parameters:
# Parameters roughly scale with the square of the dimensionality and linearly with the number of layers.

# Estimation formula: params ~ layers * (dimensionality^2) * constant
# The constant includes additional parameters from MLPs, layer norms, etc.
# For simplicity, we assume each layer has an MLP with a hidden layer size of dimensionality*4,
# and two layernorm layers with parameters equal to dimensionality.

def estimate_vit_parameters(layers, dim):
    mlp_params = layers * (dim * 4) * dim # MLP with one hidden layer
    attention_params = layers * (dim ** 2) # Self-attention mechanism
    layernorm_params = layers * 2 * dim # Two LayerNorm layers per transformer block
    return mlp_params + attention_params + layernorm_params

# Estimated parameters for each variant
param_counts = {
    "ViT-Tiny": estimate_vit_parameters(12, 192),
    "ViT-Small": estimate_vit_parameters(12, 384),
    "ViT-Base": estimate_vit_parameters(12, 768),
    "ViT-Large": estimate_vit_parameters(24, 1024),
}

param_counts



# Removing data volumes 10^3 and 10^4 and extending the range to larger volumes
# New range from 10^5 to 10^8
large_data_volumes_range = np.logspace(4, 8, num=100)

# Recompute the loss for each combination of model size and the new data volume range
large_loss_trends = {}
for model_name, params in param_counts.items():
    large_loss_trends[model_name] = [n_coefficient * params**n_exponent +
                                     d_coefficient * D**d_exponent + irreducible_error
                                     for D in large_data_volumes_range]

# Plotting the new trends
plt.figure(figsize=(14, 8))

for model_name, losses in large_loss_trends.items():
    plt.plot(large_data_volumes_range, losses, label=model_name)

plt.title('Loss Trends with Large Data Volumes for Different Model Sizes')
plt.xlabel('Data Volume (log scale)')
plt.ylabel('Loss')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('my_plot.png')
