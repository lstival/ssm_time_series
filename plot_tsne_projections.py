import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the t-SNE coordinates
tsne_encoder_path = r"c:\WUR\CM-Mamba\results\tsne_encoder\tsne_encoder_projection_icml_temporal_20251126_1750_tsne\tsne_coordinates.csv"
tsne_dual_encoder_path = r"c:\WUR\CM-Mamba\results\tsne_dual_encoder\tsne_dual_encoder_projection_icml_dual_20251126_1750_tsne\tsne_coordinates.csv"
tsne_visual_encoder_path = r"c:\WUR\CM-Mamba\results\tsne_visual_encoder\tsne_visual_encoder_projection_icml_visual_20251126_1750_tsne\tsne_coordinates.csv"

# Create output directory
output_dir = r"c:\WUR\CM-Mamba\results\tsne_plots"
os.makedirs(output_dir, exist_ok=True)

# Load data
df_encoder = pd.read_csv(tsne_encoder_path)
df_dual_encoder = pd.read_csv(tsne_dual_encoder_path)
df_visual_encoder = pd.read_csv(tsne_visual_encoder_path)

# Add encoder type to distinguish
df_encoder['encoder_type'] = 'temporal'
df_dual_encoder['encoder_type'] = 'dual'
df_visual_encoder['encoder_type'] = 'visual'

# Combine datasets
df = pd.concat([df_encoder, df_dual_encoder, df_visual_encoder], ignore_index=True)

# Map dataset types to standardized names
dataset_name_mapping = {
    'Electricity': 'Electricity',
    'Ettm': 'Ettm',
    'Etth': 'Ettm',  # Group different ETT versions together
    'Exchange': 'Exchange',
    'PEMS': 'PEMS',
    'Solar': 'Solar',
    'Weather': 'Weather'
}

# Apply mapping
df['dataset_name'] = df['dataset_type'].map(dataset_name_mapping)

# Get unique dataset names and assign colors using tab20 colormap (matching original script)
unique_datasets = sorted(df['dataset_name'].dropna().unique())
colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(unique_datasets))))
dataset_colors = {dataset: colors[i] for i, dataset in enumerate(unique_datasets)}

# Font sizes for paper readability
LABEL_SIZE = 24
TICK_SIZE = 20
LEGEND_SIZE = 28

# Plot settings
plt.rcParams['font.size'] = LABEL_SIZE
plt.rcParams['axes.labelsize'] = LABEL_SIZE
plt.rcParams['xtick.labelsize'] = TICK_SIZE
plt.rcParams['ytick.labelsize'] = TICK_SIZE
plt.rcParams['legend.fontsize'] = LEGEND_SIZE

# 1. Plot individual datasets
for dataset_name in unique_datasets:
    df_subset = df[df['dataset_name'] == dataset_name]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(df_subset['tsne_x'], df_subset['tsne_y'], 
               color=dataset_colors[dataset_name], 
               alpha=0.75, 
               s=16,
               label=dataset_name)
    
    ax.set_xlabel('Component 1', fontsize=LABEL_SIZE)
    ax.set_ylabel('Component 2', fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    # Save individual plot
    output_path = os.path.join(output_dir, f'tsne_{dataset_name.lower()}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

# 2. Create combined figure with 3 subplots (one per encoder type)
fig, axes = plt.subplots(1, 3, figsize=(36, 10))

# Calculate global axis limits for consistent scaling
x_min = df['tsne_x'].min()
x_max = df['tsne_x'].max()
y_min = df['tsne_y'].min()
y_max = df['tsne_y'].max()

# Add some padding
x_padding = (x_max - x_min) * 0.05
y_padding = (y_max - y_min) * 0.05

# Subplot 1: Temporal Encoder
ax = axes[0]
df_temporal = df[df['encoder_type'] == 'temporal']
for dataset_name in unique_datasets:
    df_subset = df_temporal[df_temporal['dataset_name'] == dataset_name]
    if not df_subset.empty:
        ax.scatter(df_subset['tsne_x'], df_subset['tsne_y'], 
                   color=dataset_colors[dataset_name], 
                   alpha=0.75, 
                   s=16,
                   label=dataset_name)

ax.set_xlabel('Component 1', fontsize=LABEL_SIZE)
ax.set_ylabel('Component 2', fontsize=LABEL_SIZE)
ax.tick_params(axis='both', labelsize=TICK_SIZE)
ax.grid(True, alpha=0.2)
ax.set_title('Temporal Only', fontsize=LABEL_SIZE, pad=20)
ax.set_xlim(x_min - x_padding, x_max + x_padding)
ax.set_ylim(y_min - y_padding, y_max + y_padding)

# Subplot 2: Visual Encoder
ax = axes[1]
df_visual = df[df['encoder_type'] == 'visual']
for dataset_name in unique_datasets:
    df_subset = df_visual[df_visual['dataset_name'] == dataset_name]
    if not df_subset.empty:
        ax.scatter(df_subset['tsne_x'], df_subset['tsne_y'], 
                   color=dataset_colors[dataset_name], 
                   alpha=0.75, 
                   s=16,
                   label=dataset_name)

ax.set_xlabel('Component 1', fontsize=LABEL_SIZE)
ax.set_ylabel('Component 2', fontsize=LABEL_SIZE)
ax.tick_params(axis='both', labelsize=TICK_SIZE)
ax.grid(True, alpha=0.2)
ax.set_title('Visual Encoder', fontsize=LABEL_SIZE, pad=20)
ax.set_xlim(x_min - x_padding, x_max + x_padding)
ax.set_ylim(y_min - y_padding, y_max + y_padding)

# Subplot 3: Dual Encoder
ax = axes[2]
df_dual_final = df[df['encoder_type'] == 'dual']
for dataset_name in unique_datasets:
    df_subset = df_dual_final[df_dual_final['dataset_name'] == dataset_name]
    if not df_subset.empty:
        ax.scatter(df_subset['tsne_x'], df_subset['tsne_y'], 
                   color=dataset_colors[dataset_name], 
                   alpha=0.75, 
                   s=16,
                   label=dataset_name)

ax.set_xlabel('Component 1', fontsize=LABEL_SIZE)
ax.set_ylabel('Component 2', fontsize=LABEL_SIZE)
ax.tick_params(axis='both', labelsize=TICK_SIZE)
ax.grid(True, alpha=0.2)
ax.set_title('CM-Mamba', fontsize=LABEL_SIZE, pad=20)
ax.set_xlim(x_min - x_padding, x_max + x_padding)
ax.set_ylim(y_min - y_padding, y_max + y_padding)

# Add shared legend at the bottom
handles = [plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=dataset_colors[name], markersize=10, label=name)
           for name in unique_datasets]
fig.legend(handles=handles, 
           loc='lower center', 
           bbox_to_anchor=(0.5, -0.1), 
           ncol=len(unique_datasets), 
           fontsize=LEGEND_SIZE,
           frameon=False)

plt.tight_layout()

# Save combined plot
output_path = os.path.join(output_dir, 'tsne_combined_3plots.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
print(f"Saved: {output_path}")
plt.close()

# 3. Save each encoder type plot individually
encoder_types = [
    ('temporal', 'Temporal Only', df_temporal),
    ('visual', 'Visual Encoder', df_visual),
    ('dual', 'CM-Mamba', df_dual_final)
]

for encoder_key, encoder_title, df_encoder_data in encoder_types:
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for dataset_name in unique_datasets:
        df_subset = df_encoder_data[df_encoder_data['dataset_name'] == dataset_name]
        if not df_subset.empty:
            ax.scatter(df_subset['tsne_x'], df_subset['tsne_y'], 
                       color=dataset_colors[dataset_name], 
                       alpha=0.75, 
                       s=16,
                       label=dataset_name)
    
    ax.set_xlabel('Component 1', fontsize=LABEL_SIZE)
    ax.set_ylabel('Component 2', fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.2)
    ax.set_title(encoder_title, fontsize=LABEL_SIZE, pad=20)
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Add legend at the bottom
    ax.legend(loc='upper center', 
              bbox_to_anchor=(0.5, -0.1), 
              ncol=len(unique_datasets), 
              fontsize=18, 
              frameon=False)
    
    plt.tight_layout()
    
    # Save individual encoder plot
    output_path = os.path.join(output_dir, f'tsne_{encoder_key}_encoder.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved: {output_path}")
    plt.close()

print(f"\nAll plots saved to: {output_dir}")
