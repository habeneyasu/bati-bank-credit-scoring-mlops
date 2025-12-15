"""
Script to generate and save EDA visualizations for the interim report.
This script creates all required visualizations and saves them to notebooks/eda_outputs/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory
output_dir = Path("eda_outputs")
output_dir.mkdir(exist_ok=True)

# Load data
data_path = Path("../data/raw/data.csv")
df = pd.read_csv(data_path)

print("Generating visualizations...")

# 1. Overview of the Data
print("1. Creating data overview visualization...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Dataset shape info
ax = axes[0, 0]
ax.axis('off')
info_text = f"""
Dataset Overview

Rows: {df.shape[0]:,}
Columns: {df.shape[1]}

Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

Numerical Columns: {len(df.select_dtypes(include=[np.number]).columns)}
Categorical Columns: {len(df.select_dtypes(include=['object']).columns)}
"""
ax.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center', 
        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Data types distribution
ax = axes[0, 1]
dtype_counts = df.dtypes.value_counts()
ax.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', startangle=90)
ax.set_title('Data Types Distribution', fontsize=12, fontweight='bold')

# Memory usage by column
ax = axes[1, 0]
mem_usage = df.memory_usage(deep=True).sort_values(ascending=False).head(10)
ax.barh(range(len(mem_usage)), mem_usage.values / 1024**2, color='steelblue', alpha=0.7)
ax.set_yticks(range(len(mem_usage)))
ax.set_yticklabels(mem_usage.index, fontsize=8)
ax.set_xlabel('Memory Usage (MB)', fontsize=10)
ax.set_title('Top 10 Columns by Memory Usage', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Column count by type
ax = axes[1, 1]
type_counts = {'Numerical': len(df.select_dtypes(include=[np.number]).columns),
                'Categorical': len(df.select_dtypes(include=['object']).columns),
                'Datetime': len(df.select_dtypes(include=['datetime']).columns)}
colors = ['#2ecc71', '#3498db', '#e74c3c']
bars = ax.bar(type_counts.keys(), type_counts.values(), color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Count', fontsize=10)
ax.set_title('Column Count by Type', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, type_counts.values()):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "data_overview.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: data_overview.png")

# 2. Distribution Visualizations - Histograms
print("2. Creating histogram visualizations...")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'PricingStrategy' in numerical_cols:
    numerical_cols.remove('PricingStrategy')  # Skip if it's categorical

n_cols = len(numerical_cols)
n_rows = (n_cols + 2) // 3
fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
axes = axes.flatten() if n_cols > 1 else [axes] if n_cols == 1 else []

for idx, col in enumerate(numerical_cols[:9]):  # Limit to 9 for readability
    ax = axes[idx] if idx < len(axes) else None
    if ax is None:
        break
    
    data = df[col].dropna()
    ax.hist(data, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel(col, fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'Distribution: {col}\nMean: {data.mean():.2f}, Std: {data.std():.2f}', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

# Hide unused subplots
for idx in range(len(numerical_cols), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig(output_dir / "histograms.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: histograms.png")

# 3. Box Plots for Outlier Detection
print("3. Creating box plots for outlier detection...")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'PricingStrategy' in numerical_cols:
    numerical_cols.remove('PricingStrategy')

n_cols = min(len(numerical_cols), 3)
n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
if n_rows == 1 and n_cols == 1:
    axes = [axes]
elif n_rows == 1:
    axes = axes.flatten()
else:
    axes = axes.flatten()

for idx, col in enumerate(numerical_cols[:9]):  # Limit to 9
    if idx >= len(axes):
        break
    ax = axes[idx]
    
    data = df[col].dropna()
    bp = ax.boxplot(data, vert=True, patch_artist=True, showfliers=True)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    # Calculate outliers using IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    outlier_pct = (len(outliers) / len(data)) * 100
    
    ax.set_ylabel(col, fontsize=10)
    ax.set_title(f'Box Plot: {col}\nOutliers: {len(outliers):,} ({outlier_pct:.2f}%)', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

# Hide unused subplots
for idx in range(len(numerical_cols), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig(output_dir / "box_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: box_plots.png")

# 4. Bar Charts for Categorical Features (ProductCategory, ChannelId, ProviderId)
print("4. Creating bar charts for categorical features...")
categorical_features = ['ProductCategory', 'ChannelId', 'ProviderId']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, col in enumerate(categorical_features):
    ax = axes[idx]
    
    value_counts = df[col].value_counts()
    colors = sns.color_palette("husl", len(value_counts))
    
    bars = ax.bar(range(len(value_counts)), value_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(value_counts)))
    ax.set_xticklabels(value_counts.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'{col}\nTotal: {len(value_counts)} categories', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, value_counts.values):
        height = bar.get_height()
        pct = (val / len(df)) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / "categorical_bars.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: categorical_bars.png")

# 5. Correlation with Target Variable (placeholder - will be updated when target is available)
print("5. Creating correlation visualization...")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'PricingStrategy' in numerical_cols:
    numerical_cols.remove('PricingStrategy')

# Calculate correlation matrix
corr_matrix = df[numerical_cols].corr()

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Matrix (Numerical Features)\nNote: Target variable correlation will be available after Task 4', 
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / "target_correlation.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: target_correlation.png")

# 6. Pairwise Scatter Plots for Strong Correlations
print("6. Creating scatter plots for strong correlations...")
# Find strong correlations
strong_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) >= 0.7:  # Strong correlation threshold
            strong_corr_pairs.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_val
            })

if strong_corr_pairs:
    top_pairs = pd.DataFrame(strong_corr_pairs).sort_values('Correlation', key=abs, ascending=False).head(6)
    
    n_pairs = len(top_pairs)
    n_rows = (n_pairs + 1) // 2
    n_cols = 2
    
    if n_pairs == 1:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        axes = [ax]
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        axes = axes.flatten() if hasattr(axes, 'flatten') else np.array(axes).flatten()
    
    for plot_idx, (_, row) in enumerate(top_pairs.iterrows()):
        if plot_idx < len(axes):
            ax = axes[plot_idx]
            feat1 = row['Feature 1']
            feat2 = row['Feature 2']
            corr_val = row['Correlation']
            
            # Sample if too large
            sample_size = min(5000, len(df))
            df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
            
            ax.scatter(df_sample[feat1], df_sample[feat2], alpha=0.5, s=10)
            ax.set_xlabel(feat1, fontsize=10)
            ax.set_ylabel(feat2, fontsize=10)
            ax.set_title(f'{feat1} vs {feat2}\nCorrelation: {corr_val:.3f}', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for plot_idx in range(n_pairs, len(axes)):
        axes[plot_idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: scatter_plots.png")
else:
    print("   ⚠ No strong correlations found for scatter plots")

print("\n✅ All visualizations generated successfully!")
print(f"📁 Output directory: {output_dir.absolute()}")

