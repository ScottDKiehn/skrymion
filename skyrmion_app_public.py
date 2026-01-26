import matplotlib
matplotlib.use('Agg')  # CRITICAL: Use non-interactive backend for Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import zipfile
import io
from skimage import io as skio

# Import your analysis functions
from stat_analyzer_public import (
    load_skyrmion_file,
    parse_filename,
    calculate_basic_stats,
    calculate_voronoi_coordination,
    calculate_bond_orientation_order,
    visualize_skyrmions,
    visualize_voronoi,
    print_stats,
    print_coordination_stats
)

# Import polar Fourier classifier for topological charge classification
try:
    from polar_fourier_classifier_public import PolarFourierClassifier, classify_from_dataframe
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Skyrmion Data Analyzer",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


def process_uploaded_file(uploaded_file, neighbor_distance_factor=3.0):
    """
    Process a single uploaded file.

    Returns:
    --------
    df : DataFrame with skyrmion data
    metadata : dict with file metadata
    stats : dict with basic statistics
    coord_stats : dict with coordination statistics
    bond_order_stats : dict with bond orientation order statistics
    df_with_coord : DataFrame with coordination numbers and bond order
    vor : Voronoi object
    neighbor_dict : dict with neighbor relationships
    """
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Load data
        df = load_skyrmion_file(tmp_path)
        df = df[['Area', 'X', 'Y']].dropna()

        # Parse metadata from ORIGINAL filename, not temp path
        metadata = parse_filename(uploaded_file.name)  # ← FIXED: use original name
        metadata['filename'] = uploaded_file.name

        # Calculate basic statistics
        stats = calculate_basic_stats(df, metadata)

        # Calculate coordination (both distance-aware and topological)
        coord_stats, df_with_coord, vor, neighbor_dict = calculate_voronoi_coordination(
            df, neighbor_distance_factor=neighbor_distance_factor
        )

        # Calculate bond orientation order parameter
        bond_order_stats = calculate_bond_orientation_order(df_with_coord, neighbor_dict)

        # Add local bond order to dataframe
        df_with_coord['bond_order'] = bond_order_stats['local_phi']

        return df, metadata, stats, coord_stats, bond_order_stats, df_with_coord, vor, neighbor_dict

    finally:
        # Clean up temp file
        Path(tmp_path).unlink()


def fig_to_bytes(fig, format='png', dpi=300):
    """Convert matplotlib figure to bytes for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


def display_stats_cards(stats, coord_stats, bond_order_stats=None):
    """Display statistics in nice cards."""
    # Main metrics in 5 columns
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("🔢 Skyrmion Count", f"{stats['n_skyrmions']}")
        st.metric("📏 Mean Area", f"{stats['mean_area']:.1f} px²")

    with col2:
        st.metric("🌡️ Temperature", f"{stats['temperature']} K")
        st.metric("🧲 Field", f"{stats['field']} Oe")

    with col3:
        st.metric("📊 Density", f"{stats['number_density']:.6f} /px²")
        st.metric("📐 Coverage", f"{stats['area_coverage']:.2%}")

    with col4:
        st.metric("🔗 Mean Coord.", f"{coord_stats['mean_coordination']:.2f}")
        st.metric("📦 Packing Eff.", f"{coord_stats['packing_efficiency']:.2%}")

    with col5:
        if bond_order_stats:
            st.metric("🔷 Mean Bond Order", f"{bond_order_stats['mean_local_phi']:.3f}")
            st.metric("🌐 Global φ", f"{bond_order_stats['global_phi_magnitude']:.3f}")

    # Add CN distribution summary below cards
    coord_dist = coord_stats['coordination_distribution']
    cn_values = np.arange(len(coord_dist))
    mask = coord_dist > 0
    cn_present = cn_values[mask]
    counts_present = coord_dist[mask]
    percentages = (counts_present / coord_dist.sum()) * 100

    # Show top 3 coordination numbers
    top_indices = np.argsort(percentages)[::-1][:3]  # Top 3
    top_cn_text = ", ".join([f"CN={cn_present[i]}: {percentages[i]:.1f}%" for i in top_indices])

    st.caption(f"**Top Coordination Numbers**: {top_cn_text}")

    # Add bond order interpretation
    if bond_order_stats:
        mean_phi = bond_order_stats['mean_local_phi']
        if mean_phi > 0.9:
            order_interpretation = "Highly ordered hexagonal lattice"
        elif mean_phi > 0.7:
            order_interpretation = "Moderately ordered lattice"
        elif mean_phi > 0.5:
            order_interpretation = "Weakly ordered lattice"
        else:
            order_interpretation = "Disordered/liquid-like state"
        st.caption(f"**Lattice Order**: {order_interpretation}")


def plot_interactive_scatter(df_with_coord, metadata, scale_factor=1.0, unit_name='pixels'):
    """Create an interactive scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Apply scale factor for display
    x_scaled = df_with_coord['X'] * scale_factor
    y_scaled = df_with_coord['Y'] * scale_factor

    scatter = ax.scatter(
        x_scaled,
        y_scaled,
        s=df_with_coord['Area'] / 5,  # Keep visual size constant (don't scale with units)
        alpha=0.6,
        c=df_with_coord['coordination'],
        cmap='RdYlGn',
        vmin=3,
        vmax=7,
        edgecolors='black',
        linewidth=0.5
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Coordination Number', rotation=270, labelpad=20)

    ax.set_xlabel(f'X Position ({unit_name})', fontsize=12)
    ax.set_ylabel(f'Y Position ({unit_name})', fontsize=12)
    ax.set_title(f"Field={metadata['field']}Oe, T={metadata['temperature']}K",
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    return fig


def plot_voronoi_diagram(df_with_coord, vor, metadata, scale_factor=1.0, unit_name='pixels'):
    """Create Voronoi diagram."""
    from scipy.spatial import voronoi_plot_2d, Voronoi

    fig, ax = plt.subplots(figsize=(12, 12))

    # Create scaled Voronoi for visualization if needed
    if scale_factor != 1.0:
        # Recompute Voronoi with scaled coordinates
        scaled_points = df_with_coord[['X', 'Y']].values * scale_factor
        vor_display = Voronoi(scaled_points)
    else:
        vor_display = vor

    # Draw Voronoi diagram
    voronoi_plot_2d(vor_display, ax=ax, show_vertices=False, line_colors='gray',
                    line_width=1, line_alpha=0.6, point_size=0)

    # Apply scale factor for display
    x_scaled = df_with_coord['X'] * scale_factor
    y_scaled = df_with_coord['Y'] * scale_factor

    scatter = ax.scatter(
        x_scaled,
        y_scaled,
        c=df_with_coord['coordination'],
        s=100,
        cmap='RdYlGn',
        vmin=3,
        vmax=7,
        edgecolors='black',
        linewidth=1,
        zorder=10
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Coordination Number', rotation=270, labelpad=20)

    ax.set_xlabel(f'X Position ({unit_name})', fontsize=12)
    ax.set_ylabel(f'Y Position ({unit_name})', fontsize=12)
    ax.set_title(f"Voronoi Tessellation: Field={metadata['field']}Oe, T={metadata['temperature']}K",
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.invert_yaxis()

    return fig

def plot_size_histogram(skyrmion_data, metadata, bins='auto', show_kde=True, scale_factor=1.0, unit_name='pixels'):
    """Create histogram of skyrmion sizes - Streamlit version."""
    from scipy import stats

    fig, ax = plt.subplots(figsize=(10, 6))

    areas = skyrmion_data['Area'].values
    # Convert area to diameter: d = 2 * sqrt(A/π)
    diameters = 2 * np.sqrt(areas / np.pi) * scale_factor

    # Create histogram with percentage y-axis
    weights = np.ones_like(diameters) / len(diameters) * 100

    # Fix: Compute bin edges first if using automatic estimation (doesn't work with weights)
    if isinstance(bins, str):
        bins = np.histogram_bin_edges(diameters, bins=bins)

    n, bins_edges, patches = ax.hist(diameters, bins=bins, alpha=0.7,
                                       color='steelblue', edgecolor='black',
                                       weights=weights, label='Data')

    # Add KDE overlay if requested (scaled to percentage)
    if show_kde:
        kde = stats.gaussian_kde(diameters)
        x_range = np.linspace(diameters.min(), diameters.max(), 200)
        # Scale KDE to match percentage histogram
        kde_values = kde(x_range)
        # Integrate KDE to get proper scaling
        bin_width = (diameters.max() - diameters.min()) / (len(np.histogram(diameters, bins=bins)[0]))
        kde_scaled = kde_values * bin_width * 100
        ax.plot(x_range, kde_scaled, 'r-', linewidth=2,
                label='Kernel Density Estimate')

    # Statistical lines
    mean_diameter = diameters.mean()
    median_diameter = np.median(diameters)
    ax.axvline(mean_diameter, color='darkgreen', linestyle='--', linewidth=2,
               label=f'Mean: {mean_diameter:.1f}')
    ax.axvline(median_diameter, color='orange', linestyle='--', linewidth=2,
               label=f'Median: {median_diameter:.1f}')

    # Labels and title
    ax.set_xlabel(f'Skyrmion Diameter ({unit_name})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')

    title = f"Size Distribution: Field={metadata['field']}Oe, T={metadata['temperature']}K"
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = f"n = {len(diameters)}\n"
    stats_text += f"Mean Diameter = {mean_diameter:.1f} ± {diameters.std():.1f} {unit_name}\n"
    stats_text += f"Median Diameter = {median_diameter:.1f} {unit_name}\n"
    stats_text += f"Range = [{diameters.min():.1f}, {diameters.max():.1f}] {unit_name}\n"
    stats_text += f"Skewness = {stats.skew(diameters):.2f}\n"
    stats_text += f"Kurtosis = {stats.kurtosis(diameters):.2f}"

    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    return fig


def plot_size_boxplot(skyrmion_data, metadata, scale_factor=1.0, unit_name='pixels'):
    """Create box and whisker plot of skyrmion sizes - Streamlit version."""
    from scipy import stats

    fig, ax = plt.subplots(figsize=(10, 6))

    areas = skyrmion_data['Area'].values
    # Convert area to diameter: d = 2 * sqrt(A/π)
    diameters = 2 * np.sqrt(areas / np.pi) * scale_factor

    # Create box plot
    bp = ax.boxplot([diameters],
                     vert=True,
                     patch_artist=True,
                     widths=0.5,
                     showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=10),
                     medianprops=dict(color='darkgreen', linewidth=2),
                     boxprops=dict(facecolor='steelblue', alpha=0.7, linewidth=2),
                     whiskerprops=dict(linewidth=2),
                     capprops=dict(linewidth=2),
                     flierprops=dict(marker='o', markerfacecolor='orange', markersize=6, alpha=0.5))

    # Add scatter points with jitter for better visibility
    y = diameters
    x = np.random.normal(1, 0.04, size=len(y))  # Add horizontal jitter
    ax.scatter(x, y, alpha=0.3, s=20, color='gray', zorder=1)

    # Calculate statistics
    q1 = np.percentile(diameters, 25)
    q3 = np.percentile(diameters, 75)
    median = np.median(diameters)
    mean = diameters.mean()
    iqr = q3 - q1
    lower_whisker = max(diameters.min(), q1 - 1.5*iqr)
    upper_whisker = min(diameters.max(), q3 + 1.5*iqr)

    # Add horizontal reference lines for quartiles
    ax.axhline(median, color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Median: {median:.1f}')
    ax.axhline(mean, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mean:.1f}')
    ax.axhline(q1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(q3, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Labels and title
    ax.set_ylabel(f'Skyrmion Diameter ({unit_name})', fontsize=12, fontweight='bold')
    ax.set_xticks([1])
    ax.set_xticklabels(['Diameter Distribution'])

    title = f"Box Plot: Field={metadata['field']}Oe, T={metadata['temperature']}K"
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    # Add statistics text box
    stats_text = f"n = {len(diameters)}\n"
    stats_text += f"Min = {diameters.min():.1f} {unit_name}\n"
    stats_text += f"Q1 (25%) = {q1:.1f} {unit_name}\n"
    stats_text += f"Median (50%) = {median:.1f} {unit_name}\n"
    stats_text += f"Mean = {mean:.1f} {unit_name}\n"
    stats_text += f"Q3 (75%) = {q3:.1f} {unit_name}\n"
    stats_text += f"Max = {diameters.max():.1f} {unit_name}\n"
    stats_text += f"IQR = {iqr:.1f} {unit_name}\n"
    stats_text += f"Outliers (beyond 1.5×IQR) = {np.sum((diameters < lower_whisker) | (diameters > upper_whisker))}"

    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    return fig


def plot_coordination_distribution(coord_stats, metadata):
    """Create bar chart showing coordination number distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))

    coord_dist = coord_stats['coordination_distribution']
    cn_values = np.arange(len(coord_dist))

    # Only plot CN values that actually exist
    mask = coord_dist > 0
    cn_values_present = cn_values[mask]
    counts_present = coord_dist[mask]

    # Convert counts to percentages
    total = coord_dist.sum()
    percentages = (counts_present / total) * 100

    # Color coding: green for CN=6 (ideal), gradient for others
    colors = []
    for cn in cn_values_present:
        if cn == 6:
            colors.append('#2ca02c')  # Green for ideal packing
        elif cn == 5 or cn == 7:
            colors.append('#ff7f0e')  # Orange for near-ideal
        else:
            colors.append('#d62728')  # Red for defects

    bars = ax.bar(cn_values_present, percentages, color=colors,
                   edgecolor='black', alpha=0.7, width=0.6)

    ax.set_xlabel('Coordination Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraction (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Coordination Distribution: Field={metadata["field"]}Oe, T={metadata["temperature"]}K',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(cn_values_present)
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage labels on bars
    for bar, pct, cn, count in zip(bars, percentages, cn_values_present, counts_present):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%\n(n={int(count)})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add legend explaining colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', edgecolor='black', alpha=0.7, label='CN=6 (Ideal hexagonal)'),
        Patch(facecolor='#ff7f0e', edgecolor='black', alpha=0.7, label='CN=5,7 (Near-ideal)'),
        Patch(facecolor='#d62728', edgecolor='black', alpha=0.7, label='CN≤4,≥8 (Defects)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Add summary text
    summary_text = f"Mean CN: {coord_stats['mean_coordination']:.2f}\n"
    summary_text += f"Packing Efficiency: {coord_stats['packing_efficiency']:.1%}"
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


def plot_phi_scatter(df_with_coord, metadata, scale_factor=1.0, unit_name='pixels'):
    """Create scatter plot colored by local bond orientation order parameter |φ|."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Apply scale factor for display
    x_scaled = df_with_coord['X'] * scale_factor
    y_scaled = df_with_coord['Y'] * scale_factor

    scatter = ax.scatter(
        x_scaled,
        y_scaled,
        s=df_with_coord['Area'] / 5,
        alpha=0.6,
        c=df_with_coord['bond_order'],
        cmap='viridis',  # viridis: purple (disordered) → yellow (ordered)
        vmin=0,
        vmax=1,
        edgecolors='black',
        linewidth=0.5
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Bond Orientation Order |φ|', rotation=270, labelpad=20)

    ax.set_xlabel(f'X Position ({unit_name})', fontsize=12)
    ax.set_ylabel(f'Y Position ({unit_name})', fontsize=12)
    ax.set_title(f"Bond Order: Field={metadata['field']}Oe, T={metadata['temperature']}K",
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    return fig


def plot_cn_phi_correlation(df_with_coord, metadata, scale_factor=1.0, unit_name='pixels'):
    """Create correlation plot showing relationship between CN and bond orientation order."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Size points by skyrmion area
    sizes = df_with_coord['Area'] * (scale_factor ** 2) / 10

    scatter = ax.scatter(
        df_with_coord['coordination'],
        df_with_coord['bond_order'],
        s=sizes,
        alpha=0.5,
        c=df_with_coord['bond_order'],
        cmap='viridis',
        edgecolors='black',
        linewidth=0.5
    )

    # Calculate and display correlation
    correlation = np.corrcoef(df_with_coord['coordination'], df_with_coord['bond_order'])[0, 1]

    ax.set_xlabel('Coordination Number (CN)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bond Orientation Order |φ|', fontsize=12, fontweight='bold')
    ax.set_title(f"CN vs Bond Order: Field={metadata['field']}Oe, T={metadata['temperature']}K\n" +
                 f"Correlation: {correlation:.3f}",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

    # Add reference lines
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='High order threshold (0.9)')
    ax.axvline(x=6, color='green', linestyle='--', alpha=0.5, label='Ideal CN (6)')
    ax.legend(loc='lower right')

    # Add text box with statistics
    stats_text = f"Mean CN: {df_with_coord['coordination'].mean():.2f}\n"
    stats_text += f"Mean |φ|: {df_with_coord['bond_order'].mean():.3f}\n"
    stats_text += f"Correlation: {correlation:.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


def plot_interactive_scatter_toggle(df_with_coord, metadata, color_by='coordination', scale_factor=1.0, unit_name='pixels'):
    """Create scatter plot with toggle between CN and bond order coloring."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Apply scale factor for display
    x_scaled = df_with_coord['X'] * scale_factor
    y_scaled = df_with_coord['Y'] * scale_factor

    if color_by == 'coordination':
        c_values = df_with_coord['coordination']
        cmap = 'RdYlGn'
        vmin, vmax = 3, 7
        label = 'Coordination Number'
    else:  # color_by == 'bond_order'
        c_values = df_with_coord['bond_order']
        cmap = 'viridis'
        vmin, vmax = 0, 1
        label = 'Bond Orientation Order |φ|'

    scatter = ax.scatter(
        x_scaled,
        y_scaled,
        s=df_with_coord['Area'] / 5,
        alpha=0.6,
        c=c_values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors='black',
        linewidth=0.5
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(label, rotation=270, labelpad=20)

    ax.set_xlabel(f'X Position ({unit_name})', fontsize=12)
    ax.set_ylabel(f'Y Position ({unit_name})', fontsize=12)
    ax.set_title(f"Skyrmion Lattice: Field={metadata['field']}Oe, T={metadata['temperature']}K",
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    return fig


def assess_distribution_modality(skyrmion_data, scale_factor=1.0, unit_name='pixels'):
    """Assess whether size distribution is unimodal or multimodal."""
    from scipy import stats

    areas = skyrmion_data['Area'].values
    # Convert to diameter for assessment
    diameters = 2 * np.sqrt(areas / np.pi) * scale_factor

    # Basic statistics
    skewness = stats.skew(diameters)
    kurtosis = stats.kurtosis(diameters)

    # Simple bimodality coefficient (BC)
    n = len(diameters)
    BC = (skewness**2 + 1) / (kurtosis + 3 * (n-1)**2 / ((n-2)*(n-3)))

    assessment = {
        'n_skyrmions': n,
        'mean': diameters.mean(),
        'median': np.median(diameters),
        'std': diameters.std(),
        'min': diameters.min(),
        'max': diameters.max(),
        'range': diameters.max() - diameters.min(),
        'q1': np.percentile(diameters, 25),
        'q3': np.percentile(diameters, 75),
        'iqr': np.percentile(diameters, 75) - np.percentile(diameters, 25),
        'skewness': skewness,
        'kurtosis': kurtosis,
        'bimodality_coefficient': BC,
        'likely_bimodal': BC > 0.555 if n > 20 else None,
        'interpretation': '',
        'unit_name': unit_name
    }

    # Interpretation
    if n < 20:
        assessment['interpretation'] = "Sample size too small for reliable modality assessment"
    elif BC > 0.555:
        assessment['interpretation'] = f"Distribution suggests bimodality (BC={BC:.3f} > 0.555). This may indicate multiple topological charge populations."
    else:
        assessment['interpretation'] = f"Distribution appears unimodal (BC={BC:.3f} < 0.555). This suggests a single topological charge phase."

    return assessment


def create_summary_dataframe(filename, metadata, stats, coord_stats, modality, scale_factor, unit_display, bond_order_stats=None):
    """
    Create a summary statistics DataFrame for CSV export.

    Returns a single-row DataFrame containing all aggregate metrics.
    """
    # Extract metadata (with fallbacks)
    field = metadata.get('field', None) if metadata else None
    temperature = metadata.get('temperature', None) if metadata else None
    sample_id = metadata.get('id', '') if metadata else ''

    # Build summary dictionary
    summary = {
        # Experimental metadata
        'filename': filename,
        'field_oe': field,
        'temperature_k': temperature,
        'sample_id': sample_id,

        # Basic statistics
        'n_skyrmions': stats['n_skyrmions'],
        'number_density': stats['number_density'],
        'area_coverage': stats['area_coverage'],
        'fov_width_px': stats['fov_width'],
        'fov_height_px': stats['fov_height'],
        'fov_area_px2': stats['fov_area'],

        # Size distribution statistics in pixels
        'mean_area_px2': stats['mean_area'],
        'median_area_px2': stats['median_area'],
        'std_area_px2': stats['std_area'],
        'min_area_px2': stats['min_area'],
        'max_area_px2': stats['max_area'],

        # Size distribution statistics (converted units)
        f'mean_area_{unit_display}2': stats['mean_area'] * (scale_factor ** 2),
        f'median_area_{unit_display}2': stats['median_area'] * (scale_factor ** 2),
        f'std_area_{unit_display}2': stats['std_area'] * (scale_factor ** 2),
        f'min_area_{unit_display}2': stats['min_area'] * (scale_factor ** 2),
        f'max_area_{unit_display}2': stats['max_area'] * (scale_factor ** 2),

        # Size distribution modality metrics (already in converted units from modality)
        f'mean_diameter_{unit_display}': modality['mean'],
        f'median_diameter_{unit_display}': modality['median'],
        f'std_diameter_{unit_display}': modality['std'],
        f'min_diameter_{unit_display}': modality['min'],
        f'q1_diameter_{unit_display}': modality['q1'],
        f'q3_diameter_{unit_display}': modality['q3'],
        f'max_diameter_{unit_display}': modality['max'],
        f'range_diameter_{unit_display}': modality['range'],
        f'iqr_diameter_{unit_display}': modality['iqr'],
        'skewness': modality['skewness'],
        'kurtosis': modality['kurtosis'],
        'bimodality_coefficient': modality['bimodality_coefficient'],
        'likely_bimodal': modality.get('likely_bimodal', None),

        # Coordination statistics (distance-aware)
        'mean_coordination': coord_stats['mean_coordination'],
        'median_coordination': coord_stats['median_coordination'],
        'std_coordination': coord_stats['std_coordination'],
        'min_coordination': coord_stats['min_coordination'],
        'max_coordination': coord_stats['max_coordination'],
        'packing_efficiency': coord_stats['packing_efficiency'],

        # Topological coordination (for comparison)
        'mean_topological_coordination': coord_stats.get('mean_topological_coordination', None),
        'topological_packing_efficiency': coord_stats.get('topological_packing_efficiency', None),

        # Mean radius
        'mean_radius_px': coord_stats.get('mean_radius_pixels', None),
        'std_radius_px': coord_stats.get('std_radius_pixels', None),

        # Units
        'scale_factor': scale_factor,
        'unit_name': unit_display
    }

    # Add bond orientation order statistics if provided
    if bond_order_stats:
        summary.update({
            'mean_local_bond_order': bond_order_stats['mean_local_phi'],
            'std_local_bond_order': bond_order_stats['std_local_phi'],
            'median_local_bond_order': bond_order_stats['median_local_phi'],
            'min_local_bond_order': bond_order_stats['min_local_phi'],
            'max_local_bond_order': bond_order_stats['max_local_phi'],
            'global_bond_order_magnitude': bond_order_stats['global_phi_magnitude'],
        })

    # Convert to single-row DataFrame
    return pd.DataFrame([summary])


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.title("🧲 Skyrmion Data Analyzer")
    st.markdown("### LTEM Skyrmion Analysis with Voronoi Coordination")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Select Mode:",
                            ["📄 Single File Analysis",
                             "📦 Batch Analysis",
                             "🤖 ML Detection"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Unit Conversion")

    # Scale factor input
    scale_factor = st.sidebar.number_input(
        "Length Scale (unit/pixel)",
        min_value=0.0,
        value=1.0,
        step=0.1,
        format="%.3f",
        help="Enter the scale factor to convert pixels to physical units (e.g., 5.2 for nm/pixel). Default is 1.0 (pixels)."
    )

    unit_name = st.sidebar.text_input(
        "Unit Name",
        value="pixels" if scale_factor == 1.0 else "nm",
        help="Name of the physical unit (e.g., 'nm', 'μm', 'Å')"
    )

    # Update unit name display
    if scale_factor == 1.0:
        unit_display = "pixels"
    else:
        unit_display = unit_name

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Coordination Settings")

    neighbor_distance_factor = st.sidebar.slider(
        "Neighbor Distance Factor",
        min_value=1.0,
        max_value=4.0,
        value=3.0,
        step=0.1,
        help="Multiplier for neighbor cutoff distance. Default 3.0 works for typical skyrmion lattices."
    )

    st.sidebar.caption(f"**Cutoff:** {neighbor_distance_factor:.1f} × avg(r_i, r_j)")
    st.sidebar.caption("**Tip:** Decrease to 2.0 for denser packing, increase to 3.5+ for very sparse lattices")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This app analyzes LTEM skyrmion data, calculating:\n"
        "- Basic statistics (size, density)\n"
        "- Voronoi coordination numbers\n"
        "- Packing efficiency\n\n"
        "Upload .xlsx or .csv files with Area, X, Y columns."
    )
    
    # ============================================================
    # SINGLE FILE MODE
    # ============================================================
    if mode == "📄 Single File Analysis":
        st.header("Single File Analysis")
        st.markdown("Upload a single skyrmion data file for detailed analysis.")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['xlsx', 'csv'],
            help="Upload a .xlsx or .csv file with skyrmion data"
        )

        # Note: Q Classification has been moved to ML Detection mode for better workflow

        if uploaded_file is not None:
            with st.spinner('🔄 Processing file...'):
                try:
                    df, metadata, stats, coord_stats, bond_order_stats, df_with_coord, vor, neighbor_dict = process_uploaded_file(
                        uploaded_file, neighbor_distance_factor
                    )
                    
                    st.success(f"✅ Successfully loaded: {uploaded_file.name}")
                    
                    # Display statistics
                    st.markdown("---")
                    st.subheader("📊 Statistics")
                    display_stats_cards(stats, coord_stats, bond_order_stats)
                    
                    # NEW: Size Distribution Analysis
                    st.markdown("---")
                    st.subheader("📈 Size Distribution Analysis")

                    from scipy import stats as scipy_stats
                    modality = assess_distribution_modality(df_with_coord, scale_factor, unit_display)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Bimodality Coefficient", f"{modality['bimodality_coefficient']:.3f}")
                    with col2:
                        st.metric("Skewness", f"{modality['skewness']:.2f}")
                    with col3:
                        st.metric("Kurtosis", f"{modality['kurtosis']:.2f}")
                    
                    # Interpretation
                    if modality['likely_bimodal'] is not None:
                        if modality['likely_bimodal']:
                            st.info("🔬 " + modality['interpretation'])
                        else:
                            st.success("✓ " + modality['interpretation'])
                    else:
                        st.warning("⚠️ " + modality['interpretation'])
                    
                    # Detailed stats in expander
                    with st.expander("🔍 View Detailed Statistics"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Basic Statistics**")
                            st.json({
                                "Mean Area": f"{stats['mean_area']:.2f}",
                                "Median Area": f"{stats['median_area']:.2f}",
                                "Std Area": f"{stats['std_area']:.2f}",
                                "Min Area": f"{stats['min_area']:.0f}",
                                "Max Area": f"{stats['max_area']:.0f}",
                                "FOV Width": f"{stats['fov_width']:.1f}",
                                "FOV Height": f"{stats['fov_height']:.1f}",
                            })
                        
                        with col2:
                            st.markdown("**Coordination Statistics**")
                            st.json({
                                "Mean Coordination": f"{coord_stats['mean_coordination']:.2f}",
                                "Median Coordination": f"{coord_stats['median_coordination']:.1f}",
                                "Std Coordination": f"{coord_stats['std_coordination']:.2f}",
                                "Min Coordination": int(coord_stats['min_coordination']),
                                "Max Coordination": int(coord_stats['max_coordination']),
                            })
                    
                    # Visualizations
                    st.markdown("---")
                    st.subheader("📈 Visualizations")

                    # Standard 6-tab layout (Q Classification moved to ML Detection mode)
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                        "🎯 Overview", "🕸️ Voronoi", "📊 Size Distribution",
                        "🔗 Coordination", "🔷 Bond Order", "📋 Data & Export"
                    ])

                    # Store all figures for bulk download
                    all_figures = {}

                    with tab1:
                        st.markdown("### Skyrmion Lattice Overview")
                        st.markdown("Toggle between coordination number and bond orientation order coloring.")

                        # Toggle selector
                        color_mode = st.radio(
                            "Color by:",
                            options=['Coordination Number (CN)', 'Bond Orientation Order (φ)'],
                            horizontal=True
                        )

                        color_by = 'coordination' if 'Coordination' in color_mode else 'bond_order'

                        fig_overview = plot_interactive_scatter_toggle(df_with_coord, metadata, color_by, scale_factor, unit_display)
                        st.pyplot(fig_overview)
                        all_figures['overview_scatter'] = fig_overview

                        # Download button for this figure
                        st.download_button(
                            label=f"📥 Download {color_mode} Scatter Plot",
                            data=fig_to_bytes(fig_overview),
                            file_name=f"scatter_{color_by}_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                            mime="image/png"
                        )
                        plt.close()

                    with tab2:
                        st.markdown("### Voronoi Tessellation")
                        st.markdown("Voronoi cells show regions of space closest to each skyrmion. Shared edges indicate topological neighbors.")

                        fig_voronoi = plot_voronoi_diagram(df_with_coord, vor, metadata, scale_factor, unit_display)
                        st.pyplot(fig_voronoi)
                        all_figures['voronoi'] = fig_voronoi

                        st.download_button(
                            label="📥 Download Voronoi Diagram",
                            data=fig_to_bytes(fig_voronoi),
                            file_name=f"voronoi_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                            mime="image/png"
                        )
                        plt.close()

                    with tab3:
                        # Size histogram tab
                        st.markdown("### Skyrmion Size Distribution")
                        st.markdown("""
                        **Hypothesis**: Bimodal distributions may indicate multiple topological charge populations,
                        while unimodal distributions suggest a single topological charge phase.
                        """)

                        # Create histogram with user controls
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            bins_choice = st.selectbox("Bins", ['auto', 'sturges', 'fd', 'sqrt', 10, 15, 20, 30])
                            show_kde = st.checkbox("Show KDE", value=True)

                        fig_histogram = plot_size_histogram(df_with_coord, metadata, bins=bins_choice, show_kde=show_kde,
                                                    scale_factor=scale_factor, unit_name=unit_display)
                        st.pyplot(fig_histogram)
                        all_figures['size_histogram'] = fig_histogram

                        st.download_button(
                            label="📥 Download Size Histogram",
                            data=fig_to_bytes(fig_histogram),
                            file_name=f"histogram_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                            mime="image/png"
                        )
                        plt.close()

                        # Box and whisker plot
                        st.markdown("---")
                        st.markdown("### Box and Whisker Plot")
                        st.markdown("""
                        **Box plot** shows the five-number summary (min, Q1, median, Q3, max) and outliers.
                        The box spans Q1 to Q3 (interquartile range), with the median line inside.
                        Whiskers extend to 1.5×IQR, and points beyond are marked as outliers.
                        """)

                        fig_boxplot = plot_size_boxplot(df_with_coord, metadata, scale_factor=scale_factor, unit_name=unit_display)
                        st.pyplot(fig_boxplot)
                        all_figures['size_boxplot'] = fig_boxplot

                        st.download_button(
                            label="📥 Download Box Plot",
                            data=fig_to_bytes(fig_boxplot),
                            file_name=f"boxplot_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                            mime="image/png"
                        )
                        plt.close()

                        # Additional statistical tests
                        with st.expander("📊 Advanced Statistical Tests"):
                            st.markdown("**Normality Test (Shapiro-Wilk)**")
                            from scipy import stats as scipy_stats
                            stat, p_value = scipy_stats.shapiro(df_with_coord['Area'].values)
                            st.write(f"Test statistic: {stat:.4f}, p-value: {p_value:.4f}")
                            if p_value < 0.05:
                                st.write("✗ Distribution is **not normally distributed** (p < 0.05)")
                            else:
                                st.write("✓ Distribution is consistent with normal distribution (p ≥ 0.05)")

                    with tab4:
                        # Coordination distribution tab
                        st.markdown("### Coordination Analysis")
                        st.markdown("""
                        **Coordination number** (CN) is calculated using distance-aware method with adaptive cutoff.
                        Ideal hexagonal packing has CN=6. Deviations indicate structural disorder or defects.
                        """)

                        # Generate text summary of CN fractions
                        coord_dist = coord_stats['coordination_distribution']
                        cn_values = np.arange(len(coord_dist))
                        mask = coord_dist > 0
                        cn_present = cn_values[mask]
                        counts_present = coord_dist[mask]
                        percentages = (counts_present / coord_dist.sum()) * 100

                        cn_summary = ", ".join([f"{pct:.1f}% with CN={cn}" for cn, pct in zip(cn_present, percentages)])
                        st.info(f"**Distribution**: {cn_summary}")

                        # Plot coordination distribution
                        fig_coord_dist = plot_coordination_distribution(coord_stats, metadata)
                        st.pyplot(fig_coord_dist)
                        all_figures['coordination_distribution'] = fig_coord_dist

                        st.download_button(
                            label="📥 Download Coordination Distribution",
                            data=fig_to_bytes(fig_coord_dist),
                            file_name=f"coord_dist_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                            mime="image/png"
                        )
                        plt.close()

                        # CN vs Bond Order Correlation
                        st.markdown("---")
                        st.markdown("### Coordination vs Bond Orientation Correlation")
                        st.markdown("""
                        Correlation between CN and bond orientation order reveals the interplay between
                        structural packing and local hexagonal symmetry.
                        """)

                        fig_correlation = plot_cn_phi_correlation(df_with_coord, metadata, scale_factor, unit_display)
                        st.pyplot(fig_correlation)
                        all_figures['cn_phi_correlation'] = fig_correlation

                        st.download_button(
                            label="📥 Download CN vs φ Correlation",
                            data=fig_to_bytes(fig_correlation),
                            file_name=f"cn_phi_correlation_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                            mime="image/png"
                        )
                        plt.close()

                    with tab5:
                        # Bond orientation order tab
                        st.markdown("### Bond Orientation Order Parameter")
                        st.markdown("""
                        **Bond orientation order** (φ) measures local hexagonal symmetry:
                        - φ = 1: Perfect hexagonal order
                        - φ = 0: Random/disordered orientation
                        - Calculated using 6-fold symmetry: φ = (1/N)∑exp(i×6×θ)
                        """)

                        # Display bond order statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean |φ|", f"{bond_order_stats['mean_local_phi']:.3f}")
                        with col2:
                            st.metric("Global |φ|", f"{bond_order_stats['global_phi_magnitude']:.3f}")
                        with col3:
                            mean_phi = bond_order_stats['mean_local_phi']
                            if mean_phi > 0.9:
                                order_label = "Highly Ordered"
                                delta_color = "normal"
                            elif mean_phi > 0.7:
                                order_label = "Moderately Ordered"
                                delta_color = "normal"
                            else:
                                order_label = "Disordered"
                                delta_color = "inverse"
                            st.metric("Lattice State", order_label)

                        # Plot bond order scatter
                        fig_phi_scatter = plot_phi_scatter(df_with_coord, metadata, scale_factor, unit_display)
                        st.pyplot(fig_phi_scatter)
                        all_figures['phi_scatter'] = fig_phi_scatter

                        st.download_button(
                            label="📥 Download Bond Order Scatter Plot",
                            data=fig_to_bytes(fig_phi_scatter),
                            file_name=f"phi_scatter_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                            mime="image/png"
                        )
                        plt.close()

                    # Data & Export tab (tab6)
                    with tab6:
                        st.dataframe(df_with_coord, width='stretch')

                        st.markdown("---")
                        st.markdown("### 📥 Export Options")

                        # Create two columns for download buttons
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            # Per-skyrmion data CSV
                            df_export = df_with_coord.copy()

                            # Add converted columns
                            df_export[f'X_{unit_display}'] = df_export['X'] * scale_factor
                            df_export[f'Y_{unit_display}'] = df_export['Y'] * scale_factor
                            df_export[f'Area_{unit_display}²'] = df_export['Area'] * (scale_factor ** 2)
                            df_export[f'Diameter_{unit_display}'] = 2 * np.sqrt(df_export['Area'] / np.pi) * scale_factor

                            csv_skyrmions = df_export.to_csv(index=False)
                            st.download_button(
                                label="📊 Download Per-Skyrmion Data",
                                data=csv_skyrmions,
                                file_name=f"analyzed_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
                                mime="text/csv",
                                help="CSV with one row per skyrmion (X, Y, Area, Coordination)"
                            )

                        with col2:
                            # Summary statistics CSV
                            df_summary = create_summary_dataframe(
                                uploaded_file.name,
                                metadata,
                                stats,
                                coord_stats,
                                modality,
                                scale_factor,
                                unit_display,
                                bond_order_stats
                            )
                            csv_summary = df_summary.to_csv(index=False)
                            st.download_button(
                                label="📈 Download Summary Statistics",
                                data=csv_summary,
                                file_name=f"summary_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
                                mime="text/csv",
                                help="Single-row CSV with all aggregate metrics (bimodality, coordination, etc.)"
                            )

                        with col3:
                            # ZIP download with both CSV files
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                # Add per-skyrmion CSV
                                zip_file.writestr(
                                    f"analyzed_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
                                    csv_skyrmions
                                )
                                # Add summary CSV
                                zip_file.writestr(
                                    f"summary_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
                                    csv_summary
                                )

                            st.download_button(
                                label="📦 Download Both Data CSVs",
                                data=zip_buffer.getvalue(),
                                file_name=f"data_package_{uploaded_file.name.rsplit('.', 1)[0]}.zip",
                                mime="application/zip",
                                help="ZIP file containing both per-skyrmion data and summary statistics"
                            )

                        # Figure Downloads Section
                        st.markdown("---")
                        st.markdown("### 🖼️ Download Figures")

                        # Create ZIP with all figures
                        figures_zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(figures_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for fig_name, fig in all_figures.items():
                                fig_bytes = fig_to_bytes(fig)
                                zip_file.writestr(
                                    f"{fig_name}_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                                    fig_bytes
                                )

                        # Single button to download all figures
                        st.download_button(
                            label="📦 Download All Figures (ZIP)",
                            data=figures_zip_buffer.getvalue(),
                            file_name=f"all_figures_{uploaded_file.name.rsplit('.', 1)[0]}.zip",
                            mime="application/zip",
                            help=f"ZIP file containing all {len(all_figures)} visualization figures",
                            type="primary"
                        )

                        # Ultimate Download Everything Section
                        st.markdown("---")
                        st.markdown("### 🎁 Download EVERYTHING")

                        # Create master ZIP with all data and figures
                        everything_zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(everything_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            # Add per-skyrmion CSV
                            zip_file.writestr(
                                f"data/analyzed_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
                                csv_skyrmions
                            )
                            # Add summary CSV
                            zip_file.writestr(
                                f"data/summary_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
                                csv_summary
                            )
                            # Add all figures
                            for fig_name, fig in all_figures.items():
                                fig_bytes = fig_to_bytes(fig)
                                zip_file.writestr(
                                    f"figures/{fig_name}_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                                    fig_bytes
                                )

                        st.download_button(
                            label="🎁 Download EVERYTHING (Data + Figures)",
                            data=everything_zip_buffer.getvalue(),
                            file_name=f"complete_analysis_{uploaded_file.name.rsplit('.', 1)[0]}.zip",
                            mime="application/zip",
                            help=f"Master ZIP containing: 2 CSV files (per-skyrmion data + summary stats) + {len(all_figures)} PNG figures",
                            type="primary"
                        )

                except Exception as e:
                    st.error(f"❌ Error processing file: {e}")
                    st.exception(e)

    # ============================================================
    # BATCH MODE
    # ============================================================
    elif mode == "📦 Batch Analysis":
        st.header("Batch Analysis")
        st.markdown("Upload multiple skyrmion data files for batch processing.")
        
        # Initialize session state
        if 'batch_results' not in st.session_state:
            st.session_state.batch_results = None
        if 'uploaded_file_names' not in st.session_state:
            st.session_state.uploaded_file_names = []
        
        uploaded_files = st.file_uploader(
            "Choose files", 
            type=['xlsx', 'csv'],
            accept_multiple_files=True,
            help="Upload multiple .xlsx or .csv files"
        )
        
        if uploaded_files:
            # Check if files have changed
            current_file_names = [f.name for f in uploaded_files]
            files_changed = current_file_names != st.session_state.uploaded_file_names
            
            st.info(f"📁 {len(uploaded_files)} files uploaded")
            
            # Only show button if files changed or no results yet
            if files_changed or st.session_state.batch_results is None:
                if st.button("🚀 Start Batch Processing", type="primary"):
                    st.session_state.uploaded_file_names = current_file_names
                    results_list = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")

                        try:
                            df, metadata, stats, coord_stats, bond_order_stats, df_with_coord, vor, neighbor_dict = process_uploaded_file(
                                uploaded_file, neighbor_distance_factor
                            )

                            # Combine results
                            result = {
                                **metadata,
                                **stats,
                                **{f'coord_{k}': v for k, v in coord_stats.items()
                                   if k != 'coordination_distribution'},
                                **{f'bond_{k}': v for k, v in bond_order_stats.items()
                                   if k not in ['local_phi', 'local_phi_complex', 'global_phi']}  # Exclude arrays and complex numbers
                            }
                            results_list.append(result)
                            
                        except Exception as e:
                            st.warning(f"⚠️ Error processing {uploaded_file.name}: {e}")
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    status_text.text("✅ Processing complete!")
                    
                    # Check if any files were processed successfully
                    if not results_list:
                        st.error("❌ No files were successfully processed. Please check your file formats.")
                        return
                    
                    # Create results DataFrame and store in session state
                    results_df = pd.DataFrame(results_list)
                    results_df = results_df.sort_values(['field', 'temperature'], ascending=[True, False])
                    st.session_state.batch_results = results_df
        
        # Display results if they exist in session state
        if st.session_state.batch_results is not None:
            results_df = st.session_state.batch_results
            
            st.success(f"✅ Successfully processed {len(results_df)} files")
            
            # Summary statistics
            st.markdown("---")
            st.subheader("📊 Batch Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Files Processed", len(results_df))
            with col2:
                st.metric("Field Strengths", len(results_df['field'].unique()))
            with col3:
                st.metric("Temp. Range", f"{results_df['temperature'].min()}-{results_df['temperature'].max()}K")
            with col4:
                st.metric("Total Skyrmions", int(results_df['n_skyrmions'].sum()))
            
            # Display results table
            st.markdown("---")
            st.subheader("📋 Results Table")
            st.dataframe(
                results_df[['filename', 'field', 'temperature', 'n_skyrmions', 
                           'mean_area', 'number_density', 'coord_mean_coordination', 
                           'coord_packing_efficiency']],
                width='stretch'
            )
            
            # Download button with converted columns
            df_batch_export = results_df.copy()

            # Add scale factor information
            df_batch_export['scale_factor'] = scale_factor
            df_batch_export['unit'] = unit_display

            # Add converted diameter column
            df_batch_export[f'mean_diameter_{unit_display}'] = 2 * np.sqrt(df_batch_export['mean_area'] / np.pi) * scale_factor

            csv = df_batch_export.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="batch_analysis_results.csv",
                mime="text/csv"
            )
            
            # Visualizations
            st.markdown("---")
            st.subheader("📈 Visualizations")
            
            # Tabs for different visualization types
            tab1, tab2 = st.tabs(["📊 Trend Plots", "🔥 Heatmaps"])
            
            # TAB 1: Trend Plots
            with tab1:
                metric_options = {
                    "Skyrmion Count": "n_skyrmions",
                    "Mean Area": "mean_area",
                    "Number Density": "number_density",
                    "Packing Efficiency": "coord_packing_efficiency",
                    "Mean Coordination": "coord_mean_coordination",
                    "Area Coverage": "area_coverage"
                }
                
                selected_metric = st.selectbox("Select metric to plot:", list(metric_options.keys()))
                metric_col = metric_options[selected_metric]
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                fields = sorted(results_df['field'].unique())
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                markers = ['o', 's', '^']
                
                for i, field in enumerate(fields):
                    field_data = results_df[results_df['field'] == field].sort_values('temperature')
                    ax.plot(field_data['temperature'], field_data[metric_col], 
                           marker=markers[i], linestyle='-', linewidth=2, markersize=8,
                           color=colors[i], label=f'{field} Oe', alpha=0.8)
                
                ax.set_xlabel('Temperature (K)', fontsize=12, fontweight='bold')
                ax.set_ylabel(selected_metric, fontsize=12, fontweight='bold')
                ax.set_title(f'{selected_metric} vs Temperature', fontsize=14, fontweight='bold')
                ax.legend(title='Applied Field')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
            
            # TAB 2: Heatmaps
            with tab2:
                heatmap_options = {
                    "Packing Efficiency": "coord_packing_efficiency",
                    "Number Density": "number_density",
                    "Skyrmion Count": "n_skyrmions",
                    "Mean Area": "mean_area",
                    "Mean Coordination": "coord_mean_coordination"
                }
                
                selected_heatmap = st.selectbox("Select metric for heatmap:", list(heatmap_options.keys()))
                heatmap_col = heatmap_options[selected_heatmap]
                
                # Create heatmap
                pivot_data = results_df.pivot(index='temperature', columns='field', values=heatmap_col)
                pivot_data = pivot_data.sort_index(ascending=False)
                
                fig, ax = plt.subplots(figsize=(8, 10))
                
                im = ax.imshow(pivot_data.values, aspect='auto', cmap='viridis')
                
                # Set ticks
                ax.set_xticks(range(len(pivot_data.columns)))
                ax.set_yticks(range(len(pivot_data.index)))
                ax.set_xticklabels([f'{int(f)}' for f in pivot_data.columns])
                ax.set_yticklabels([f'{int(t)}' for t in pivot_data.index])
                
                ax.set_xlabel('Applied Field (Oe)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Temperature (K)', fontsize=12, fontweight='bold')
                ax.set_title(f'{selected_heatmap}: Temperature vs Field', fontsize=14, fontweight='bold')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(selected_heatmap, rotation=270, labelpad=20)
                
                # Add values as text
                for i in range(len(pivot_data.index)):
                    for j in range(len(pivot_data.columns)):
                        value = pivot_data.values[i, j]
                        # Format based on magnitude
                        if value < 0.01:
                            text_str = f'{value:.6f}'
                        elif value < 1:
                            text_str = f'{value:.3f}'
                        else:
                            text_str = f'{value:.1f}'
                        
                        text = ax.text(j, i, text_str,
                                      ha="center", va="center", color="white", fontsize=9,
                                      weight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    # ============================================================
    # ML DETECTION MODE
    # ============================================================
    else:  # ML Detection
        st.header("🤖 ML-Powered Skyrmion Detection")
        st.markdown("""
        Upload LTEM images for automated skyrmion detection, segmentation, and property extraction
        using deep learning (StarDist) and topological classification.
        """)

        # Model information display
        st.markdown("---")
        st.subheader("🔬 Active Models")
        st.metric("StarDist Segmentation", "skyrmion_detector_20251124_224848")
        st.caption("F1 Score: 0.925 (49 training images, 5 field strengths)")

        # Detection mode selector
        detection_mode = st.radio(
            "Detection Mode:",
            ["📄 Single Image", "📁 Temperature Series (Multiple Images)"],
            horizontal=True
        )

        # Optional pixel scale input
        st.markdown("---")
        st.subheader("⚙️ Detection Settings")

        col1, col2 = st.columns(2)
        with col1:
            use_pixel_scale = st.checkbox("Use physical pixel scale", value=False)
            if use_pixel_scale:
                pixel_scale = st.number_input(
                    "Pixel scale (nm/pixel)",
                    min_value=0.01,
                    value=scale_factor,
                    step=0.1,
                    format="%.3f"
                )
            else:
                pixel_scale = None

        with col2:
            show_classification_viz = st.checkbox("Show classification overlay", value=True)
            st.caption("Blue = Skyrmions (Q=±1), Red = Skyrmioniums (Q=0)")

        # Classification method selector
        st.subheader("🎯 Classification Method")

        # Check if ML classifier is available
        if not CLASSIFIER_AVAILABLE:
            st.warning("⚠️ ML classifier not available (polar_fourier_classifier.py not found). Using Algorithmic mode.")
            classifier_method = "Algorithmic (Peak-based)"
        else:
            classifier_method = st.radio(
                "Select classifier:",
                ["ML (Polar Fourier v3)", "Algorithmic (Peak-based)"],
                horizontal=True,
                help="ML uses trained RandomForest on azimuthal Fourier features (F1=0.903). Algorithmic uses diameter profile peak detection."
            )

        # Show different parameters based on classifier selection
        if classifier_method == "ML (Polar Fourier v3)" and CLASSIFIER_AVAILABLE:
            st.info("🤖 **ML Classifier**: Uses 30 physics-informed features from azimuthal intensity profiles. Trained RandomForest achieves F1=0.903 on 3,088 labeled samples.")
            st.markdown("""
            **Key discriminative features:**
            - `phase_diff_r2_r8`: Phase difference between inner/outer radii (d'=2.46)
            - `amp_ratio_r3_r7`: Inner/outer amplitude ratio (d'=2.41)
            - `amp_m1_r3`: Inner asymmetry amplitude (d'=2.39)
            """)
            ml_confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.5,
                max_value=0.95,
                value=0.5,
                step=0.05,
                help="Predictions below this confidence will be flagged as uncertain"
            )
            # Set algorithmic params to defaults (won't be used)
            peak_prominence = 0.08
            center_window_fraction = 0.20
            skyrmionium_threshold = 0.11
            top_n_angles = 3
            merge_threshold = 0.15
            width_bonus_enabled = True
            angle_selection_mode = 'top_n'
            selected_angles = [0, 45, 90, 135]
            use_nonlinear_window = False
            window_scale_a = 0.15
            window_scale_b = 2.0
        else:
            ml_confidence_threshold = 0.5  # Default, won't be used
            # Algorithmic parameters
            st.markdown("---")
            st.markdown("### Algorithmic Parameters")
            st.markdown("Adjust parameters to tune skyrmion vs skyrmionium distinction:")
            st.info("⚙️ **Note**: Parameter changes take effect when you upload a new image or click 'Run Classification Diagnostics'")

            col1, col2, col3 = st.columns(3)
            with col1:
                peak_prominence = st.slider(
                    "Peak Prominence",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.08,
                    step=0.01,
                    help="Minimum peak height to detect (lower = more sensitive)"
                )
            with col2:
                center_window_fraction = st.slider(
                    "Center Window Size (NEW)",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.20,
                    step=0.05,
                    help="Radius fraction defining 'center region' for skyrmionium peaks (scales with skyrmion size)"
                )
            with col3:
                skyrmionium_threshold = st.slider(
                    "Skyrmionium Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.11,
                    step=0.01,
                    help="Classification cutoff (higher = fewer skyrmioniums)"
                )

            # Additional advanced parameters (row 2)
            col4, col5, col6 = st.columns(3)
            with col4:
                top_n_angles = st.slider(
                    "Top-N Angles (NEW)",
                    min_value=1,
                    max_value=4,
                    value=3,
                    step=1,
                    help="Use only the best N diameter angles (3 recommended for balanced detection)"
                )
            with col5:
                merge_threshold = st.slider(
                    "Peak Merge Threshold (NEW)",
                    min_value=0.05,
                    max_value=0.3,
                    value=0.15,
                    step=0.05,
                    help="Merge peaks closer than this fraction of radius (prevents split-peak inflation)"
                )
            with col6:
                width_bonus_enabled = st.checkbox(
                    "Width Bonus (NEW)",
                    value=True,
                    help="Give bonus score to wide peaks (compensates for size-dependent peak widths)"
                )

            # Advanced angle selection controls
            with st.expander("🎯 Advanced: Angle Selection Options", expanded=False):
                st.markdown("""
                **Purpose**: Fine-tune which diameter angles contribute to classification.

                - **Top-N** (default): Use the N highest scoring angles
                - **Manual**: Select specific angles you want to use
                - **Middle Two**: Use only the 2nd and 3rd highest scoring angles (excludes outliers)
                """)

                angle_selection_mode = st.radio(
                    "Selection Mode",
                    options=['top_n', 'manual', 'middle_two'],
                    format_func=lambda x: {
                        'top_n': 'Top-N (use best N angles)',
                        'manual': 'Manual (select specific angles)',
                        'middle_two': 'Middle Two (exclude highest and lowest)'
                    }[x],
                    index=0,
                    help="Method for selecting which diameter angles to use"
                )

                if angle_selection_mode == 'manual':
                    st.markdown("**Select which angles to use:**")
                    col_0, col_45, col_90, col_135 = st.columns(4)
                    with col_0:
                        use_0 = st.checkbox("0° (horizontal)", value=True, key="angle_0")
                    with col_45:
                        use_45 = st.checkbox("45°", value=True, key="angle_45")
                    with col_90:
                        use_90 = st.checkbox("90° (vertical)", value=True, key="angle_90")
                    with col_135:
                        use_135 = st.checkbox("135°", value=True, key="angle_135")

                    selected_angles = []
                    if use_0:
                        selected_angles.append(0)
                    if use_45:
                        selected_angles.append(1)
                    if use_90:
                        selected_angles.append(2)
                    if use_135:
                        selected_angles.append(3)

                    if len(selected_angles) == 0:
                        st.warning("⚠️ You must select at least one angle!")
                        selected_angles = [0, 1, 2, 3]  # Fallback

                elif angle_selection_mode == 'middle_two':
                    st.info("ℹ️ Using only the 2nd and 3rd highest scoring angles (excludes both highest and lowest outliers)")
                    selected_angles = None

                else:  # top_n mode
                    selected_angles = None
                    st.info(f"ℹ️ Using top-{top_n_angles} mode (configured above)")

            # Non-linear window scaling controls (inside algorithmic else block)
            with st.expander("🔬 Advanced: Non-Linear Window Scaling", expanded=False):
                st.markdown("""
                **Purpose**: Eliminate size bias by using logarithmic window scaling.

                - **Small skyrmions** (<30px radius): Get proportionally smaller center windows
                - **Large skyrmions** (>60px radius): Get proportionally larger center windows
                - **Formula**: `window_fraction = max(0.08, a + b × log(radius))`
                """)

                use_nonlinear_window = st.checkbox(
                    "Enable Logarithmic Window Scaling",
                    value=True,
                    help="Use logarithmic scaling instead of fixed fraction. Recommended for mixed-size datasets."
                )

                if use_nonlinear_window:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        window_scale_a = st.slider(
                            "Intercept (a)",
                            min_value=-0.5,
                            max_value=0.0,
                            value=-0.27,
                            step=0.01,
                            help="Logarithmic scaling intercept (default: -0.27)"
                        )
                    with col_b:
                        window_scale_b = st.slider(
                            "Slope (b)",
                            min_value=0.05,
                            max_value=0.25,
                            value=0.13,
                            step=0.01,
                            help="Logarithmic scaling slope (default: 0.13)"
                        )

                    # Preview plot
                    st.markdown("**Preview: Window Size vs. Skyrmion Radius**")

                    radii = np.linspace(10, 100, 100)
                    window_fractions = np.maximum(0.08, window_scale_a + window_scale_b * np.log(radii))
                    window_sizes = window_fractions * radii

                    fig_preview, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

                    # Plot 1: Window fraction vs radius
                    ax1.plot(radii, window_fractions, 'b-', linewidth=2)
                    ax1.axhline(y=center_window_fraction, color='r', linestyle='--', label=f'Linear ({center_window_fraction:.2f}×R)', alpha=0.5)
                    ax1.set_xlabel('Skyrmion Radius (pixels)')
                    ax1.set_ylabel('Window Fraction')
                    ax1.set_title('Center Window Fraction')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)

                    # Plot 2: Window size vs radius
                    ax2.plot(radii, window_sizes, 'b-', linewidth=2, label='Logarithmic')
                    ax2.plot(radii, center_window_fraction * radii, 'r--', alpha=0.5, label='Linear')
                    ax2.set_xlabel('Skyrmion Radius (pixels)')
                    ax2.set_ylabel('Window Size (pixels)')
                    ax2.set_title('Absolute Window Size')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig_preview)
                    plt.close(fig_preview)

                    # Example calculations
                    st.markdown("**Example Window Sizes:**")
                    example_radii = [15, 20, 30, 50, 80]
                    examples = []
                    for r in example_radii:
                        frac = max(0.08, window_scale_a + window_scale_b * np.log(r))
                        size = frac * r
                        examples.append(f"**Radius {r}px**: {frac:.3f}×R = {size:.1f}px")
                    st.markdown(" | ".join(examples))
                else:
                    # Use linear scaling (fixed fraction)
                    window_scale_a = -0.27  # Not used
                    window_scale_b = 0.13   # Not used

        # Deprecated parameter (kept for backward compatibility but hidden)
        center_falloff_sigma = 0.2  # Not used anymore, replaced by center_window_fraction

        st.markdown("---")

        # ============================================================
        # SINGLE IMAGE MODE
        # ============================================================
        if detection_mode == "📄 Single Image":
            st.subheader("Upload LTEM Image")

            uploaded_image = st.file_uploader(
                "Choose an LTEM image",
                type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
                help="Upload a grayscale LTEM image showing skyrmions"
            )

            if uploaded_image is not None:
                with st.spinner('🔄 Loading ML model and processing image...'):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_image.name) as tmp_file:
                            tmp_file.write(uploaded_image.getvalue())
                            tmp_path = tmp_file.name

                        # Import ML pipeline
                        import sys
                        sys.path.insert(0, str(Path(__file__).parent / 'ml_pipeline_public'))
                        from ml_pipeline_public.extract_properties_public import SkyrmionPropertyExtractor

                        # Initialize extractor (loads model)
                        # Note: Parameters are passed directly, no caching for classifier parameters
                        @st.cache_resource
                        def load_model_only():
                            # Only cache the StarDist model loading (expensive)
                            from ml_pipeline_public.extract_properties_public import SkyrmionPropertyExtractor
                            return SkyrmionPropertyExtractor(
                                peak_prominence=0.15,  # Default, will be overridden
                                center_falloff_sigma=0.2,
                                skyrmionium_threshold=0.3
                            )

                        # Create extractor with ALL current slider values
                        extractor = SkyrmionPropertyExtractor(
                            peak_prominence=peak_prominence,
                            center_falloff_sigma=center_falloff_sigma,
                            skyrmionium_threshold=skyrmionium_threshold,
                            center_window_fraction=center_window_fraction,
                            top_n_angles=top_n_angles,
                            merge_threshold=merge_threshold,
                            width_bonus_enabled=width_bonus_enabled,
                            use_nonlinear_window=use_nonlinear_window,
                            window_scale_a=window_scale_a,
                            window_scale_b=window_scale_b,
                            angle_selection_mode=angle_selection_mode,
                            selected_angles=selected_angles
                        )

                        # Process image (StarDist segmentation + Algorithmic classification)
                        properties, labels = extractor.process_image(
                            tmp_path,
                            pixel_scale=pixel_scale,
                            save_visualization=False,
                            output_dir=None
                        )

                        if len(properties) == 0:
                            # Clean up temp file
                            Path(tmp_path).unlink()
                            st.warning("⚠️ No skyrmions detected in this image.")
                        else:
                            st.success(f"✅ Detected {len(properties)} skyrmions!")

                            # If ML classifier selected, re-classify using Polar Fourier v3
                            if classifier_method == "ML (Polar Fourier v3)" and CLASSIFIER_AVAILABLE:
                                st.info("🤖 Running Polar Fourier v3 classification...")
                                try:
                                    # Load image for ML classification
                                    image_for_ml = skio.imread(tmp_path)
                                    if len(image_for_ml.shape) == 3:
                                        image_for_ml = image_for_ml.mean(axis=2)
                                    image_for_ml = image_for_ml.astype(np.float32)
                                    if image_for_ml.max() > 1:
                                        image_for_ml = image_for_ml / 255.0

                                    # Initialize and load v3 classifier
                                    clf = PolarFourierClassifier(version=3)
                                    model_path = Path(__file__).parent / 'ml_pipeline_public' / 'polar_fourier_v3_model.pkl'
                                    if model_path.exists():
                                        clf.load_model(str(model_path))

                                    # Classify each detection
                                    ml_classifications = []
                                    ml_confidences = []
                                    for _, row in properties.iterrows():
                                        cx, cy = row['centroid_x'], row['centroid_y']
                                        radius = row.get('diameter_px', 50) / 2

                                        pred, conf, feats = clf.classify(image_for_ml, cy, cx, radius)
                                        ml_classifications.append('skyrmionium' if pred == 0 else 'skyrmion')
                                        ml_confidences.append(conf)

                                    # Update properties with ML classifications
                                    properties['classification'] = ml_classifications
                                    properties['ml_confidence'] = ml_confidences
                                    # Also update topological_charge to match ML classification
                                    properties['topological_charge'] = [0 if c == 'skyrmionium' else 1 for c in ml_classifications]

                                    # Filter by confidence threshold if set
                                    if ml_confidence_threshold > 0.5:
                                        uncertain_mask = properties['ml_confidence'] < ml_confidence_threshold
                                        if uncertain_mask.any():
                                            st.warning(f"⚠️ {uncertain_mask.sum()} detections have confidence < {ml_confidence_threshold:.0%}")

                                except Exception as e:
                                    st.error(f"ML classification error: {e}")
                                    st.info("Falling back to Algorithmic classification")

                            # Summary metrics
                            n_skyrmions = (properties['classification'] == 'skyrmion').sum()
                            n_skyrmioniums = (properties['classification'] == 'skyrmionium').sum()

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Detected", len(properties))
                            with col2:
                                st.metric("Skyrmions (Q=±1)", n_skyrmions)
                            with col3:
                                st.metric("Skyrmioniums (Q=0)", n_skyrmioniums)
                            with col4:
                                q_ratio = n_skyrmioniums / len(properties) if len(properties) > 0 else 0
                                st.metric("Q=0 Fraction", f"{q_ratio:.1%}")

                            # Visualizations
                            st.markdown("---")
                            st.subheader("📊 Results")

                            tab1, tab2, tab3 = st.tabs(["🎯 Detection Overlay", "📊 Properties", "📈 Distributions"])

                            with tab1:
                                # Show classification overlay
                                if show_classification_viz:
                                    # Load original image for visualization
                                    image_normalized, image_raw = extractor.load_image(tmp_path)

                                    # Create dual output: Segmented + Classified
                                    from matplotlib.patches import Circle as MPLCircle

                                    st.markdown("### Detection Results")
                                    col_seg, col_class = st.columns(2)

                                    # Get unit suffix for radius
                                    unit_suffix = 'nm' if pixel_scale else 'px'

                                    # === LEFT: Segmented Image (all same color) ===
                                    with col_seg:
                                        st.markdown("**Segmented Image** (StarDist detections)")
                                        fig_seg, ax_seg = plt.subplots(1, 1, figsize=(10, 10))
                                        ax_seg.imshow(image_raw, cmap='gray')

                                        for _, row in properties.iterrows():
                                            unit_col = f'diameter_{unit_suffix}'
                                            diameter = row.get(unit_col, row.get('diameter_px', 0))
                                            radius = diameter / 2

                                            # All circles same color (green/teal)
                                            circle = MPLCircle(
                                                (row['centroid_x'], row['centroid_y']),
                                                radius,
                                                fill=False,
                                                edgecolor='#00CED1',  # Dark Turquoise
                                                linewidth=2,
                                                alpha=0.9
                                            )
                                            ax_seg.add_patch(circle)

                                        ax_seg.set_title(f"StarDist Segmentation: {len(properties)} detections", fontsize=12)
                                        ax_seg.axis('off')
                                        plt.tight_layout()
                                        st.pyplot(fig_seg)
                                        plt.close(fig_seg)

                                    # === RIGHT: Classified Image (color-coded by Q) ===
                                    with col_class:
                                        st.markdown("**Classified Image** (Q=0 Red, Q=1 Blue)")
                                        fig_class, ax_class = plt.subplots(1, 1, figsize=(10, 10))
                                        ax_class.imshow(image_raw, cmap='gray')

                                        n_skyrmions = 0
                                        n_skyrmioniums = 0

                                        for _, row in properties.iterrows():
                                            # Color based on classification
                                            if row['classification'] == 'skyrmion':
                                                color = '#3366FF'  # Blue
                                                n_skyrmions += 1
                                            else:
                                                color = '#FF3333'  # Red
                                                n_skyrmioniums += 1

                                            unit_col = f'diameter_{unit_suffix}'
                                            diameter = row.get(unit_col, row.get('diameter_px', 0))
                                            radius = diameter / 2

                                            # Draw circle
                                            circle = MPLCircle(
                                                (row['centroid_x'], row['centroid_y']),
                                                radius,
                                                fill=False,
                                                edgecolor=color,
                                                linewidth=2,
                                                alpha=0.9
                                            )
                                            ax_class.add_patch(circle)

                                            # Add ID label
                                            ax_class.text(
                                                row['centroid_x'],
                                                row['centroid_y'],
                                                str(row['label_id']),
                                                color='white',
                                                fontsize=8,
                                                fontweight='bold',
                                                ha='center',
                                                va='center',
                                                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6, edgecolor='none')
                                            )

                                        classifier_label = "ML (Polar Fourier v3)" if classifier_method == "ML (Polar Fourier v3)" else "Algorithmic"
                                        ax_class.set_title(
                                            f"{classifier_label}: {n_skyrmions} Skyrmions (blue), {n_skyrmioniums} Skyrmioniums (red)",
                                            fontsize=12
                                        )
                                        ax_class.axis('off')
                                        plt.tight_layout()
                                        st.pyplot(fig_class)
                                        plt.close(fig_class)

                                    # Legend
                                    st.markdown("""
                                    <div style='text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 5px;'>
                                    <span style='color: #00CED1; font-weight: bold;'>●</span> Segmented (all detections) &nbsp;&nbsp;|&nbsp;&nbsp;
                                    <span style='color: #3366FF; font-weight: bold;'>●</span> Skyrmion (Q=±1) &nbsp;&nbsp;|&nbsp;&nbsp;
                                    <span style='color: #FF3333; font-weight: bold;'>●</span> Skyrmionium (Q=0)
                                    </div>
                                    """, unsafe_allow_html=True)

                                    # Add diagnostic button with pagination
                                    st.markdown("---")

                                    # Initialize session state for diagnostics
                                    if 'diag_page' not in st.session_state:
                                        st.session_state.diag_page = 1
                                    if 'diag_total_pages' not in st.session_state:
                                        st.session_state.diag_total_pages = 1

                                    if st.button("🔬 Run Classification Diagnostics"):
                                        st.session_state.diag_page = 1  # Reset to page 1
                                        with st.spinner('Analyzing radial profiles...'):
                                            import sys
                                            sys.path.insert(0, str(Path(__file__).parent / 'ml_pipeline_public'))
                                            from diagnose_classifier_public import diagnose_classification

                                            # Run diagnostics with ALL current UI parameters
                                            try:
                                                # Calculate total pages first (50 skyrmions per page)
                                                total_pages = diagnose_classification(
                                                    tmp_path,
                                                    labels_path=None,
                                                    n_examples=None,  # Show ALL skyrmions
                                                    peak_prominence=peak_prominence,
                                                    center_falloff_sigma=center_falloff_sigma,
                                                    skyrmionium_threshold=skyrmionium_threshold,
                                                    center_window_fraction=center_window_fraction,
                                                    top_n_angles=top_n_angles,
                                                    merge_threshold=merge_threshold,
                                                    width_bonus_enabled=width_bonus_enabled,
                                                    use_nonlinear_window=use_nonlinear_window,
                                                    window_scale_a=window_scale_a,
                                                    window_scale_b=window_scale_b,
                                                    angle_selection_mode=angle_selection_mode,
                                                    selected_angles=selected_angles,
                                                    page=st.session_state.diag_page,
                                                    page_size=50
                                                )
                                                st.session_state.diag_total_pages = total_pages
                                                st.success(f"✓ Generated diagnostics (Page {st.session_state.diag_page} of {total_pages})")
                                            except Exception as e:
                                                st.error(f"Error running diagnostics: {e}")
                                                st.exception(e)

                                    # Check if current page needs to be generated FIRST (before displaying)
                                    current_page_path = Path('ml_pipeline_public/results') / f'classifier_diagnostics_{Path(tmp_path).stem}_page{st.session_state.diag_page}.png'
                                    if not current_page_path.exists():
                                        with st.spinner(f'Loading page {st.session_state.diag_page}...'):
                                            import sys
                                            sys.path.insert(0, str(Path(__file__).parent / 'ml_pipeline_public'))
                                            from diagnose_classifier_public import diagnose_classification

                                            try:
                                                diagnose_classification(
                                                    tmp_path,
                                                    labels_path=None,
                                                    n_examples=None,
                                                    peak_prominence=peak_prominence,
                                                    center_falloff_sigma=center_falloff_sigma,
                                                    skyrmionium_threshold=skyrmionium_threshold,
                                                    center_window_fraction=center_window_fraction,
                                                    top_n_angles=top_n_angles,
                                                    merge_threshold=merge_threshold,
                                                    width_bonus_enabled=width_bonus_enabled,
                                                    use_nonlinear_window=use_nonlinear_window,
                                                    window_scale_a=window_scale_a,
                                                    window_scale_b=window_scale_b,
                                                    angle_selection_mode=angle_selection_mode,
                                                    selected_angles=selected_angles,
                                                    page=st.session_state.diag_page,
                                                    page_size=50
                                                )
                                            except Exception as e:
                                                st.error(f"Error generating page {st.session_state.diag_page}: {e}")

                                    # Display diagnostic image if it exists
                                    if current_page_path.exists():
                                        # Pagination controls
                                        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
                                        with col1:
                                            if st.button("⏮️ First", disabled=(st.session_state.diag_page <= 1)):
                                                st.session_state.diag_page = 1
                                                st.rerun()
                                        with col2:
                                            if st.button("◀️ Previous", disabled=(st.session_state.diag_page <= 1)):
                                                st.session_state.diag_page -= 1
                                                st.rerun()
                                        with col3:
                                            st.markdown(f"<div style='text-align: center; padding: 5px;'><b>Page {st.session_state.diag_page} of {st.session_state.diag_total_pages}</b><br/><small>Showing skyrmions {(st.session_state.diag_page-1)*50 + 1}-{min(st.session_state.diag_page*50, len(properties))}</small></div>", unsafe_allow_html=True)
                                        with col4:
                                            if st.button("Next ▶️", disabled=(st.session_state.diag_page >= st.session_state.diag_total_pages)):
                                                st.session_state.diag_page += 1
                                                st.rerun()
                                        with col5:
                                            if st.button("Last ⏭️", disabled=(st.session_state.diag_page >= st.session_state.diag_total_pages)):
                                                st.session_state.diag_page = st.session_state.diag_total_pages
                                                st.rerun()

                                        # Display the image
                                        st.image(str(current_page_path), caption=f"Topological Classifier Diagnostics - Page {st.session_state.diag_page}", use_column_width=True)
                                        st.info("""
                                        **How to interpret diagnostics:**
                                        - **Left**: Skyrmion patch with detected boundary
                                        - **Middle**: Diameter profiles at 4 angles (0°, 45°, 90°, 135°)
                                        - **Right**: Classification scores per angle and final decision

                                        If classification seems random, LTEM contrast may be too low for reliable distinction.
                                        Consider supervised training with manually labeled examples.
                                        """)

                            with tab2:
                                # Properties table
                                st.markdown("### Detected Skyrmion Properties")

                                # Select relevant columns for display
                                unit_suffix = 'nm' if pixel_scale else 'px'
                                display_cols = [
                                    'label_id', 'classification', 'topological_charge', 'confidence',
                                    'centroid_x', 'centroid_y',
                                    f'area_{unit_suffix}', f'diameter_{unit_suffix}',
                                    'circularity', 'mean_intensity'
                                ]

                                # Filter columns that exist
                                available_cols = [col for col in display_cols if col in properties.columns]
                                st.dataframe(properties[available_cols], height=400)

                                # Download button - add compatible columns for Single File Analysis
                                export_df = properties.copy()
                                # Add Area, X, Y columns for compatibility with Single File Analysis
                                export_df['Area'] = export_df['area_px']
                                export_df['X'] = export_df['centroid_x']
                                export_df['Y'] = export_df['centroid_y']
                                csv = export_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Properties as CSV",
                                    data=csv,
                                    file_name=f"ml_detected_{uploaded_image.name.rsplit('.', 1)[0]}.csv",
                                    mime="text/csv"
                                )

                            with tab3:
                                # Distribution plots
                                st.markdown("### Size Distribution by Topological Charge")

                                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                                # Size distribution
                                unit_col = f'diameter_{unit_suffix}'
                                skyrmion_sizes = properties[properties['classification'] == 'skyrmion'][unit_col]
                                skyrmionium_sizes = properties[properties['classification'] == 'skyrmionium'][unit_col]

                                axes[0].hist(skyrmion_sizes, bins=15, alpha=0.7, label='Skyrmions (Q=±1)', color='blue', edgecolor='black')
                                axes[0].hist(skyrmionium_sizes, bins=15, alpha=0.7, label='Skyrmioniums (Q=0)', color='red', edgecolor='black')
                                axes[0].set_xlabel(f'Diameter ({unit_suffix})')
                                axes[0].set_ylabel('Count')
                                axes[0].set_title('Size Distribution by Topological Charge')
                                axes[0].legend()
                                axes[0].grid(alpha=0.3)

                                # Confidence distribution
                                axes[1].hist(properties['confidence'], bins=20, alpha=0.7, color='green', edgecolor='black')
                                axes[1].set_xlabel('Classification Confidence')
                                axes[1].set_ylabel('Count')
                                axes[1].set_title('Topological Classification Confidence')
                                axes[1].axvline(properties['confidence'].mean(), color='red', linestyle='--', label='Mean')
                                axes[1].legend()
                                axes[1].grid(alpha=0.3)

                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()

                                # Statistical summary
                                st.markdown("### Statistical Summary")
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("**Skyrmions (Q=±1)**")
                                    if len(skyrmion_sizes) > 0:
                                        st.write(f"Count: {len(skyrmion_sizes)}")
                                        st.write(f"Mean diameter: {skyrmion_sizes.mean():.2f} {unit_suffix}")
                                        st.write(f"Std diameter: {skyrmion_sizes.std():.2f} {unit_suffix}")
                                    else:
                                        st.write("None detected")

                                with col2:
                                    st.markdown("**Skyrmioniums (Q=0)**")
                                    if len(skyrmionium_sizes) > 0:
                                        st.write(f"Count: {len(skyrmionium_sizes)}")
                                        st.write(f"Mean diameter: {skyrmionium_sizes.mean():.2f} {unit_suffix}")
                                        st.write(f"Std diameter: {skyrmionium_sizes.std():.2f} {unit_suffix}")
                                    else:
                                        st.write("None detected")

                            # Clean up temp file
                            Path(tmp_path).unlink()

                    except Exception as e:
                        # Clean up temp file on error
                        if 'tmp_path' in locals():
                            try:
                                Path(tmp_path).unlink()
                            except:
                                pass
                        st.error(f"❌ Error during ML detection: {e}")
                        st.exception(e)

        # ============================================================
        # TEMPERATURE SERIES MODE
        # ============================================================
        elif detection_mode == "📁 Temperature Series (Multiple Images)":
            st.subheader("Upload Temperature Series")
            st.markdown("Upload multiple LTEM images from a temperature series for batch processing and tracking.")

            uploaded_images = st.file_uploader(
                "Choose LTEM images",
                type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
                accept_multiple_files=True,
                help="Upload multiple images (e.g., 110K.jpg, 120K.jpg, etc.)"
            )

            if uploaded_images:
                st.info(f"📁 {len(uploaded_images)} images uploaded")

                if st.button("🚀 Start ML Detection", type="primary"):
                    with st.spinner('🔄 Processing images with ML pipeline...'):
                        try:
                            # Import ML pipeline
                            import sys
                            sys.path.insert(0, str(Path(__file__).parent / 'ml_pipeline_public'))
                            from ml_pipeline_public.extract_properties_public import SkyrmionPropertyExtractor

                            # Initialize extractor
                            @st.cache_resource
                            def load_extractor():
                                return SkyrmionPropertyExtractor()

                            extractor = load_extractor()

                            # Save images temporarily
                            temp_paths = []
                            for uploaded_image in uploaded_images:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_image.name) as tmp_file:
                                    tmp_file.write(uploaded_image.getvalue())
                                    temp_paths.append(tmp_file.name)

                            # Process batch
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            # Initialize Polar Fourier classifier for ML classification
                            use_polar_fourier = CLASSIFIER_AVAILABLE
                            pf_classifier = None
                            if use_polar_fourier:
                                try:
                                    pf_classifier = PolarFourierClassifier(version=3)
                                    model_path = Path(__file__).parent / 'ml_pipeline_public' / 'polar_fourier_v3_model.pkl'
                                    if model_path.exists():
                                        pf_classifier.load_model(str(model_path))
                                        st.info("🤖 Using Polar Fourier v3 classifier for classification")
                                    else:
                                        st.warning("⚠️ Polar Fourier model not found, using algorithmic classifier")
                                        use_polar_fourier = False
                                except Exception as e:
                                    st.warning(f"⚠️ Could not load Polar Fourier classifier: {e}")
                                    use_polar_fourier = False

                            all_properties = []
                            for i, (tmp_path, uploaded_image) in enumerate(zip(temp_paths, uploaded_images)):
                                status_text.text(f"Processing {i+1}/{len(temp_paths)}: {uploaded_image.name}")

                                try:
                                    props, labels = extractor.process_image(
                                        tmp_path,
                                        pixel_scale=pixel_scale,
                                        save_visualization=False,
                                        output_dir=None
                                    )

                                    if len(props) > 0:
                                        # Apply Polar Fourier classification if available
                                        if use_polar_fourier and pf_classifier is not None:
                                            try:
                                                # Load image for ML classification
                                                image_for_ml = skio.imread(tmp_path)
                                                if len(image_for_ml.shape) == 3:
                                                    image_for_ml = image_for_ml.mean(axis=2)
                                                image_for_ml = image_for_ml.astype(np.float32)
                                                if image_for_ml.max() > 1:
                                                    image_for_ml = image_for_ml / 255.0

                                                # Classify each detection
                                                ml_classifications = []
                                                ml_confidences = []
                                                for _, row in props.iterrows():
                                                    cx, cy = row['centroid_x'], row['centroid_y']
                                                    radius = row.get('diameter_px', 50) / 2

                                                    pred, conf, feats = pf_classifier.classify(image_for_ml, cy, cx, radius)
                                                    ml_classifications.append('skyrmionium' if pred == 0 else 'skyrmion')
                                                    ml_confidences.append(conf)

                                                # Update properties with ML classifications
                                                props['classification'] = ml_classifications
                                                props['ml_confidence'] = ml_confidences
                                                props['topological_charge'] = [0 if c == 'skyrmionium' else 1 for c in ml_classifications]

                                            except Exception as ml_err:
                                                # Keep algorithmic classification on ML error
                                                pass

                                        props['image_name'] = uploaded_image.name
                                        props['image_index'] = i
                                        all_properties.append(props)

                                except Exception as e:
                                    st.warning(f"⚠️ Error processing {uploaded_image.name}: {e}")

                                progress_bar.progress((i + 1) / len(temp_paths))

                            # Clean up temp files
                            for tmp_path in temp_paths:
                                Path(tmp_path).unlink()

                            status_text.text("✅ Processing complete!")

                            if not all_properties:
                                st.error("❌ No skyrmions detected in any images")
                            else:
                                # Combine results
                                combined_props = pd.concat(all_properties, ignore_index=True)

                                st.success(f"✅ Detected {len(combined_props)} total skyrmions across {len(uploaded_images)} images")

                                # Summary statistics
                                st.markdown("---")
                                st.subheader("📊 Batch Summary")

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Images Processed", len(uploaded_images))
                                with col2:
                                    st.metric("Total Skyrmions", len(combined_props))
                                with col3:
                                    n_sky = (combined_props['classification'] == 'skyrmion').sum()
                                    st.metric("Skyrmions (Q=±1)", n_sky)
                                with col4:
                                    n_skx = (combined_props['classification'] == 'skyrmionium').sum()
                                    st.metric("Skyrmioniums (Q=0)", n_skx)

                                # Per-image summary
                                st.markdown("---")
                                st.subheader("📋 Per-Image Results")

                                image_summary = combined_props.groupby('image_name').agg({
                                    'label_id': 'count',
                                    'topological_charge': lambda x: (x == 1).sum(),
                                    'confidence': 'mean'
                                }).rename(columns={
                                    'label_id': 'total_detected',
                                    'topological_charge': 'n_skyrmions',
                                    'confidence': 'mean_confidence'
                                })
                                image_summary['n_skyrmioniums'] = image_summary['total_detected'] - image_summary['n_skyrmions']

                                st.dataframe(image_summary, use_container_width=True)

                                # Download combined results
                                st.markdown("---")
                                csv = combined_props.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download All Properties as CSV",
                                    data=csv,
                                    file_name="ml_batch_detection_results.csv",
                                    mime="text/csv"
                                )

                                # Trend visualization
                                st.markdown("---")
                                st.subheader("📈 Detection Trends Across Series")

                                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                                # Count trends
                                axes[0].plot(image_summary.index, image_summary['total_detected'], marker='o', label='Total', linewidth=2)
                                axes[0].plot(image_summary.index, image_summary['n_skyrmions'], marker='s', label='Skyrmions', linewidth=2)
                                axes[0].plot(image_summary.index, image_summary['n_skyrmioniums'], marker='^', label='Skyrmioniums', linewidth=2)
                                axes[0].set_xlabel('Image')
                                axes[0].set_ylabel('Count')
                                axes[0].set_title('Detection Counts Across Series')
                                axes[0].legend()
                                axes[0].grid(alpha=0.3)
                                axes[0].tick_params(axis='x', rotation=45)

                                # Size trends
                                unit_suffix = 'nm' if pixel_scale else 'px'
                                size_col = f'diameter_{unit_suffix}'
                                size_by_image = combined_props.groupby('image_name')[size_col].mean()
                                axes[1].plot(size_by_image.index, size_by_image.values, marker='o', linewidth=2, color='purple')
                                axes[1].set_xlabel('Image')
                                axes[1].set_ylabel(f'Mean Diameter ({unit_suffix})')
                                axes[1].set_title('Mean Skyrmion Size Across Series')
                                axes[1].grid(alpha=0.3)
                                axes[1].tick_params(axis='x', rotation=45)

                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()

                        except Exception as e:
                            st.error(f"❌ Error during batch ML detection: {e}")
                            st.exception(e)


if __name__ == "__main__":
    main()