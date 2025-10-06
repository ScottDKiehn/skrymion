import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import re
from tqdm import tqdm 

# Set your base path here
BASE_PATH = Path("/Users/scottkiehn/MSE/Skyrmion")

def load_skyrmion_file(filepath):
    """
    Load a single skyrmion data file and extract relevant columns.
    Handles both .xlsx and .csv formats, with proper encoding for CSV.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the data file (can be relative to BASE_PATH or absolute)
    
    Returns:
    --------
    df : pandas DataFrame
        DataFrame with columns: Area, X, Y
    """
    filepath = Path(filepath)
    
    # Make path absolute if it's relative
    if not filepath.is_absolute():
        filepath = BASE_PATH / filepath
    
    # Load based on file extension
    if filepath.suffix == '.xlsx':
        df = pd.read_excel(filepath)
    elif filepath.suffix == '.csv':
        # Try different encodings (CSV files may have Chinese encoding)
        encodings_to_try = ['gb18030', 'gbk', 'gb2312', 'utf-8', 'latin1']
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                print(f"  (Using encoding: {encoding})")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not read CSV with any standard encoding")
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")
    
    # Extract only the columns we need
    skyrmion_data = df[['Area', 'X', 'Y']].copy()
    
    print(f"✓ Loaded: {filepath.name}")
    print(f"  Format: {filepath.suffix}")
    print(f"  Number of skyrmions: {len(skyrmion_data)}")
    print(f"  Area range: {skyrmion_data['Area'].min():.0f} - {skyrmion_data['Area'].max():.0f} pixels²")
    print(f"\nFirst few skyrmions:")
    print(skyrmion_data.head())
    print("\n" + "="*60 + "\n")
    
    return skyrmion_data

def parse_filename(filepath):
    """
    Extract metadata from skyrmion data filename.
    
    Expected format: "OL=[FIELD]，[ID]-1，T=[TEMP]K -E.xlsx" or with "（new）" suffix
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the data file
    
    Returns:
    --------
    metadata : dict
        Dictionary with keys: 'field', 'id', 'temperature', 'filename'
    """
    filepath = Path(filepath)
    filename = filepath.name
    
    # Remove file extension and any "（new）" or "(new)" suffix
    base_name = filename.replace('（new）', '').replace('(new)', '').replace('.xlsx', '').replace('.csv', '')
    
    # Split by full-width comma (，)
    parts = base_name.split('，')
    
    try:
        # Extract field strength from "OL=0800" -> 800 (as integer)
        field_str = parts[0].replace('OL=', '').strip()
        field = int(field_str) if field_str != '0' else 0
        
        # Extract ID from "286-1" -> 286
        id_num = int(parts[1].split('-')[0].strip())
        
        # Extract temperature from "T=200K -E" -> 200
        temp_str = parts[2].replace('T=', '').replace('K', '').replace('-E', '').strip()
        temperature = int(temp_str)
        
        metadata = {
            'field': field,
            'id': id_num,
            'temperature': temperature,
            'filename': filename
        }
        
        return metadata
        
    except (IndexError, ValueError) as e:
        raise ValueError(f"Could not parse filename: {filename}. Error: {e}")


def load_skyrmion_file_with_metadata(filepath):
    """
    Load skyrmion data AND extract metadata from filename.
    
    Returns:
    --------
    skyrmion_data : pandas DataFrame
    metadata : dict
    """
    skyrmion_data = load_skyrmion_file(filepath)
    metadata = parse_filename(filepath)
    
    print(f"  Metadata: Field={metadata['field']} Oe, T={metadata['temperature']} K, ID={metadata['id']}")
    print("\n" + "="*60 + "\n")
    
    return skyrmion_data, metadata

def visualize_skyrmions(skyrmion_data, title="Skyrmion Positions", show_actual_size=False):
    """
    Create a scatter plot of skyrmion positions with size representing area.
    
    Parameters:
    -----------
    skyrmion_data : pandas DataFrame
        DataFrame with Area, X, Y columns
    title : str
        Plot title
    show_actual_size : bool
        If True, attempts to show skyrmions at more accurate sizes
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if show_actual_size:
        # Calculate radius from area (assuming circular skyrmions)
        # Area = π * r², so r = sqrt(Area / π)
        skyrmion_data['radius'] = np.sqrt(skyrmion_data['Area'] / np.pi)
        
        # Plot each skyrmion as a circle with actual size
        for _, skyrmion in skyrmion_data.iterrows():
            circle = plt.Circle(
                (skyrmion['X'], skyrmion['Y']),
                skyrmion['radius'],
                color=plt.cm.viridis(skyrmion['Area'] / skyrmion_data['Area'].max()),
                alpha=0.6,
                edgecolor='black',
                linewidth=0.5
            )
            ax.add_patch(circle)
        
        # Set plot limits based on data range plus padding
        x_range = skyrmion_data['X'].max() - skyrmion_data['X'].min()
        y_range = skyrmion_data['Y'].max() - skyrmion_data['Y'].min()
        ax.set_xlim(skyrmion_data['X'].min() - 0.05*x_range, 
                    skyrmion_data['X'].max() + 0.05*x_range)
        ax.set_ylim(skyrmion_data['Y'].min() - 0.05*y_range, 
                    skyrmion_data['Y'].max() + 0.05*y_range)
        
        # Manual colorbar
        sm = plt.cm.ScalarMappable(
            cmap='viridis',
            norm=plt.Normalize(vmin=skyrmion_data['Area'].min(), 
                             vmax=skyrmion_data['Area'].max())
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        
    else:
        # Simplified scatter plot
        scatter = ax.scatter(
            skyrmion_data['X'], 
            skyrmion_data['Y'],
            s=skyrmion_data['Area'] / 5,  # Adjusted scaling
            alpha=0.6,
            c=skyrmion_data['Area'],
            cmap='viridis',
            edgecolors='black',
            linewidth=0.5
        )
        cbar = plt.colorbar(scatter, ax=ax)
    
    cbar.set_label('Skyrmion Area (pixels²)', rotation=270, labelpad=20)
    
    ax.set_xlabel('X Position (pixels)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # **FIX: Invert Y-axis to match image convention**
    ax.invert_yaxis()
    
    # Add summary statistics
    stats_text = f"N = {len(skyrmion_data)}\n"
    stats_text += f"Mean Area = {skyrmion_data['Area'].mean():.1f} px²\n"
    stats_text += f"Std Area = {skyrmion_data['Area'].std():.1f} px²"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def calculate_basic_stats(skyrmion_data, metadata=None):
    """
    Calculate basic skyrmion statistics.
    
    Parameters:
    -----------
    skyrmion_data : pandas DataFrame
        DataFrame with Area, X, Y columns
    metadata : dict, optional
        Metadata dictionary from parse_filename
    
    Returns:
    --------
    stats : dict
        Dictionary containing various statistics
    """
    # Estimate field of view from data range
    x_range = skyrmion_data['X'].max() - skyrmion_data['X'].min()
    y_range = skyrmion_data['Y'].max() - skyrmion_data['Y'].min()
    fov_area = x_range * y_range  # in pixels²
    
    stats = {
        # Count
        'n_skyrmions': len(skyrmion_data),
        
        # Area statistics
        'mean_area': skyrmion_data['Area'].mean(),
        'median_area': skyrmion_data['Area'].median(),
        'std_area': skyrmion_data['Area'].std(),
        'min_area': skyrmion_data['Area'].min(),
        'max_area': skyrmion_data['Area'].max(),
        
        # Density (number per unit area)
        'number_density': len(skyrmion_data) / fov_area if fov_area > 0 else 0,
        
        # Area coverage (fraction of FOV covered by skyrmions)
        'area_coverage': skyrmion_data['Area'].sum() / fov_area if fov_area > 0 else 0,
        
        # Field of view dimensions
        'fov_width': x_range,
        'fov_height': y_range,
        'fov_area': fov_area,
    }
    
    # Add metadata if provided
    if metadata:
        stats['field'] = metadata['field']
        stats['temperature'] = metadata['temperature']
        stats['id'] = metadata['id']
    
    return stats


def print_stats(stats):
    """Pretty print statistics dictionary."""
    print("\n" + "="*60)
    print("SKYRMION STATISTICS")
    print("="*60)
    
    if 'field' in stats:
        print(f"\nConditions:")
        print(f"  Field: {stats['field']} Oe")
        print(f"  Temperature: {stats['temperature']} K")
        print(f"  ID: {stats['id']}")
    
    print(f"\nCount:")
    print(f"  Number of skyrmions: {stats['n_skyrmions']}")
    
    print(f"\nArea Statistics (pixels²):")
    print(f"  Mean:   {stats['mean_area']:.1f}")
    print(f"  Median: {stats['median_area']:.1f}")
    print(f"  Std:    {stats['std_area']:.1f}")
    print(f"  Range:  {stats['min_area']:.0f} - {stats['max_area']:.0f}")
    
    print(f"\nDensity:")
    print(f"  Number density: {stats['number_density']:.6f} skyrmions/pixel²")
    print(f"  Area coverage:  {stats['area_coverage']:.2%}")
    
    print(f"\nField of View:")
    print(f"  Width:  {stats['fov_width']:.1f} pixels")
    print(f"  Height: {stats['fov_height']:.1f} pixels")
    print(f"  Area:   {stats['fov_area']:.0f} pixels²")
    
    print("="*60 + "\n")

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import cKDTree

def calculate_voronoi_coordination(skyrmion_data):
    """
    Calculate coordination numbers using Voronoi tessellation.
    
    The Voronoi tessellation divides space into regions (cells) where each 
    region contains all points closest to a particular skyrmion. Skyrmions
    whose cells share an edge are neighbors.
    
    Parameters:
    -----------
    skyrmion_data : pandas DataFrame
        DataFrame with Area, X, Y columns
    
    Returns:
    --------
    coord_stats : dict
        Dictionary with coordination statistics
    skyrmion_data : pandas DataFrame
        Cleaned dataframe with added 'coordination' column
    vor : scipy.spatial.Voronoi
        Voronoi object for visualization
    """
    # Clean data: remove any rows with NaN values
    skyrmion_data_clean = skyrmion_data.dropna(subset=['X', 'Y', 'Area']).copy()
    
    n_removed = len(skyrmion_data) - len(skyrmion_data_clean)
    if n_removed > 0:
        print(f"  ⚠ Removed {n_removed} skyrmions with missing position/area data")
    
    # Extract positions as array
    points = skyrmion_data_clean[['X', 'Y']].values
    
    if len(points) < 4:
        raise ValueError(f"Not enough valid points for Voronoi tessellation (need ≥4, have {len(points)})")
    
    # Compute Voronoi tessellation
    vor = Voronoi(points)
    
    # Calculate coordination number for each skyrmion
    coordination_numbers = []
    
    # vor.ridge_points contains pairs of point indices that share a Voronoi edge
    # This tells us which skyrmions are neighbors
    neighbor_dict = {i: set() for i in range(len(points))}
    
    for point_pair in vor.ridge_points:
        # Each pair shares an edge, so they're neighbors
        neighbor_dict[point_pair[0]].add(point_pair[1])
        neighbor_dict[point_pair[1]].add(point_pair[0])
    
    # Coordination number = number of neighbors
    coordination_numbers = [len(neighbors) for neighbors in neighbor_dict.values()]
    
    # Add coordination to dataframe
    skyrmion_data_clean['coordination'] = coordination_numbers
    
    # Calculate statistics
    coord_stats = {
        'mean_coordination': np.mean(coordination_numbers),
        'median_coordination': np.median(coordination_numbers),
        'std_coordination': np.std(coordination_numbers),
        'min_coordination': np.min(coordination_numbers),
        'max_coordination': np.max(coordination_numbers),
        'coordination_distribution': np.bincount(coordination_numbers),
    }
    
    # Calculate "packing efficiency" - how close to ideal hexagonal packing (6 neighbors)
    ideal_coordination = 6  # Hexagonal close packing
    coord_stats['packing_efficiency'] = np.mean(coordination_numbers) / ideal_coordination
    
    return coord_stats, skyrmion_data_clean, vor


def visualize_voronoi(skyrmion_data, vor, title="Voronoi Tessellation"):
    """
    Visualize the Voronoi tessellation with skyrmion positions.
    
    Parameters:
    -----------
    skyrmion_data : pandas DataFrame
        DataFrame with X, Y columns
    vor : scipy.spatial.Voronoi
        Voronoi object from calculate_voronoi_coordination
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot Voronoi diagram
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray', 
                    line_width=1, line_alpha=0.6, point_size=0)
    
    # Overlay skyrmion positions colored by coordination
    if 'coordination' in skyrmion_data.columns:
        scatter = ax.scatter(
            skyrmion_data['X'], 
            skyrmion_data['Y'],
            c=skyrmion_data['coordination'],
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
    else:
        ax.scatter(skyrmion_data['X'], skyrmion_data['Y'], 
                  s=100, color='red', edgecolors='black', linewidth=1, zorder=10)
    
    ax.set_xlabel('X Position (pixels)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def print_coordination_stats(coord_stats):
    """Pretty print coordination statistics."""
    print("\n" + "="*60)
    print("COORDINATION ANALYSIS (Voronoi)")
    print("="*60)
    
    print(f"\nCoordination Statistics:")
    print(f"  Mean:   {coord_stats['mean_coordination']:.2f}")
    print(f"  Median: {coord_stats['median_coordination']:.1f}")
    print(f"  Std:    {coord_stats['std_coordination']:.2f}")
    print(f"  Range:  {coord_stats['min_coordination']} - {coord_stats['max_coordination']}")
    
    print(f"\nPacking Analysis:")
    print(f"  Packing efficiency: {coord_stats['packing_efficiency']:.2%}")
    print(f"  (relative to ideal hexagonal packing with 6 neighbors)")
    
    print(f"\nCoordination Distribution:")
    for coord_num, count in enumerate(coord_stats['coordination_distribution']):
        if count > 0:
            print(f"  {coord_num} neighbors: {count} skyrmions")
    
    print("="*60 + "\n")

def check_data_quality(skyrmion_data):
    """
    Check for data quality issues.
    
    Parameters:
    -----------
    skyrmion_data : pandas DataFrame
        DataFrame to check
    """
    print("\nData Quality Check:")
    print(f"  Total rows: {len(skyrmion_data)}")
    print(f"  Missing X values: {skyrmion_data['X'].isna().sum()}")
    print(f"  Missing Y values: {skyrmion_data['Y'].isna().sum()}")
    print(f"  Missing Area values: {skyrmion_data['Area'].isna().sum()}")
    
    # Check for any weird values
    if len(skyrmion_data) > 0:
        print(f"  X range: {skyrmion_data['X'].min():.1f} to {skyrmion_data['X'].max():.1f}")
        print(f"  Y range: {skyrmion_data['Y'].min():.1f} to {skyrmion_data['Y'].max():.1f}")
        print(f"  Area range: {skyrmion_data['Area'].min():.1f} to {skyrmion_data['Area'].max():.1f}")

from pathlib import Path
import pandas as pd
from tqdm import tqdm  # For progress bars - install with: pip install tqdm

def find_all_datafiles(base_path=None):
    """
    Find all skyrmion data files in the directory structure.
    
    Parameters:
    -----------
    base_path : Path or str, optional
        Base path to search. Uses BASE_PATH if not provided.
    
    Returns:
    --------
    file_list : list of Path objects
        List of all found data files
    """
    if base_path is None:
        base_path = BASE_PATH
    else:
        base_path = Path(base_path)
    
    # Define the folders to search
    data_dir = base_path / "Skyrmion Data"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all .xlsx and .csv files in subdirectories
    file_list = []
    file_list.extend(data_dir.glob("**/*.xlsx"))
    file_list.extend(data_dir.glob("**/*.csv"))
    
    print(f"Found {len(file_list)} data files")
    
    # Organize by folder
    folder_counts = {}
    for f in file_list:
        folder = f.parent.name
        folder_counts[folder] = folder_counts.get(folder, 0) + 1
    
    print("\nFiles per folder:")
    for folder, count in sorted(folder_counts.items()):
        print(f"  {folder}: {count} files")
    
    return sorted(file_list)


def process_single_file(filepath, verbose=False):
    """
    Process a single file and return all statistics.
    
    Parameters:
    -----------
    filepath : Path or str
        Path to the data file
    verbose : bool
        If True, print detailed output
    
    Returns:
    --------
    results : dict
        Dictionary containing all statistics and metadata
    """
    try:
        # Load data and metadata
        if verbose:
            df, metadata = load_skyrmion_file_with_metadata(filepath)
        else:
            # Silent loading
            df = load_skyrmion_file(filepath)
            df = df[['Area', 'X', 'Y']].dropna()
            metadata = parse_filename(filepath)
        
        # Basic statistics
        stats = calculate_basic_stats(df, metadata)
        
        # Coordination analysis
        coord_stats, df_with_coord, vor = calculate_voronoi_coordination(df)
        
        # Combine all results
        results = {
            **metadata,  # field, temperature, id, filename
            **stats,     # all basic stats
            **{f'coord_{k}': v for k, v in coord_stats.items() 
               if k != 'coordination_distribution'}  # coordination stats (skip distribution array)
        }
        
        return results
        
    except Exception as e:
        print(f"  ✗ Error processing {Path(filepath).name}: {e}")
        return None


def batch_process_all_files(base_path=None, save_results=True):
    """
    Process all data files and compile results.
    
    Parameters:
    -----------
    base_path : Path or str, optional
        Base path to search
    save_results : bool
        If True, save results to CSV
    
    Returns:
    --------
    results_df : pandas DataFrame
        DataFrame with all results
    """
    print("\n" + "="*60)
    print("BATCH PROCESSING ALL FILES")
    print("="*60 + "\n")
    
    # Find all files
    file_list = find_all_datafiles(base_path)
    
    # Process each file
    results_list = []
    
    print("\nProcessing files...")
    for filepath in tqdm(file_list, desc="Processing"):
        result = process_single_file(filepath, verbose=False)
        if result is not None:
            results_list.append(result)
    
    # Create DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Sort by field and temperature
    results_df = results_df.sort_values(['field', 'temperature'], ascending=[True, False])
    
    print(f"\n✓ Successfully processed {len(results_df)} / {len(file_list)} files")
    
    if save_results:
        output_file = BASE_PATH / "batch_analysis_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"✓ Results saved to: {output_file}")
    
    return results_df


def summarize_results(results_df):
    """
    Print a summary of the batch processing results.
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        Results from batch_process_all_files
    """
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    
    print(f"\nTotal files processed: {len(results_df)}")
    
    print(f"\nField strengths analyzed:")
    for field in sorted(results_df['field'].unique()):
        count = len(results_df[results_df['field'] == field])
        print(f"  {field} Oe: {count} files")
    
    print(f"\nTemperature range:")
    print(f"  {results_df['temperature'].min()}K - {results_df['temperature'].max()}K")
    
    print(f"\nSkyrmion count range:")
    print(f"  {results_df['n_skyrmions'].min()} - {results_df['n_skyrmions'].max()} per file")
    
    print(f"\nMean area range:")
    print(f"  {results_df['mean_area'].min():.1f} - {results_df['mean_area'].max():.1f} pixels²")
    
    print(f"\nPacking efficiency range:")
    print(f"  {results_df['coord_packing_efficiency'].min():.1%} - {results_df['coord_packing_efficiency'].max():.1%}")
    
    print("="*60 + "\n")

def plot_vs_temperature(results_df, metric, ylabel, title=None, save_path=None):
    """
    Plot a metric as a function of temperature for each field strength.
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        Results from batch processing
    metric : str
        Column name to plot
    ylabel : str
        Y-axis label
    title : str, optional
        Plot title
    save_path : str or Path, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get unique field strengths
    fields = sorted(results_df['field'].unique())
    
    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    for i, field in enumerate(fields):
        # Filter data for this field
        field_data = results_df[results_df['field'] == field].sort_values('temperature')
        
        # Plot
        ax.plot(field_data['temperature'], field_data[metric], 
                marker=markers[i], linestyle='-', linewidth=2, markersize=8,
                color=colors[i], label=f'{field} Oe', alpha=0.8)
    
    ax.set_xlabel('Temperature (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(title='Applied Field', fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    plt.show()
    
    return fig, ax


def create_summary_plots(results_df, save_dir=None):
    """
    Create a comprehensive set of summary plots.
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        Results from batch processing
    save_dir : str or Path, optional
        Directory to save plots
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING SUMMARY PLOTS")
    print("="*60 + "\n")
    
    # Plot 1: Number of skyrmions vs temperature
    plot_vs_temperature(
        results_df, 
        'n_skyrmions', 
        'Number of Skyrmions',
        'Skyrmion Count vs Temperature',
        save_path=save_dir / "skyrmion_count_vs_temp.png" if save_dir else None
    )
    
    # Plot 2: Mean area vs temperature
    plot_vs_temperature(
        results_df, 
        'mean_area', 
        'Mean Skyrmion Area (pixels²)',
        'Skyrmion Size vs Temperature',
        save_path=save_dir / "skyrmion_size_vs_temp.png" if save_dir else None
    )
    
    # Plot 3: Number density vs temperature
    plot_vs_temperature(
        results_df, 
        'number_density', 
        'Number Density (skyrmions/pixel²)',
        'Skyrmion Density vs Temperature',
        save_path=save_dir / "density_vs_temp.png" if save_dir else None
    )
    
    # Plot 4: Packing efficiency vs temperature
    plot_vs_temperature(
        results_df, 
        'coord_packing_efficiency', 
        'Packing Efficiency',
        'Packing Efficiency vs Temperature',
        save_path=save_dir / "packing_efficiency_vs_temp.png" if save_dir else None
    )
    
    # Plot 5: Mean coordination vs temperature
    plot_vs_temperature(
        results_df, 
        'coord_mean_coordination', 
        'Mean Coordination Number',
        'Coordination Number vs Temperature',
        save_path=save_dir / "coordination_vs_temp.png" if save_dir else None
    )
    
    # Plot 6: Area coverage vs temperature
    plot_vs_temperature(
        results_df, 
        'area_coverage', 
        'Area Coverage (fraction)',
        'Skyrmion Area Coverage vs Temperature',
        save_path=save_dir / "coverage_vs_temp.png" if save_dir else None
    )
    
    print("\n✓ All plots generated!")


def create_heatmap_plot(results_df, metric, title, save_path=None):
    """
    Create a heatmap showing how a metric varies with both temperature and field.
    
    Parameters:
    -----------
    results_df : pandas DataFrame
        Results from batch processing
    metric : str
        Column name to plot
    title : str
        Plot title
    save_path : str or Path, optional
        Path to save figure
    """
    # Pivot data for heatmap
    pivot_data = results_df.pivot(index='temperature', columns='field', values=metric)
    
    # Sort by temperature (descending for heatmap - hot at top)
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
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.replace('_', ' ').title(), rotation=270, labelpad=20)
    
    # Add values as text
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            text = ax.text(j, i, f'{pivot_data.values[i, j]:.2f}',
                          ha="center", va="center", color="white", fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    plt.show()
    
    return fig, ax

import argparse

def create_cli():
    """
    Create command-line interface for the skyrmion analyzer.
    """
    parser = argparse.ArgumentParser(
        description='Skyrmion Data Analyzer - Batch process LTEM skyrmion data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch process all files and create plots
  python stat_analyzer.py --batch --plots
  
  # Analyze a single file
  python stat_analyzer.py --file "Skyrmion Data/OL=0800/OL=0800，286-1，T=200K -E.xlsx"
  
  # Batch process and save results only (no plots)
  python stat_analyzer.py --batch --output results.csv
        """
    )
    
    parser.add_argument('--batch', action='store_true',
                       help='Process all files in batch mode')
    
    parser.add_argument('--file', type=str,
                       help='Process a single file')
    
    parser.add_argument('--plots', action='store_true',
                       help='Generate summary plots (only with --batch)')
    
    parser.add_argument('--heatmap', action='store_true',
                       help='Generate heatmap plots (only with --batch)')
    
    parser.add_argument('--output', type=str,
                       help='Output CSV filename (default: batch_analysis_results.csv)')
    
    parser.add_argument('--plot-dir', type=str, default='analysis_plots',
                       help='Directory for saving plots (default: analysis_plots)')
    
    parser.add_argument('--show-viz', action='store_true',
                       help='Show detailed visualizations for single file analysis')
    
    return parser


def main():
    """Main entry point for CLI."""
    parser = create_cli()
    args = parser.parse_args()
    
    # Check that at least one mode is selected
    if not args.batch and not args.file:
        parser.print_help()
        print("\n❌ Error: Must specify either --batch or --file")
        return
    
    # Single file mode
    if args.file:
        print("\n" + "="*60)
        print("SINGLE FILE ANALYSIS")
        print("="*60 + "\n")
        
        filepath = args.file
        df, metadata = load_skyrmion_file_with_metadata(filepath)
        check_data_quality(df)
        
        # Basic stats
        stats = calculate_basic_stats(df, metadata)
        print_stats(stats)
        
        # Coordination analysis
        coord_stats, df_with_coord, vor = calculate_voronoi_coordination(df)
        print_coordination_stats(coord_stats)
        
        # Visualizations (optional)
        if args.show_viz:
            print("\n📊 Generating visualizations (close windows to continue)...")
            visualize_skyrmions(df_with_coord, 
                              title=f"Field={metadata['field']}Oe, T={metadata['temperature']}K")
            visualize_voronoi(df_with_coord, vor,
                            title=f"Voronoi: Field={metadata['field']}Oe, T={metadata['temperature']}K")
            print("✓ Visualizations closed")
        
        # Don't continue to batch processing
        print("\n✅ Single file analysis complete!\n")
        return  # EXIT HERE - don't run batch code
    
    # Batch mode (only runs if --file was NOT specified)
    if args.batch:
        # Determine output filename
        output_csv = args.output if args.output else "batch_analysis_results.csv"
        output_path = BASE_PATH / output_csv
        
        # Process all files
        results_df = batch_process_all_files(save_results=True)
        summarize_results(results_df)
        
        # Save with custom name if specified
        if args.output:
            results_df.to_csv(output_path, index=False)
            print(f"✓ Results saved to: {output_path}")
        
        # Generate plots if requested
        if args.plots or args.heatmap:
            plots_dir = BASE_PATH / args.plot_dir
            plots_dir.mkdir(exist_ok=True)
            
            if args.plots:
                create_summary_plots(results_df, save_dir=plots_dir)
            
            if args.heatmap:
                print("\nGenerating heatmaps...")
                create_heatmap_plot(
                    results_df, 
                    'coord_packing_efficiency',
                    'Packing Efficiency: Temperature vs Field',
                    save_path=plots_dir / "packing_efficiency_heatmap.png"
                )
                create_heatmap_plot(
                    results_df,
                    'number_density',
                    'Number Density: Temperature vs Field',
                    save_path=plots_dir / "density_heatmap.png"
                )
        
        print("\n✅ Batch analysis complete!\n")


if __name__ == "__main__":
    main()