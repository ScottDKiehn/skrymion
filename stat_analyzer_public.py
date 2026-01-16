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
    Note: OL=2000/3000 files may have encoding issues (ĢŽ instead of ，)

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

    # Strip common prefixes from ML Detection exports
    base_name = filename
    if base_name.startswith('ml_detected_'):
        base_name = base_name[len('ml_detected_'):]

    # Fix encoding: Replace ĢŽ with full-width comma (for OL=2000/3000 files)
    base_name = base_name.replace('ĢŽ', '，')

    # Remove file extension and any "（new）" or "(new)" suffix
    base_name = base_name.replace('（new）', '').replace('(new)', '')
    base_name = base_name.replace('.xlsx', '').replace('.csv', '').replace('.png', '').replace('.jpg', '')

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
from scipy.spatial.distance import pdist, squareform

def calculate_voronoi_coordination(skyrmion_data, neighbor_distance_factor=3.0):
    """
    Calculate coordination numbers using both distance-aware and Voronoi methods.

    Distance-aware method: Uses adaptive Heaviside cutoff where each skyrmion pair
    has a cutoff distance = neighbor_distance_factor × (r_i + r_j) / 2, where r = sqrt(Area/π).

    Default factor of 2.0 accounts for lattice spacing. Increase if lattice is sparse,
    decrease if skyrmions are densely packed.

    Voronoi method: Traditional topological neighbors (cells sharing edges).

    Parameters:
    -----------
    skyrmion_data : pandas DataFrame
        DataFrame with Area, X, Y columns
    neighbor_distance_factor : float, optional (default=3.0)
        Multiplier for neighbor cutoff distance.
        - 1.0 = touching skyrmions (r_i + r_j) / 2
        - 2.0 = moderate spacing
        - 3.0 = typical lattice spacing (recommended default)

    Returns:
    --------
    coord_stats : dict
        Dictionary with coordination statistics (now includes both methods)
    skyrmion_data : pandas DataFrame
        Cleaned dataframe with added 'coordination' and 'topological_coordination' columns
    vor : scipy.spatial.Voronoi
        Voronoi object for visualization
    neighbor_dict_distance : dict
        Distance-based neighbor relationships (point_idx -> set of neighbor indices)
    """
    # Clean data: remove any rows with NaN values
    skyrmion_data_clean = skyrmion_data.dropna(subset=['X', 'Y', 'Area']).copy()

    n_removed = len(skyrmion_data) - len(skyrmion_data_clean)
    if n_removed > 0:
        print(f"  ⚠ Removed {n_removed} skyrmions with missing position/area data")

    # Extract positions and areas
    points = skyrmion_data_clean[['X', 'Y']].values
    areas = skyrmion_data_clean['Area'].values

    if len(points) < 4:
        raise ValueError(f"Not enough valid points for Voronoi tessellation (need ≥4, have {len(points)})")

    # Calculate radii for each skyrmion: r = sqrt(Area/π)
    radii = np.sqrt(areas / np.pi)

    # === DISTANCE-AWARE COORDINATION ===
    # Calculate pairwise distances
    dist_matrix = squareform(pdist(points))

    # Build distance-based neighbor dictionary with adaptive cutoff
    neighbor_dict_distance = {i: set() for i in range(len(points))}
    distance_coordination = np.zeros(len(points), dtype=int)

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            # Adaptive cutoff: factor × average radius of the pair
            # Average radius of pair = (r_i + r_j) / 2
            avg_radius = (radii[i] + radii[j]) / 2
            cutoff_distance = neighbor_distance_factor * avg_radius

            if dist_matrix[i, j] <= cutoff_distance:
                neighbor_dict_distance[i].add(j)
                neighbor_dict_distance[j].add(i)
                distance_coordination[i] += 1
                distance_coordination[j] += 1

    # === VORONOI TOPOLOGICAL COORDINATION ===
    # Compute Voronoi tessellation for visualization and comparison
    vor = Voronoi(points)

    # vor.ridge_points contains pairs of point indices that share a Voronoi edge
    neighbor_dict_voronoi = {i: set() for i in range(len(points))}

    for point_pair in vor.ridge_points:
        neighbor_dict_voronoi[point_pair[0]].add(point_pair[1])
        neighbor_dict_voronoi[point_pair[1]].add(point_pair[0])

    topological_coordination = np.array([len(neighbors) for neighbors in neighbor_dict_voronoi.values()])

    # Add both coordination columns to dataframe
    skyrmion_data_clean['coordination'] = distance_coordination  # Primary (distance-aware)
    skyrmion_data_clean['topological_coordination'] = topological_coordination  # For comparison

    # Calculate statistics for DISTANCE-AWARE coordination (primary)
    coord_stats = {
        # Distance-aware metrics (primary)
        'mean_coordination': np.mean(distance_coordination),
        'median_coordination': np.median(distance_coordination),
        'std_coordination': np.std(distance_coordination),
        'min_coordination': np.min(distance_coordination),
        'max_coordination': np.max(distance_coordination),
        'coordination_distribution': np.bincount(distance_coordination),

        # Topological metrics (for comparison)
        'mean_topological_coordination': np.mean(topological_coordination),
        'median_topological_coordination': np.median(topological_coordination),
        'std_topological_coordination': np.std(topological_coordination),

        # Mean radius for reference
        'mean_radius_pixels': np.mean(radii),
        'std_radius_pixels': np.std(radii),

        # Cutoff parameters
        'neighbor_distance_factor': neighbor_distance_factor,
        'mean_cutoff_distance': neighbor_distance_factor * np.mean(radii),
    }

    # Calculate "packing efficiency" - how close to ideal hexagonal packing (6 neighbors)
    ideal_coordination = 6  # Hexagonal close packing
    coord_stats['packing_efficiency'] = np.mean(distance_coordination) / ideal_coordination
    coord_stats['topological_packing_efficiency'] = np.mean(topological_coordination) / ideal_coordination

    return coord_stats, skyrmion_data_clean, vor, neighbor_dict_distance


def calculate_bond_orientation_order(skyrmion_data, neighbor_dict):
    """
    Calculate bond orientation order parameter (hexagonal symmetry).

    For each skyrmion, computes the local 6-fold bond orientation order:
        φ_i = (1/N_i) × Σ_j exp(i × 6 × θ_j)
    where θ_j is the angle of the bond to neighbor j, and N_i is the number of neighbors.

    Also computes global order parameter across all bonds in the system.

    Parameters:
    -----------
    skyrmion_data : pandas DataFrame
        DataFrame with X, Y columns (must match indices used in neighbor_dict)
    neighbor_dict : dict
        Dictionary mapping skyrmion index -> set of neighbor indices

    Returns:
    --------
    bond_order_results : dict
        Dictionary containing:
        - 'local_phi': numpy array of |φ_i| for each skyrmion (0-1, 1=perfect hexagonal order)
        - 'global_phi': complex global order parameter
        - 'global_phi_magnitude': |φ_global|
        - 'mean_local_phi': mean of local |φ| values
        - 'std_local_phi': std of local |φ| values
        - 'median_local_phi': median of local |φ| values
        - 'min_local_phi': minimum local |φ|
        - 'max_local_phi': maximum local |φ|
    """
    points = skyrmion_data[['X', 'Y']].values
    n_skyrmions = len(points)

    # Array to store local bond orientation order for each skyrmion
    local_phi = np.zeros(n_skyrmions, dtype=complex)
    local_phi_magnitude = np.zeros(n_skyrmions)

    # List to collect all bond angles for global order parameter
    all_bond_angles = []

    # Calculate local order parameter for each skyrmion
    for i in range(n_skyrmions):
        neighbors = list(neighbor_dict[i])
        n_neighbors = len(neighbors)

        if n_neighbors == 0:
            # Isolated skyrmion has undefined order
            local_phi_magnitude[i] = 0.0
            continue

        # Calculate bond angles to all neighbors
        bond_angles = []
        for j in neighbors:
            # Vector from skyrmion i to neighbor j
            dx = points[j, 0] - points[i, 0]
            dy = points[j, 1] - points[i, 1]

            # Bond angle (in radians)
            theta = np.arctan2(dy, dx)
            bond_angles.append(theta)
            all_bond_angles.append(theta)  # Also collect for global calculation

        # Compute local 6-fold order parameter
        # φ_i = (1/N) × Σ exp(i × 6 × θ)
        bond_angles = np.array(bond_angles)
        phi_i = np.mean(np.exp(1j * 6 * bond_angles))

        local_phi[i] = phi_i
        local_phi_magnitude[i] = np.abs(phi_i)

    # Calculate global order parameter (average over all bonds)
    if len(all_bond_angles) > 0:
        all_bond_angles = np.array(all_bond_angles)
        global_phi = np.mean(np.exp(1j * 6 * all_bond_angles))
        global_phi_magnitude = np.abs(global_phi)
    else:
        global_phi = 0.0
        global_phi_magnitude = 0.0

    # Compile statistics
    bond_order_results = {
        'local_phi': local_phi_magnitude,  # Return magnitudes (real values)
        'local_phi_complex': local_phi,    # Keep complex values for advanced analysis
        'global_phi': global_phi,
        'global_phi_magnitude': global_phi_magnitude,
        'mean_local_phi': np.mean(local_phi_magnitude),
        'std_local_phi': np.std(local_phi_magnitude),
        'median_local_phi': np.median(local_phi_magnitude),
        'min_local_phi': np.min(local_phi_magnitude),
        'max_local_phi': np.max(local_phi_magnitude),
    }

    return bond_order_results


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


def plot_size_histogram(skyrmion_data, metadata=None, bins='auto', show_kde=True, scale_factor=1.0, unit_name='pixels'):
  """
  Create histogram of skyrmion sizes with statistical overlay.

  Parameters:
  -----------
  skyrmion_data : pandas DataFrame
      DataFrame with Area column
  metadata : dict, optional
      Metadata with field, temperature info
  bins : int or str
      Number of bins or method ('auto', 'sturges', 'fd', etc.)
  show_kde : bool
      If True, overlay kernel density estimate
  scale_factor : float
      Scale factor to convert pixels to physical units (default: 1.0)
  unit_name : str
      Name of the physical unit (default: 'pixels')

  Returns:
  --------
  fig, ax : matplotlib figure and axis
  """
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

  if metadata:
      title = f"Size Distribution: Field={metadata['field']}Oe, T={metadata['temperature']}K"
  else:
      title = "Skyrmion Size Distribution"
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

  return fig, ax


def assess_distribution_modality(skyrmion_data, scale_factor=1.0, unit_name='pixels'):
    """
    Assess whether size distribution is unimodal or multimodal.
    Uses Hartigan's dip test for unimodality.

    Parameters:
    -----------
    skyrmion_data : pandas DataFrame
        DataFrame with Area column
    scale_factor : float
        Scale factor to convert pixels to physical units (default: 1.0)
    unit_name : str
        Name of the physical unit (default: 'pixels')

    Returns:
    --------
    assessment : dict
        Dictionary with modality assessment results
    """
    from scipy import stats

    areas = skyrmion_data['Area'].values
    # Convert to diameter for assessment
    diameters = 2 * np.sqrt(areas / np.pi) * scale_factor

    # Basic statistics
    skewness = stats.skew(diameters)
    kurtosis = stats.kurtosis(diameters)

    # Simple bimodality coefficient (BC)
    # BC > 0.555 suggests bimodality for n > 20
    n = len(diameters)
    BC = (skewness**2 + 1) / (kurtosis + 3 * (n-1)**2 / ((n-2)*(n-3)))

    assessment = {
        'n_skyrmions': n,
        'mean': diameters.mean(),
        'std': diameters.std(),
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
        
        # NEW: Size distribution analysis
        print("\n" + "="*60)
        print("SIZE DISTRIBUTION ANALYSIS")
        print("="*60)
        modality = assess_distribution_modality(df_with_coord)
        print(f"\nDistribution Assessment:")
        print(f"  Bimodality Coefficient: {modality['bimodality_coefficient']:.3f}")
        print(f"  Skewness: {modality['skewness']:.2f}")
        print(f"  Kurtosis: {modality['kurtosis']:.2f}")
        print(f"\n  {modality['interpretation']}")
        print("="*60 + "\n")
        
        # Visualizations (optional)
        if args.show_viz:
            print("\n📊 Generating visualizations (close windows to continue)...")
            
            # Original visualizations
            visualize_skyrmions(df_with_coord, 
                              title=f"Field={metadata['field']}Oe, T={metadata['temperature']}K")
            visualize_voronoi(df_with_coord, vor,
                            title=f"Voronoi: Field={metadata['field']}Oe, T={metadata['temperature']}K")
            
            # NEW: Size histogram
            plot_size_histogram(df_with_coord, metadata)
            
            print("✓ Visualizations closed")
        
        print("\n✅ Single file analysis complete!\n")
        return  # EXIT HERE
    
    # Batch mode continues as before...
    
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