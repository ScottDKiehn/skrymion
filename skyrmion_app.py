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

# Import your analysis functions
from stat_analyzer import (
    load_skyrmion_file,
    parse_filename,
    calculate_basic_stats,
    calculate_voronoi_coordination,
    visualize_skyrmions,
    visualize_voronoi,
    print_stats,
    print_coordination_stats
)

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


def process_uploaded_file(uploaded_file):
    """
    Process a single uploaded file.
    
    Returns:
    --------
    df : DataFrame with skyrmion data
    metadata : dict with file metadata
    stats : dict with basic statistics
    coord_stats : dict with coordination statistics
    df_with_coord : DataFrame with coordination numbers
    vor : Voronoi object
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
        
        stats = calculate_basic_stats(df, metadata)
        coord_stats, df_with_coord, vor = calculate_voronoi_coordination(df)
        
        return df, metadata, stats, coord_stats, df_with_coord, vor
        
    finally:
        # Clean up temp file
        Path(tmp_path).unlink()


def display_stats_cards(stats, coord_stats):
    """Display statistics in nice cards."""
    col1, col2, col3, col4 = st.columns(4)
    
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


def plot_interactive_scatter(df_with_coord, metadata):
    """Create an interactive scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    scatter = ax.scatter(
        df_with_coord['X'], 
        df_with_coord['Y'],
        s=df_with_coord['Area'] / 5,
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
    
    ax.set_xlabel('X Position (pixels)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)
    ax.set_title(f"Field={metadata['field']}Oe, T={metadata['temperature']}K", 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_voronoi_diagram(df_with_coord, vor, metadata):
    """Create Voronoi diagram."""
    from scipy.spatial import voronoi_plot_2d
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray', 
                    line_width=1, line_alpha=0.6, point_size=0)
    
    scatter = ax.scatter(
        df_with_coord['X'], 
        df_with_coord['Y'],
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
    
    ax.set_xlabel('X Position (pixels)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)
    ax.set_title(f"Voronoi Tessellation: Field={metadata['field']}Oe, T={metadata['temperature']}K", 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    return fig


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
                             "📦 Batch Analysis"])
    
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
        
        if uploaded_file is not None:
            with st.spinner('🔄 Processing file...'):
                try:
                    df, metadata, stats, coord_stats, df_with_coord, vor = process_uploaded_file(uploaded_file)
                    
                    st.success(f"✅ Successfully loaded: {uploaded_file.name}")
                    
                    # Display statistics
                    st.markdown("---")
                    st.subheader("📊 Statistics")
                    display_stats_cards(stats, coord_stats)
                    
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
                    
                    tab1, tab2, tab3 = st.tabs(["🎯 Scatter Plot", "🕸️ Voronoi Diagram", "📋 Data Table"])
                    
                    with tab1:
                        fig1 = plot_interactive_scatter(df_with_coord, metadata)
                        st.pyplot(fig1)
                        plt.close()
                    
                    with tab2:
                        fig2 = plot_voronoi_diagram(df_with_coord, vor, metadata)
                        st.pyplot(fig2)
                        plt.close()
                    
                    with tab3:
                        st.dataframe(df_with_coord, width='stretch')
                        
                        # Download button
                        csv = df_with_coord.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Data as CSV",
                            data=csv,
                            file_name=f"analyzed_{uploaded_file.name}.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"❌ Error processing file: {e}")
                    st.exception(e)
    
# ============================================================
    # BATCH MODE
    # ============================================================
    else:  # Batch Analysis
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
                            df, metadata, stats, coord_stats, df_with_coord, vor = process_uploaded_file(uploaded_file)
                            
                            # Combine results
                            result = {
                                **metadata,
                                **stats,
                                **{f'coord_{k}': v for k, v in coord_stats.items() 
                                   if k != 'coordination_distribution'}
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
            
            # Download button
            csv = results_df.to_csv(index=False)
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


if __name__ == "__main__":
    main()