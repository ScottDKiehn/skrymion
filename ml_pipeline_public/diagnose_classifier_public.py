#!/usr/bin/env python3
"""
Diagnostic tool for topological classifier.

Visualizes radial profiles and classification decisions to help tune parameters.

Author: AI Assistant + Scott Kiehn
Date: 2025-11-11
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage.measure import regionprops
from topological_classifier_public import TopologicalClassifier


def diagnose_classification(image_path, labels_path=None, n_examples=None,
                          peak_prominence=0.15, center_falloff_sigma=0.2,
                          skyrmionium_threshold=0.3,
                          center_window_fraction=0.3, top_n_angles=4,
                          merge_threshold=0.15, width_bonus_enabled=True,
                          use_nonlinear_window=True, window_scale_a=-0.27,
                          window_scale_b=0.13, angle_selection_mode='top_n',
                          selected_angles=None, page=1, page_size=50):
    """
    Visualize radial profiles and classification for debugging.

    IMPROVED: Now accepts ALL classifier parameters to respect UI settings
              Shows ALL skyrmions by default
              Visualizes peak merging, width factors, and top-N angle selection
              PAGINATED: Shows max 50 skyrmions per page to avoid image size errors

    Parameters:
    -----------
    image_path : str or Path
        Path to LTEM image
    labels_path : str or Path, optional
        Path to label mask. If None, uses StarDist to segment.
    n_examples : int, optional
        Number of skyrmions to show. If None, shows ALL detected skyrmions
    peak_prominence : float
        Peak detection prominence threshold
    center_falloff_sigma : float
        Gaussian falloff parameter for center weighting (DEPRECATED)
    skyrmionium_threshold : float
        Classification threshold
    center_window_fraction : float
        Fraction of radius defining 'center region' (default: 0.3)
    top_n_angles : int
        Number of best angles to use (1-4, default: 4)
    merge_threshold : float
        Fraction of radius for merging peaks (default: 0.15)
    width_bonus_enabled : bool
        Enable width-aware scoring (default: True)
    use_nonlinear_window : bool
        Enable logarithmic window scaling (default: True)
    window_scale_a : float
        Logarithmic scaling intercept (default: -0.27)
    window_scale_b : float
        Logarithmic scaling slope (default: 0.13)
    page : int
        Page number to display (1-indexed, default: 1)
    page_size : int
        Number of skyrmions per page (default: 50)

    Returns:
    --------
    total_pages : int
        Total number of pages available
    """
    # Load image
    img = Image.open(image_path)
    if img.mode == 'RGB':
        img = img.convert('L')
    image = np.array(img, dtype=np.float32)

    # Load or generate labels
    if labels_path is None:
        print("Running StarDist segmentation...")
        from extract_properties_public import SkyrmionPropertyExtractor
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
        properties, labels = extractor.process_image(image_path, save_visualization=False)
    else:
        labels = np.array(Image.open(labels_path), dtype=np.uint16)

    # Initialize classifier with ALL provided parameters
    classifier = TopologicalClassifier(
        n_radial_points=50,
        smoothing_window=11,
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

    # Get skyrmion properties
    props = regionprops(labels)
    total_skyrmions = len(props)

    # Calculate pagination
    import math
    total_pages = math.ceil(total_skyrmions / page_size)

    # Validate page number
    if page < 1:
        page = 1
    if page > total_pages:
        page = total_pages

    # Calculate start and end indices for this page
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_skyrmions)

    # Apply n_examples limit if specified (takes precedence over pagination)
    if n_examples is not None:
        end_idx = min(end_idx, n_examples)

    n_show = end_idx - start_idx

    print(f"\n{'='*70}")
    print(f"PAGINATION INFO")
    print(f"{'='*70}")
    print(f"Total skyrmions detected: {total_skyrmions}")
    print(f"Page {page} of {total_pages}")
    print(f"Showing skyrmions {start_idx + 1}-{end_idx} ({n_show} skyrmions)")
    print(f"{'='*70}\n")

    # Create diagnostic plot
    cols = 3
    rows = n_show
    fig = plt.figure(figsize=(15, 5 * rows))

    for i, prop in enumerate(props[start_idx:end_idx]):
        cy, cx = prop.centroid
        radius = np.sqrt(prop.area / np.pi)

        # Extract 4 diameter profiles
        profiles = classifier.extract_diameter_profiles(image, (cy, cx), radius)

        # Run classification
        classification, confidence, details = classifier.classify(image, (cy, cx), radius)

        # Plot 1: Skyrmion patch
        ax1 = plt.subplot(rows, cols, i * cols + 1)

        # Extract patch around skyrmion
        patch_size = int(radius * 3)
        y_min = max(0, int(cy - patch_size))
        y_max = min(image.shape[0], int(cy + patch_size))
        x_min = max(0, int(cx - patch_size))
        x_max = min(image.shape[1], int(cx + patch_size))

        patch = image[y_min:y_max, x_min:x_max]
        ax1.imshow(patch, cmap='gray')

        # Draw circle at center
        center_in_patch = (cx - x_min, cy - y_min)
        circle = plt.Circle(center_in_patch, radius, fill=False, edgecolor='red', linewidth=2)
        ax1.add_patch(circle)

        ax1.set_title(f'Skyrmion {prop.label}\n{classification.upper()} (conf={confidence:.2f})')
        ax1.axis('off')

        # Plot 2: Diameter profiles (4 lines) with NEW peak annotations
        ax2 = plt.subplot(rows, cols, i * cols + 2)

        angle_labels = ['0°', '45°', '90°', '135°']
        colors = ['red', 'orange', 'green', 'blue']

        for j, (profile, angle_label, color) in enumerate(zip(profiles, angle_labels, colors)):
            smoothed = classifier.smooth_profile(profile)
            distances = np.linspace(-radius * 1.5, radius * 1.5, len(profile))

            # Check if this angle was selected in top-N
            is_selected = details['diameter_scores'][j]['selected']
            linestyle = '-' if is_selected else ':'
            alpha = 0.9 if is_selected else 0.3

            ax2.plot(distances, smoothed, linestyle, linewidth=2, label=f'{angle_label}{"*" if is_selected else ""}',
                    color=color, alpha=alpha)

            # Get peak details for this angle
            angle_details = details['diameter_scores'][j]['details']
            if 'peaks' in angle_details and len(angle_details['peaks']) > 0:
                for peak_info in angle_details['peaks']:
                    # Accepted peaks (green circle)
                    ax2.plot(peak_info['position'], smoothed[np.argmin(np.abs(distances - peak_info['position']))],
                            'o', markersize=8, color='green', markeredgecolor='black', markeredgewidth=1.5,
                            zorder=10)

                    # Annotation with peak info
                    ax2.text(peak_info['position'], smoothed[np.argmin(np.abs(distances - peak_info['position']))] + 0.05,
                            f"w={peak_info['width']:.0f}\np={peak_info['position_weight']:.2f}",
                            fontsize=6, ha='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

                    # Mark merged peaks with blue annotation
                    if peak_info.get('was_merged', False):
                        ax2.plot(peak_info['position'], smoothed[np.argmin(np.abs(distances - peak_info['position']))],
                                's', markersize=10, color='blue', markerfacecolor='none', markeredgewidth=2, zorder=11)

        # Show center window region (size-scaled)
        center_window_size = classifier.center_window_fraction * radius
        ax2.axvspan(-center_window_size, center_window_size, color='cyan', alpha=0.1, label='Center Window')
        ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)

        ax2.set_xlabel('Distance from center (pixels)', fontweight='bold')
        ax2.set_ylabel('Intensity (normalized)', fontweight='bold')
        ax2.set_title(f'Diameter Profiles\n(* = selected in top-{classifier.top_n_angles})', fontsize=10)
        ax2.legend(fontsize=6, loc='upper right', ncol=2)
        ax2.grid(alpha=0.3)

        # Plot 3: Diameter scores with NEW top-N selection visualization
        ax3 = plt.subplot(rows, cols, i * cols + 3)

        # Get individual diameter scores and selection status
        diameter_scores = [d['score'] for d in details['diameter_scores']]
        selected = [d['selected'] for d in details['diameter_scores']]
        angles = ['0°', '45°', '90°', '135°']
        colors_bars = ['red', 'orange', 'green', 'blue']

        # Color bars based on selection (bright if selected, dim if not)
        bar_colors = []
        for j, (color, is_selected) in enumerate(zip(colors_bars, selected)):
            if is_selected:
                bar_colors.append(color)
            else:
                bar_colors.append('#CCCCCC')  # Gray for non-selected

        bars = ax3.bar(angles, diameter_scores, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=2)

        # Highlight selected bars with star
        for j, (score, is_selected) in enumerate(zip(diameter_scores, selected)):
            if is_selected:
                ax3.text(j, score + 0.02, f'{score:.3f}★', ha='center', fontsize=8, fontweight='bold', color='darkgreen')
            else:
                ax3.text(j, score + 0.02, f'{score:.3f}', ha='center', fontsize=7, color='gray')

        # Show threshold line and scores
        normalized_score = details['normalized_score']
        threshold = details['threshold']
        mean_score = details.get('mean_score', normalized_score)
        n_angles_used = details.get('n_angles_used', 4)
        selected_sum = sum([s for s, sel in zip(diameter_scores, selected) if sel])

        ax3.axhline(threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Threshold ({threshold:.2f})')
        ax3.set_ylim([0, max(1.0, max(diameter_scores) * 1.2)])
        ax3.set_ylabel('Skyrmionium Score', fontweight='bold')
        ax3.set_xlabel('Diameter Angle', fontweight='bold')

        # Classification result with detailed score breakdown
        result_color = 'red' if classification == 'skyrmionium' else 'blue'
        ax3.set_title(
            f'Diameter Scores (Top-{n_angles_used} used: ★)\n'
            f'Sum: {selected_sum:.3f} | Mean: {mean_score:.3f} | Norm: {normalized_score:.3f}\n'
            f'→ {classification.upper()}',
            color=result_color,
            fontweight='bold',
            fontsize=9
        )

        # Add legend with parameters
        param_text = f"Window: {details.get('center_window_fraction', 0.3):.2f}×R\nMerge: {details.get('merge_threshold', 0.15):.2f}×R"
        ax3.text(0.98, 0.97, param_text, transform=ax3.transAxes,
                verticalalignment='top', horizontalalignment='right',
                fontsize=7, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax3.legend(fontsize=7, loc='upper left')
        ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save diagnostic plot with page number for caching
    output_path = Path('ml_pipeline_public/results') / f'classifier_diagnostics_{Path(image_path).stem}_page{page}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved diagnostics to {output_path}")

    plt.show()

    # Print summary statistics
    classifications = []
    for prop in props:
        cy, cx = prop.centroid
        radius = np.sqrt(prop.area / np.pi)
        classification, confidence, details = classifier.classify(image, (cy, cx), radius)
        classifications.append(classification)

    n_skyrmions = classifications.count('skyrmion')
    n_skyrmioniums = classifications.count('skyrmionium')

    print("\n" + "=" * 70)
    print("CLASSIFICATION SUMMARY")
    print("=" * 70)
    print(f"Total detected: {len(classifications)}")
    print(f"Classified as skyrmions (Q=±1): {n_skyrmions} ({n_skyrmions/len(classifications)*100:.1f}%)")
    print(f"Classified as skyrmioniums (Q=0): {n_skyrmioniums} ({n_skyrmioniums/len(classifications)*100:.1f}%)")
    print("\nIf classification seems random:")
    print("1. Check radial profiles - do you see clear peaks for some objects?")
    print("2. Adjust peak_prominence (currently 0.1) - lower = more sensitive")
    print("3. LTEM contrast may be too low for reliable classification")
    print("4. Consider manual labeling of 20-30 examples for supervised training")

    # Return total pages for UI pagination
    return total_pages


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose topological classifier")
    parser.add_argument('image', type=str, help='Path to LTEM image')
    parser.add_argument('--labels', type=str, help='Path to label mask (optional)')
    parser.add_argument('--n', type=int, default=None, help='Number of examples to show (default: ALL)')
    parser.add_argument('--peak-prominence', type=float, default=0.15, help='Peak prominence threshold')
    parser.add_argument('--center-falloff', type=float, default=0.2, help='Center falloff sigma (deprecated)')
    parser.add_argument('--threshold', type=float, default=0.3, help='Skyrmionium threshold')
    parser.add_argument('--center-window', type=float, default=0.3, help='Center window fraction (linear mode)')
    parser.add_argument('--top-n-angles', type=int, default=4, help='Number of best angles to use (1-4)')
    parser.add_argument('--merge-threshold', type=float, default=0.15, help='Peak merge threshold')
    parser.add_argument('--no-width-bonus', action='store_true', help='Disable width-aware scoring')
    parser.add_argument('--linear-window', action='store_true', help='Use linear window scaling (default: logarithmic)')
    parser.add_argument('--window-scale-a', type=float, default=-0.27, help='Logarithmic scaling intercept')
    parser.add_argument('--window-scale-b', type=float, default=0.13, help='Logarithmic scaling slope')
    parser.add_argument('--page', type=int, default=1, help='Page number to display (default: 1)')
    parser.add_argument('--page-size', type=int, default=50, help='Skyrmions per page (default: 50)')

    args = parser.parse_args()

    diagnose_classification(
        args.image,
        args.labels,
        args.n,
        peak_prominence=args.peak_prominence,
        center_falloff_sigma=args.center_falloff,
        skyrmionium_threshold=args.threshold,
        center_window_fraction=args.center_window,
        top_n_angles=args.top_n_angles,
        merge_threshold=args.merge_threshold,
        width_bonus_enabled=not args.no_width_bonus,
        use_nonlinear_window=not args.linear_window,
        window_scale_a=args.window_scale_a,
        window_scale_b=args.window_scale_b,
        page=args.page,
        page_size=args.page_size
    )
