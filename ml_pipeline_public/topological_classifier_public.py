#!/usr/bin/env python3
"""
Topological charge classification for magnetic skyrmions.

Classifies detected skyrmions as Q=±1 (skyrmion) or Q=0 (skyrmionium)
using ensemble voting of three algorithmic methods based on radial
intensity profiles.

Author: AI Assistant + Scott Kiehn
Date: 2025-11-11
"""

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks, savgol_filter
from skimage.measure import regionprops


class TopologicalClassifier:
    """
    Peak-based classifier for skyrmion vs skyrmionium distinction.

    Uses diameter profile analysis with Gaussian proximity weighting:
    - Skyrmioniums have prominent peaks at CENTER of diameter profiles
    - Peak score = peak_height × exp(-(distance_from_center)^2 / (2*sigma^2))
    - Total score normalized to 0-1 and compared to threshold
    """

    def __init__(self, n_radial_points=50, smoothing_window=11, peak_prominence=0.15,
                 center_falloff_sigma=0.2, skyrmionium_threshold=0.3,
                 center_window_fraction=0.3, top_n_angles=4, merge_threshold=0.15,
                 width_bonus_enabled=True, window_falloff_sharpness=0.1,
                 use_nonlinear_window=True, window_scale_a=-0.27, window_scale_b=0.13,
                 angle_selection_mode='top_n', selected_angles=None):
        """
        Initialize classifier.

        Parameters:
        -----------
        n_radial_points : int
            Number of points to sample along diameter
        smoothing_window : int
            Savitzky-Golay filter window (must be odd)
        peak_prominence : float
            Minimum prominence for peak detection (0-1 scale)
        center_falloff_sigma : float
            Gaussian falloff parameter (relative to radius) - DEPRECATED, use center_window_fraction
            Smaller = sharper dropoff away from center
        skyrmionium_threshold : float
            Classification threshold (0-1 normalized score)
        center_window_fraction : float (NEW)
            Fraction of radius defining 'center region' for skyrmionium detection
            Default: 0.3 (30% of radius). Peaks outside this window get low weight.
            Scales with skyrmion size to prevent edge-peak contamination in small skyrmions.
        top_n_angles : int (NEW)
            Number of best diameter angles to use (1-4). Default: 4 (use all)
            Set to 2 to use only the 2 best angles, reducing noise sensitivity.
        merge_threshold : float (NEW)
            Fraction of radius for merging nearby peaks (prevents split-peak inflation)
            Default: 0.15. Peaks closer than this are merged into one wide peak.
        width_bonus_enabled : bool (NEW)
            If True, give bonus score to wide peaks (compensates for size-dependent widths)
            Default: True
        window_falloff_sharpness : float (NEW)
            Sharpness of exponential falloff outside center window (default: 0.1)
            Lower = sharper rejection of outside peaks. Range: 0.05-0.5
            0.05 = very sharp (almost hard cutoff)
            0.1 = sharp (recommended, FIXED from 0.5)
            0.5 = gentle (original buggy value)
        use_nonlinear_window : bool (NEW - CRITICAL FIX)
            Use logarithmic window scaling instead of linear (default: True)
            Fixes size bias: small skyrmions get tighter windows, large get looser
        window_scale_a : float (NEW)
            Intercept for logarithmic window scaling (default: -0.27)
            window_fraction = max(0.08, a + b×log(radius))
        window_scale_b : float (NEW)
            Slope for logarithmic window scaling (default: 0.13)
            Controls how quickly window grows with size
        angle_selection_mode : str (NEW)
            Mode for angle selection:
            - 'top_n': Use top N highest scoring angles (default)
            - 'manual': Use specific angles selected by user
            - 'middle_two': Use only the 2nd and 3rd highest scoring angles (excludes outliers)
        selected_angles : list of int, optional (NEW)
            For 'manual' mode: list of angle indices to use [0, 1, 2, 3]
            (0=0°, 1=45°, 2=90°, 3=135°). Default: None
        """
        self.n_radial_points = n_radial_points
        self.smoothing_window = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1
        self.peak_prominence = peak_prominence
        self.center_falloff_sigma = center_falloff_sigma  # Keep for backward compatibility
        self.skyrmionium_threshold = skyrmionium_threshold

        # NEW parameters for improved classification
        self.center_window_fraction = center_window_fraction
        self.top_n_angles = top_n_angles
        self.merge_threshold = merge_threshold
        self.width_bonus_enabled = width_bonus_enabled
        self.window_falloff_sharpness = window_falloff_sharpness

        # Non-linear window scaling (fixes size bias)
        self.use_nonlinear_window = use_nonlinear_window
        self.window_scale_a = window_scale_a
        self.window_scale_b = window_scale_b

        # Advanced angle selection options
        self.angle_selection_mode = angle_selection_mode
        self.selected_angles = selected_angles if selected_angles is not None else [0, 1, 2, 3]


    def extract_diameter_profiles(self, image, center, radius):
        """
        Extract intensity profiles along 4 diameters (edge-to-edge).

        This approach avoids averaging out peaks that occur due to the
        alternating bright/dark contrast in skyrmions.

        Parameters:
        -----------
        image : ndarray (H, W)
            Grayscale LTEM image
        center : tuple
            (y, x) center coordinates
        radius : float
            Maximum radius to sample

        Returns:
        --------
        profiles : list of ndarray
            4 diameter profiles (0°, 45°, 90°, 135°)
        """
        cy, cx = center

        # 4 diameter angles: 0°, 45°, 90°, 135°
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        profiles = []
        for angle in angles:
            # Sample from -radius to +radius along diameter
            distances = np.linspace(-radius * 1.5, radius * 1.5, self.n_radial_points)

            # Coordinates along this diameter line
            y_coords = cy + distances * np.sin(angle)
            x_coords = cx + distances * np.cos(angle)

            # Bilinear interpolation for sub-pixel sampling
            coords = np.array([y_coords, x_coords])
            profile = map_coordinates(image, coords, order=1, mode='constant', cval=0)
            profiles.append(profile)

        return profiles


    def smooth_profile(self, profile):
        """Apply Savitzky-Golay smoothing to reduce noise."""
        if len(profile) < self.smoothing_window:
            return profile
        return savgol_filter(profile, self.smoothing_window, polyorder=3)


    def compute_proximity_weight(self, distance_from_center, radius):
        """
        Compute Gaussian proximity weight based on distance from center.

        DEPRECATED: Use compute_center_window_weight() for size-scaled center detection.

        Peaks closer to center get higher weight (skyrmionium ring structure).

        Parameters:
        -----------
        distance_from_center : float
            Absolute distance from diameter center
        radius : float
            Skyrmion radius (for normalization)

        Returns:
        --------
        weight : float (0-1)
        """
        # Normalize distance by radius
        normalized_distance = distance_from_center / radius

        # Gaussian falloff: exp(-(d^2) / (2*sigma^2))
        weight = np.exp(-(normalized_distance ** 2) / (2 * self.center_falloff_sigma ** 2))

        return weight


    def get_window_fraction_for_radius(self, radius):
        """
        Compute center window fraction for given skyrmion radius.

        CRITICAL FIX: Uses logarithmic scaling to fix size bias.
        Small skyrmions get tighter windows, large skyrmions get looser windows.

        Parameters:
        -----------
        radius : float
            Skyrmion radius in pixels

        Returns:
        --------
        window_fraction : float
            Fraction of radius to use as center window
        """
        if self.use_nonlinear_window:
            # Logarithmic scaling: window_fraction = a + b×log(radius)
            # Calibrated for: small (20px) → 0.12×R, large (80px) → 0.30×R
            window_fraction = self.window_scale_a + self.window_scale_b * np.log(radius)

            # Clip to reasonable range (prevent negative or extreme values)
            window_fraction = np.clip(window_fraction, 0.08, 0.5)
        else:
            # Linear scaling (backward compatibility)
            window_fraction = self.center_window_fraction

        return window_fraction


    def compute_center_window_weight(self, distance_from_center, radius):
        """
        Compute weight using size-scaled center window.

        IMPROVED: Now supports logarithmic scaling to fix size bias.
        This prevents edge-peak contamination in small skyrmions and handles
        off-center skyrmionium disks in large skyrmions.

        Parameters:
        -----------
        distance_from_center : float
            Absolute distance from diameter center (pixels)
        radius : float
            Skyrmion radius (pixels)

        Returns:
        --------
        weight : float (0-1)
        """
        # Compute window fraction (may be size-dependent)
        window_fraction = self.get_window_fraction_for_radius(radius)

        # Compute center window size in pixels
        center_window_size = window_fraction * radius

        # Normalize distance
        normalized_distance = distance_from_center / center_window_size

        if normalized_distance <= 1.0:
            # Inside center window: full weight
            weight = 1.0
        else:
            # Outside center window: Sharp falloff (configurable)
            # Lower sharpness = steeper falloff = stronger rejection
            # Default 0.1 provides sharp cutoff (FIXED from original buggy 0.5)
            weight = np.exp(-((normalized_distance - 1.0) ** 2) / self.window_falloff_sharpness)

        return weight


    def compute_width_factor(self, peak_width, radius, distance_from_center):
        """
        Compute width bonus factor for peaks (POSITION-AWARE).

        CRITICAL FIX: Width preference depends on peak location:
        - Central peaks (skyrmionium detection): NARROW is good (sharp disk)
        - Edge peaks (natural variation): WIDE is good (size scaling)

        Parameters:
        -----------
        peak_width : float
            Width of peak in pixels (from scipy.signal.find_peaks)
        radius : float
            Skyrmion radius (pixels)
        distance_from_center : float
            Distance of peak from center (pixels)

        Returns:
        --------
        width_factor : float (0.5-1.5)
            Multiplier for peak score
        """
        if not self.width_bonus_enabled:
            return 1.0

        # Expected peak width scales with skyrmion size
        expected_width_pixels = 0.2 * radius

        # Normalize actual width
        normalized_width = peak_width / expected_width_pixels

        # Determine if peak is in center region
        center_window_size = self.center_window_fraction * radius
        normalized_distance = distance_from_center / center_window_size

        if normalized_distance <= 1.0:
            # CENTRAL PEAK: Narrow is good (skyrmionium disk)
            # Invert the width preference!
            # Sharp peak (width=0.3×expected): factor = 1.35
            # Normal peak (width=1.0×expected): factor = 1.0
            # Wide peak (width=2.0×expected): factor = 0.75 (penalty!)
            width_factor = 1.5 - 0.5 * normalized_width
        else:
            # EDGE PEAK: Wide is good (natural size variation)
            # Original logic applies
            # Sharp peak (width=0.3×expected): factor = 0.65
            # Normal peak (width=1.0×expected): factor = 1.0
            # Wide peak (width=2.0×expected): factor = 1.25 (bonus)
            width_factor = 0.5 + 0.5 * normalized_width

        # Cap factor to prevent extreme values
        width_factor = np.clip(width_factor, 0.5, 1.5)

        return width_factor


    def merge_nearby_peaks(self, peaks, properties, distances, radius):
        """
        Merge peaks that are close together (likely wide peak split by noise) (NEW).

        Parameters:
        -----------
        peaks : ndarray
            Peak indices from scipy.signal.find_peaks
        properties : dict
            Peak properties (prominences, widths, etc.)
        distances : ndarray
            Distance values along profile
        radius : float
            Skyrmion radius

        Returns:
        --------
        merged_peaks : list of dict
            Each dict contains: {index, prominence, width, position, merged_from}
        """
        if len(peaks) == 0:
            return []

        # Convert to list of peak dictionaries
        peak_list = []
        for i, peak_idx in enumerate(peaks):
            peak_list.append({
                'index': peak_idx,
                'position': distances[peak_idx],
                'prominence': properties['prominences'][i],
                'width': properties['widths'][i] if 'widths' in properties else 5.0,
                'merged_from': [peak_idx]  # Track merges
            })

        # Merge threshold in pixels
        merge_distance = self.merge_threshold * radius

        # Iteratively merge closest peaks
        merged = True
        while merged:
            merged = False

            for i in range(len(peak_list) - 1):
                if i >= len(peak_list) - 1:
                    break

                peak_a = peak_list[i]
                peak_b = peak_list[i + 1]

                distance = abs(peak_a['position'] - peak_b['position'])

                if distance < merge_distance:
                    # Merge peak_b into peak_a
                    total_prominence = peak_a['prominence'] + peak_b['prominence']

                    # Weighted average position
                    merged_position = (
                        peak_a['position'] * peak_a['prominence'] +
                        peak_b['position'] * peak_b['prominence']
                    ) / total_prominence

                    # Combined width
                    merged_width = peak_a['width'] + peak_b['width']

                    # Use max prominence (dominant peak)
                    merged_prominence = max(peak_a['prominence'], peak_b['prominence'])

                    # Create merged peak
                    peak_list[i] = {
                        'index': peak_a['index'],  # Keep first index
                        'position': merged_position,
                        'prominence': merged_prominence,
                        'width': merged_width,
                        'merged_from': peak_a['merged_from'] + peak_b['merged_from']
                    }

                    # Remove peak_b
                    peak_list.pop(i + 1)
                    merged = True
                    break

        return peak_list


    def validate_profile_quality(self, profile):
        """
        Validate that profile has sufficient contrast and isn't just noise.

        Returns:
        --------
        is_valid : bool
            True if profile has meaningful structure
        quality_score : float
            Quality metric (0-1)
        """
        # Compute contrast (range)
        contrast = profile.max() - profile.min()

        # Compute signal-to-noise ratio estimate
        # SNR = range / std_dev of high-frequency component
        if len(profile) > 10:
            # High-pass filter to isolate noise
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(profile, sigma=2)
            noise = profile - smoothed
            noise_std = np.std(noise)

            if noise_std > 1e-6:
                snr = contrast / noise_std
            else:
                snr = 100.0  # Very clean signal
        else:
            snr = 10.0

        # Profile is valid if it has sufficient contrast and SNR
        is_valid = contrast > 10.0 and snr > 2.0
        quality_score = np.clip(snr / 10.0, 0, 1)

        return is_valid, quality_score


    def score_diameter_profile(self, profile, radius):
        """
        Score a single diameter profile for skyrmionium characteristics.

        Higher score = more likely skyrmionium
        Score = peak_height × proximity_weight(distance_from_center)

        IMPROVED: Now validates profile quality and filters weak peaks

        Parameters:
        -----------
        profile : ndarray
            Diameter intensity profile (edge-to-edge)
        radius : float
            Skyrmion radius

        Returns:
        --------
        score : float
            Skyrmionium score for this diameter
        details : dict
            Scoring details (peaks, weights, etc.)
        """
        # Validate profile quality first
        is_valid, quality_score = self.validate_profile_quality(profile)

        if not is_valid:
            return 0.0, {
                'peaks': [],
                'weighted_scores': [],
                'quality_score': quality_score,
                'valid': False
            }

        smoothed = self.smooth_profile(profile)

        # Normalize to [0, 1]
        if smoothed.max() - smoothed.min() < 1e-6:
            return 0.0, {
                'peaks': [],
                'weighted_scores': [],
                'quality_score': 0.0,
                'valid': False
            }

        normalized = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())

        # Find all peaks with width information
        peaks, properties = find_peaks(
            normalized,
            prominence=self.peak_prominence,
            width=2,  # Min width
            distance=5  # Peaks must be at least 5 points apart
        )

        if len(peaks) == 0:
            return 0.0, {
                'peaks': [],
                'merged_peaks': [],
                'weighted_scores': [],
                'quality_score': quality_score,
                'valid': True
            }

        # Compute distances from center
        center_idx = len(normalized) // 2
        distances = np.linspace(-radius * 1.5, radius * 1.5, len(normalized))

        # STEP 1: Merge nearby peaks (prevents split-peak inflation)
        merged_peaks = self.merge_nearby_peaks(peaks, properties, distances, radius)

        # STEP 2: Score each merged peak with NEW logic
        peak_scores = []
        peak_details = []

        for merged_peak in merged_peaks:
            # Peak characteristics
            peak_prominence = merged_peak['prominence']
            peak_width = merged_peak['width']
            peak_position = merged_peak['position']
            distance_from_center = abs(peak_position)

            # NEW: Size-scaled center window weight (replaces Gaussian)
            position_weight = self.compute_center_window_weight(distance_from_center, radius)

            # NEW: Width-aware scoring (POSITION-DEPENDENT - CRITICAL FIX!)
            # Central peaks: narrow=good, wide=bad (skyrmionium has sharp disk)
            # Edge peaks: wide=good (natural size variation)
            width_factor = self.compute_width_factor(peak_width, radius, distance_from_center)

            # Combined score: prominence × width_factor × position_weight
            weighted_score = peak_prominence * width_factor * position_weight

            # Filter: only include peaks with meaningful contribution
            min_weighted_score = 0.05  # Must contribute at least 5%

            if weighted_score >= min_weighted_score:
                peak_scores.append(weighted_score)

                peak_details.append({
                    'position': peak_position,
                    'prominence': peak_prominence,
                    'width': peak_width,
                    'width_factor': width_factor,
                    'position_weight': position_weight,
                    'weighted_score': weighted_score,
                    'merged_from': merged_peak['merged_from'],
                    'was_merged': len(merged_peak['merged_from']) > 1
                })

        # FIXED: Use MAX peak score instead of SUM (prevents noise accumulation)
        # Physical reasoning: Skyrmioniums have ONE central peak, not many oscillations.
        # Multiple peaks indicate noise, which should NOT increase the score.
        if len(peak_scores) == 0:
            total_score = 0.0
        elif len(peak_scores) == 1:
            # Single peak: use it directly
            total_score = peak_scores[0]
        else:
            # Multiple peaks: use MAXIMUM (the strongest evidence)
            # Alternative: could use mean of top-2, but max is more robust
            total_score = np.max(peak_scores)

        # Apply quality scaling
        total_score *= quality_score

        return total_score, {
            'peaks': peak_details,
            'merged_peaks': merged_peaks,
            'n_original_peaks': len(peaks),
            'n_merged_peaks': len(merged_peaks),
            'n_accepted_peaks': len(peak_scores),
            'total': total_score,
            'quality_score': quality_score,
            'valid': True,
            'scoring_method': 'max'  # Track which method was used
        }




    def classify(self, image, center, radius):
        """
        Classify single skyrmion using peak scoring on diameter profiles.

        Score calculation:
        1. Extract 4 diameter profiles (0°, 45°, 90°, 135°)
        2. For each profile, score peaks: peak_height × proximity_weight(distance_from_center)
        3. Sum scores across all 4 diameters
        4. Average across diameters for final score (more interpretable than sum)
        5. Compare to threshold

        IMPROVED: Better normalization and clearer score interpretation

        Parameters:
        -----------
        image : ndarray (H, W)
            Grayscale LTEM image
        center : tuple
            (y, x) center of skyrmion
        radius : float
            Equivalent radius of skyrmion

        Returns:
        --------
        classification : str
            'skyrmion' or 'skyrmionium'
        confidence : float
            Mean skyrmionium score across diameters (0-1)
        details : dict
            Scoring details for each diameter
        """
        # Extract 4 diameter profiles (0°, 45°, 90°, 135°)
        profiles = self.extract_diameter_profiles(image, center, radius)

        # Score each diameter
        diameter_scores = []
        diameter_details = []

        for i, profile in enumerate(profiles):
            score, details = self.score_diameter_profile(profile, radius)
            diameter_scores.append(score)
            diameter_details.append({
                'angle': i * 45,  # 0°, 45°, 90°, 135°
                'score': score,
                'details': details,
                'selected': False  # Will be set to True if used in top-N
            })

        # NEW: Flexible angle selection with multiple modes
        sorted_indices = np.argsort(diameter_scores)[::-1]  # Descending order

        if self.angle_selection_mode == 'top_n':
            # Use top N highest scoring angles
            if self.top_n_angles < len(diameter_scores):
                selected_indices = sorted_indices[:self.top_n_angles]
            else:
                selected_indices = list(range(len(diameter_scores)))

        elif self.angle_selection_mode == 'manual':
            # Use specific angles selected by user
            selected_indices = [i for i in self.selected_angles if i < len(diameter_scores)]

        elif self.angle_selection_mode == 'middle_two':
            # Use only the 2nd and 3rd highest scoring angles (exclude highest and lowest)
            if len(diameter_scores) >= 4:
                selected_indices = sorted_indices[1:3]  # Indices 1 and 2 (2nd and 3rd highest)
            elif len(diameter_scores) == 3:
                selected_indices = sorted_indices[1:2]  # Just the middle one
            else:
                # Fallback: use all available if < 3 angles
                selected_indices = list(range(len(diameter_scores)))

        else:
            # Fallback: use all angles
            selected_indices = list(range(len(diameter_scores)))

        # Mark which angles were selected
        for idx in selected_indices:
            diameter_details[idx]['selected'] = True

        # Use only selected scores
        selected_scores = [diameter_scores[i] for i in selected_indices]
        n_angles_used = len(selected_indices)

        # Total score (sum across selected diameters)
        total_score_sum = np.sum(selected_scores)

        # IMPROVED NORMALIZATION:
        # Use MEAN of selected angles instead of arbitrary division by 2.0
        # This makes the score more interpretable: it's the average skyrmionium
        # evidence across the BEST N diameter directions (reduces noise)
        mean_score = total_score_sum / n_angles_used if n_angles_used > 0 else 0.0

        # Additional robustness: clip to reasonable range
        # Theoretical max per diameter ≈ 1.0 (one perfect central peak with weight=1.0)
        # But in practice, max is typically ~0.5-0.7
        # We'll use adaptive normalization: divide by max(selected_scores) if > 0.5
        max_diameter_score = max(selected_scores) if len(selected_scores) > 0 else 1.0

        if max_diameter_score > 0.5:
            # Normalize by observed maximum to spread out the score range
            normalized_score = np.clip(mean_score / max_diameter_score, 0, 1)
        else:
            # Use mean directly if all scores are low
            normalized_score = np.clip(mean_score, 0, 1)

        # Classification based on threshold
        is_skyrmionium = normalized_score >= self.skyrmionium_threshold

        # Classification result
        classification = 'skyrmionium' if is_skyrmionium else 'skyrmion'

        # Detailed results
        details = {
            'diameter_scores': diameter_details,
            'individual_scores': diameter_scores,  # All 4 scores
            'selected_scores': selected_scores,  # Only top-N scores
            'top_n_angles': self.top_n_angles,
            'n_angles_used': n_angles_used,
            'total_score_sum': total_score_sum,
            'mean_score': mean_score,
            'normalized_score': normalized_score,
            'max_diameter_score': max_diameter_score,
            'threshold': self.skyrmionium_threshold,
            'profiles': profiles,
            # NEW parameters used
            'center_window_fraction': self.center_window_fraction,
            'merge_threshold': self.merge_threshold,
            'width_bonus_enabled': self.width_bonus_enabled
        }

        return classification, normalized_score, details


    def classify_batch(self, image, labels):
        """
        Classify all skyrmions in a labeled image.

        Parameters:
        -----------
        image : ndarray (H, W)
            Grayscale LTEM image
        labels : ndarray (H, W)
            Label mask from StarDist (each skyrmion has unique ID)

        Returns:
        --------
        results : list of dict
            Classification results for each skyrmion
        """
        props = regionprops(labels)
        results = []

        for prop in props:
            # Centroid and radius
            cy, cx = prop.centroid
            radius = np.sqrt(prop.area / np.pi)

            # Classify
            classification, confidence, details = self.classify(image, (cy, cx), radius)

            # Store results
            result = {
                'label_id': prop.label,
                'centroid_y': cy,
                'centroid_x': cx,
                'radius': radius,
                'area': prop.area,
                'classification': classification,
                'confidence': confidence,
                'topological_charge': 0 if classification == 'skyrmionium' else 1,  # Q=0 or Q=±1
                'details': details
            }
            results.append(result)

        return results


def visualize_classification(image, labels, classifications, output_path=None):
    """
    Visualize classification results with color-coded overlays.

    Parameters:
    -----------
    image : ndarray (H, W)
        LTEM image
    labels : ndarray (H, W)
        StarDist label mask
    classifications : list of dict
        Results from classify_batch()
    output_path : str or Path, optional
        Save visualization to file

    Returns:
    --------
    fig : matplotlib figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('LTEM Image')
    axes[0].axis('off')

    # 2. All detections
    axes[1].imshow(image, cmap='gray')
    for result in classifications:
        cy, cx = result['centroid_y'], result['centroid_x']
        radius = result['radius']
        circle = Circle((cx, cy), radius, fill=False, edgecolor='cyan', linewidth=2)
        axes[1].add_patch(circle)
    axes[1].set_title(f'All Detections ({len(classifications)} total)')
    axes[1].axis('off')

    # 3. Classification overlay
    axes[2].imshow(image, cmap='gray')
    n_skyrmions = 0
    n_skyrmioniums = 0

    for result in classifications:
        cy, cx = result['centroid_y'], result['centroid_x']
        radius = result['radius']

        if result['classification'] == 'skyrmion':
            color = 'blue'
            n_skyrmions += 1
        else:
            color = 'red'
            n_skyrmioniums += 1

        circle = Circle((cx, cy), radius, fill=False, edgecolor=color, linewidth=2, alpha=0.8)
        axes[2].add_patch(circle)

    axes[2].set_title(f'Classification: {n_skyrmions} Skyrmions (blue), {n_skyrmioniums} Skyrmioniums (red)')
    axes[2].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")

    return fig


if __name__ == "__main__":
    # Example usage and testing
    print("Topological Classifier - Example Usage")
    print("=" * 70)
    print("This module classifies skyrmions vs skyrmioniums using radial")
    print("intensity profile analysis with ensemble voting.")
    print("\nUsage:")
    print("  from topological_classifier_public import TopologicalClassifier")
    print("  classifier = TopologicalClassifier()")
    print("  classification, confidence, details = classifier.classify(image, center, radius)")
    print("  # or for batch:")
    print("  results = classifier.classify_batch(image, label_mask)")
