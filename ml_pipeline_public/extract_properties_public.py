#!/usr/bin/env python3
"""
Complete property extraction pipeline for skyrmion analysis.

Pipeline:
1. Load LTEM image
2. Segment skyrmions using trained StarDist model
3. Extract morphological properties
4. Classify topological charge (skyrmion vs skyrmionium)
5. Output comprehensive property table

Author: AI Assistant + Scott Kiehn
Date: 2025-11-11
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from skimage.measure import regionprops
from stardist.models import StarDist2D
from topological_classifier_public import TopologicalClassifier, visualize_classification

# Import CNN classifier (optional - only if trained model exists)
try:
    from cnn_inference import CNNClassifier
    CNN_AVAILABLE = True
except (ImportError, FileNotFoundError):
    CNN_AVAILABLE = False


class SkyrmionPropertyExtractor:
    """
    Complete pipeline for skyrmion detection and property extraction.
    """

    def __init__(self, model_path='ml_pipeline_public/models', model_name=None,
                 classifier_type='algorithmic',
                 peak_prominence=0.15, center_falloff_sigma=0.2, skyrmionium_threshold=0.3,
                 center_window_fraction=0.3, top_n_angles=4, merge_threshold=0.15,
                 width_bonus_enabled=True, use_nonlinear_window=True,
                 window_scale_a=-0.27, window_scale_b=0.13,
                 angle_selection_mode='top_n', selected_angles=None):
        """
        Initialize extractor with trained StarDist model.

        Parameters:
        -----------
        model_path : str or Path
            Directory containing trained StarDist models
        model_name : str, optional
            Specific StarDist model name. If None, uses latest model.
        classifier_type : str
            Type of classifier to use: 'algorithmic' or 'cnn' (default: 'algorithmic')
        peak_prominence : float
            Minimum peak prominence for detection (0-1) [algorithmic only]
        center_falloff_sigma : float
            Gaussian falloff parameter (relative to radius) - DEPRECATED
        skyrmionium_threshold : float
            Classification threshold (0-1 normalized score)
        center_window_fraction : float (NEW)
            Fraction of radius defining 'center region' (default: 0.3)
        top_n_angles : int (NEW)
            Number of best angles to use (1-4, default: 4)
        merge_threshold : float (NEW)
            Fraction of radius for merging peaks (default: 0.15)
        width_bonus_enabled : bool (NEW)
            Enable width-aware scoring (default: True)
        use_nonlinear_window : bool (NEW)
            Enable logarithmic window scaling (default: True)
        window_scale_a : float (NEW)
            Logarithmic scaling intercept (default: -0.27)
        window_scale_b : float (NEW)
            Logarithmic scaling slope (default: 0.13)
        angle_selection_mode : str (NEW)
            'top_n', 'manual', or 'middle_two'
        selected_angles : list of int (NEW)
            For manual mode: which angles to use [0, 1, 2, 3]
        """
        model_path = Path(model_path)

        # Auto-detect latest model if not specified
        if model_name is None:
            model_dirs = sorted([d for d in model_path.glob('skyrmion_detector_*') if d.is_dir()])
            if not model_dirs:
                raise FileNotFoundError(f"No trained models found in {model_path}")
            model_name = model_dirs[-1].name
            print(f"Using latest model: {model_name}")

        # Load StarDist model
        self.model = StarDist2D(None, name=model_name, basedir=str(model_path))
        print(f"✓ Loaded StarDist model: {model_name}")
        print(f"  Thresholds: prob={self.model.thresholds.prob:.3f}, nms={self.model.thresholds.nms:.3f}")

        # Validate and initialize classifier
        self.classifier_type = classifier_type.lower()

        if self.classifier_type == 'cnn':
            if not CNN_AVAILABLE:
                raise ValueError(
                    "CNN classifier requested but not available. "
                    "Please train a CNN model first using: "
                    "python ml_pipeline_public/train_cnn_classifier.py"
                )
            self.classifier = CNNClassifier()
            print(f"✓ Initialized CNN classifier")

        elif self.classifier_type == 'algorithmic':
            # Initialize topological classifier with ALL new parameters
            self.classifier = TopologicalClassifier(
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
            print(f"✓ Initialized algorithmic classifier (size-normalized peak scoring)")
            window_mode = "logarithmic" if use_nonlinear_window else f"linear ({center_window_fraction:.2f}×R)"
            print(f"  Parameters: prominence={peak_prominence:.3f}, window={window_mode}, threshold={skyrmionium_threshold:.3f}")
            print(f"  Top-N angles={top_n_angles}, merge={merge_threshold:.2f}×R, width_bonus={width_bonus_enabled}")
            if use_nonlinear_window:
                print(f"  Logarithmic scaling: {window_scale_a:.3f} + {window_scale_b:.3f}×log(R)")
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}. Must be 'algorithmic' or 'cnn'")


    def load_image(self, image_path):
        """
        Load LTEM image and normalize for StarDist.

        Parameters:
        -----------
        image_path : str or Path
            Path to LTEM image (JPG, PNG, TIFF)

        Returns:
        --------
        image : ndarray (H, W)
            Normalized grayscale image
        image_raw : ndarray (H, W)
            Original image for visualization
        """
        image_path = Path(image_path)

        # Load image
        img = Image.open(image_path)
        if img.mode == 'RGB':
            img = img.convert('L')  # Convert to grayscale

        image_raw = np.array(img, dtype=np.float32)

        # Normalize using percentile clipping (same as training)
        vmin, vmax = np.percentile(image_raw, (1, 99.8))
        image_normalized = np.clip((image_raw - vmin) / (vmax - vmin), 0, 1).astype(np.float32)

        return image_normalized, image_raw


    def segment(self, image):
        """
        Segment skyrmions using StarDist.

        Parameters:
        -----------
        image : ndarray (H, W)
            Normalized grayscale image

        Returns:
        --------
        labels : ndarray (H, W)
            Label mask (each skyrmion has unique integer ID)
        details : dict
            Prediction details from StarDist
        """
        # Add channel dimension for StarDist
        if image.ndim == 2:
            image = image[..., np.newaxis]

        # Predict
        labels, details = self.model.predict_instances(
            image,
            n_tiles=self.model._guess_n_tiles(image),
            show_tile_progress=False
        )

        n_detected = len(np.unique(labels)) - 1  # Exclude background
        print(f"✓ Detected {n_detected} skyrmions")

        return labels, details


    def extract_morphological_properties(self, image, labels, pixel_scale=None):
        """
        Extract morphological properties for each detected skyrmion.

        Parameters:
        -----------
        image : ndarray (H, W)
            Original grayscale image
        labels : ndarray (H, W)
            Label mask from StarDist
        pixel_scale : float, optional
            Physical scale (nm/pixel). If None, uses pixel units.

        Returns:
        --------
        properties : pandas DataFrame
            Morphological properties for each skyrmion
        """
        props = regionprops(labels, intensity_image=image)

        data = []
        for prop in props:
            # Basic measurements
            area_px = prop.area
            diameter_px = 2 * np.sqrt(area_px / np.pi)  # Equivalent circle diameter
            perimeter_px = prop.perimeter
            circularity = 4 * np.pi * area_px / (perimeter_px ** 2) if perimeter_px > 0 else 0

            # Position
            cy, cx = prop.centroid
            min_row, min_col, max_row, max_col = prop.bbox

            # Intensity statistics
            mean_intensity = prop.mean_intensity
            min_intensity = prop.min_intensity
            max_intensity = prop.max_intensity

            # Physical units conversion
            if pixel_scale is not None:
                area_nm2 = area_px * (pixel_scale ** 2)
                diameter_nm = diameter_px * pixel_scale
                perimeter_nm = perimeter_px * pixel_scale
                unit_label = 'nm'
            else:
                area_nm2 = area_px
                diameter_nm = diameter_px
                perimeter_nm = perimeter_px
                unit_label = 'px'

            data.append({
                'label_id': prop.label,
                'centroid_x': cx,
                'centroid_y': cy,
                f'area_{unit_label}': area_nm2 if pixel_scale else area_px,
                f'diameter_{unit_label}': diameter_nm if pixel_scale else diameter_px,
                f'perimeter_{unit_label}': perimeter_nm if pixel_scale else perimeter_px,
                'circularity': circularity,
                'eccentricity': prop.eccentricity,
                'mean_intensity': mean_intensity,
                'min_intensity': min_intensity,
                'max_intensity': max_intensity,
                'bbox_min_row': min_row,
                'bbox_min_col': min_col,
                'bbox_max_row': max_row,
                'bbox_max_col': max_col,
            })

        df = pd.DataFrame(data)
        return df


    def classify_topological_charge(self, image, labels):
        """
        Classify topological charge for each skyrmion.

        Parameters:
        -----------
        image : ndarray (H, W)
            Grayscale LTEM image
        labels : ndarray (H, W)
            Label mask from StarDist

        Returns:
        --------
        classifications : pandas DataFrame
            Topological classifications for each skyrmion
        """
        results = self.classifier.classify_batch(image, labels)

        # Convert to DataFrame
        data = []
        for result in results:
            # Extract individual diameter scores
            diameter_scores = [d['score'] for d in result['details']['diameter_scores']]

            # Get score information (handle both old and new key names for compatibility)
            details = result['details']
            total_score_sum = details.get('total_score_sum', details.get('total_score', 0))
            mean_score = details.get('mean_score', total_score_sum / 4.0 if total_score_sum > 0 else 0)

            data.append({
                'label_id': result['label_id'],
                'topological_charge': result['topological_charge'],
                'classification': result['classification'],
                'confidence': result['confidence'],  # This is the normalized_score (0-1)
                'total_score_sum': total_score_sum,
                'mean_score': mean_score,
                'score_0deg': diameter_scores[0] if len(diameter_scores) > 0 else 0,
                'score_45deg': diameter_scores[1] if len(diameter_scores) > 1 else 0,
                'score_90deg': diameter_scores[2] if len(diameter_scores) > 2 else 0,
                'score_135deg': diameter_scores[3] if len(diameter_scores) > 3 else 0,
            })

        df = pd.DataFrame(data)
        return df


    def process_image(self, image_path, pixel_scale=None, save_visualization=True, output_dir=None):
        """
        Complete processing pipeline for single LTEM image.

        Parameters:
        -----------
        image_path : str or Path
            Path to LTEM image
        pixel_scale : float, optional
            Physical scale (nm/pixel)
        save_visualization : bool
            If True, save classification visualization
        output_dir : str or Path, optional
            Directory to save outputs. If None, creates ml_pipeline_public/results/

        Returns:
        --------
        properties : pandas DataFrame
            Complete property table with morphology + topological charge
        labels : ndarray
            Label mask for visualization
        """
        image_path = Path(image_path)
        print("\n" + "=" * 70)
        print(f"PROCESSING: {image_path.name}")
        print("=" * 70)

        # 1. Load image
        print("\n1. Loading image...")
        image_normalized, image_raw = self.load_image(image_path)
        print(f"   Image shape: {image_raw.shape}")

        # 2. Segment skyrmions
        print("\n2. Segmenting skyrmions...")
        labels, details = self.segment(image_normalized)

        if len(np.unique(labels)) == 1:
            print("   ⚠ No skyrmions detected!")
            return pd.DataFrame(), labels

        # 3. Extract morphological properties
        print("\n3. Extracting morphological properties...")
        morph_props = self.extract_morphological_properties(image_raw, labels, pixel_scale)
        print(f"   Extracted {len(morph_props)} measurements")

        # 4. Classify topological charge
        print("\n4. Classifying topological charge...")
        topo_props = self.classify_topological_charge(image_raw, labels)

        n_skyrmions = (topo_props['classification'] == 'skyrmion').sum()
        n_skyrmioniums = (topo_props['classification'] == 'skyrmionium').sum()
        print(f"   Skyrmions (Q=±1): {n_skyrmions}")
        print(f"   Skyrmioniums (Q=0): {n_skyrmioniums}")

        # 5. Merge properties
        properties = morph_props.merge(topo_props, on='label_id')

        # 6. Save visualization
        if save_visualization:
            print("\n5. Generating visualization...")
            if output_dir is None:
                output_dir = Path('ml_pipeline_public/results') / image_path.stem
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Convert classifications to format expected by visualize_classification
            classifications = []
            for _, row in properties.iterrows():
                classifications.append({
                    'label_id': row['label_id'],
                    'centroid_y': row['centroid_y'],
                    'centroid_x': row['centroid_x'],
                    'radius': row.get('diameter_px', row.get('diameter_nm', 0)) / 2,
                    'classification': row['classification'],
                    'confidence': row['confidence']
                })

            viz_path = output_dir / f'{image_path.stem}_classification.png'
            visualize_classification(image_raw, labels, classifications, viz_path)

            # Save CSV
            csv_path = output_dir / f'{image_path.stem}_properties.csv'
            properties.to_csv(csv_path, index=False)
            print(f"   ✓ Saved properties to {csv_path}")

        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70)

        return properties, labels


    def process_batch(self, image_paths, pixel_scale=None, output_dir=None):
        """
        Process multiple images (e.g., temperature series).

        Parameters:
        -----------
        image_paths : list of str/Path
            List of LTEM image paths
        pixel_scale : float, optional
            Physical scale (nm/pixel)
        output_dir : str or Path, optional
            Directory to save outputs

        Returns:
        --------
        all_properties : pandas DataFrame
            Combined properties from all images with 'image_name' column
        """
        all_properties = []

        for i, image_path in enumerate(image_paths):
            print(f"\n{'='*70}")
            print(f"Processing image {i+1}/{len(image_paths)}: {Path(image_path).name}")
            print(f"{'='*70}")

            try:
                props, labels = self.process_image(image_path, pixel_scale, save_visualization=True, output_dir=output_dir)

                if len(props) > 0:
                    props['image_name'] = Path(image_path).name
                    props['image_index'] = i
                    all_properties.append(props)
            except Exception as e:
                print(f"   ✗ Error processing {Path(image_path).name}: {e}")
                continue

        if not all_properties:
            print("\n⚠ No skyrmions detected in any images!")
            return pd.DataFrame()

        # Combine all results
        combined = pd.concat(all_properties, ignore_index=True)

        # Save combined CSV
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / 'all_properties.csv'
            combined.to_csv(csv_path, index=False)
            print(f"\n✓ Saved combined properties to {csv_path}")

        return combined


def main():
    """Command-line interface for property extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract skyrmion properties from LTEM images using ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image (algorithmic classifier)
  python extract_properties.py --image "path/to/ltem_image.jpg"

  # Process with CNN classifier
  python extract_properties.py --image "image.jpg" --classifier cnn

  # Process with physical scale
  python extract_properties.py --image "image.jpg" --scale 2.5

  # Process multiple images (temperature series)
  python extract_properties.py --images image1.jpg image2.jpg image3.jpg

  # Specify custom StarDist model
  python extract_properties.py --image "image.jpg" --model skyrmion_detector_20251111_090556
        """
    )

    parser.add_argument('--image', type=str, help='Single LTEM image to process')
    parser.add_argument('--images', nargs='+', type=str, help='Multiple images to process (batch mode)')
    parser.add_argument('--scale', type=float, help='Pixel scale (nm/pixel)')
    parser.add_argument('--model', type=str, help='StarDist model name (default: latest)')
    parser.add_argument('--classifier', type=str, default='algorithmic',
                       choices=['algorithmic', 'cnn'],
                       help='Classifier type: algorithmic or cnn (default: algorithmic)')
    parser.add_argument('--output', type=str, default='ml_pipeline_public/results', help='Output directory')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')

    args = parser.parse_args()

    if not args.image and not args.images:
        parser.error("Must specify either --image or --images")

    # Initialize extractor
    extractor = SkyrmionPropertyExtractor(model_name=args.model, classifier_type=args.classifier)

    # Process
    if args.image:
        # Single image mode
        properties, labels = extractor.process_image(
            args.image,
            pixel_scale=args.scale,
            save_visualization=not args.no_viz,
            output_dir=args.output
        )

        print(f"\n✓ Processed {len(properties)} skyrmions")
        print("\nSummary:")
        print(properties[['classification', 'topological_charge']].value_counts())

    else:
        # Batch mode
        properties = extractor.process_batch(
            args.images,
            pixel_scale=args.scale,
            output_dir=args.output
        )

        print(f"\n✓ Processed {len(properties)} total skyrmions across {len(args.images)} images")
        print("\nSummary by classification:")
        print(properties.groupby('classification').size())


if __name__ == "__main__":
    main()
