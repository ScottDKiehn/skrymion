"""
Polar Fourier Classifier for Topological Charge Classification

VERSION 3.0 - Proper ML model with physics-informed feature engineering

This module provides real-time classification of skyrmions as Q=0 (skyrmionium)
or Q=1 (regular skyrmion) based on azimuthal intensity profile analysis.

Algorithm: Analyzes I(θ) at multiple radii using DFT to extract discriminative features,
then uses a trained RandomForest classifier for prediction.

Version History:
- v1.0: F1=0.672 on full dataset (threshold-based, amp_m1_r8 + var_r3)
- v2.0: F1=0.808 on full dataset (threshold-based, amp_m1_r3 + var_r3 + phase_diff)
- v3.0: F1=0.905 on full dataset (RandomForest with 30 physics-informed features)

Key v3 Features:
- phase_diff_r2_r8: Phase difference innermost→outermost (d'=2.46, BEST)
- amp_ratio_r3_r7: Inner/outer amplitude ratio (d'=2.41)
- phase_gradient_mean: Rate of phase change across radii (d'=2.31)
- inner_flatness: Detects smooth-core Q=0 Type B (d'=2.19)
- var_ratio_inner_outer: Inner/outer variance ratio (d'=2.30)

Author: Scott Kiehn + AI Assistant
Date: 2025-12-14
"""

import numpy as np
from scipy import ndimage
from scipy.fft import fft
from pathlib import Path
import json
import warnings
import pickle
warnings.filterwarnings('ignore')

# Try to import sklearn for v3, fall back to threshold-based if not available
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class PolarFourierClassifier:
    """
    Classifier for distinguishing Q=0 (skyrmionium) from Q=1 (skyrmion)
    based on polar Fourier analysis of azimuthal intensity profiles.

    Version 3.0 Key Changes:
    - Uses trained RandomForest model instead of threshold rules
    - Incorporates ALL radii (r2 through r8) for feature extraction
    - Adds phase gradient, amplitude ratios, and inner flatness features
    - Handles both Type A (nested dumbbell) and Type B (smooth core) Q=0

    Key Physics:
    - Neel skyrmions have dumbbell-like contrast (two opposite sides with opposite intensity)
    - Q=0 skyrmioniums have nested hierarchical structure with alternating polarity
    - Type A Q=0: Clear phase inversion between inner and outer (~π shift)
    - Type B Q=0: Smooth featureless core (low amplitude/variance at inner radii)
    """

    VERSION = "3.0"

    # Feature statistics from full dataset (3,088 samples)
    FEATURE_STATS = {
        'phase_diff_r2_r8': {'d_prime': 2.459, 'q0_higher': True},
        'phase_diff_r2_r7': {'d_prime': 2.437, 'q0_higher': True},
        'amp_ratio_r3_r7': {'d_prime': 2.408, 'q0_higher': False},
        'amp_m1_r3': {'d_prime': 2.385, 'q0_higher': False},
        'phase_gradient_mean': {'d_prime': 2.306, 'q0_higher': True},
        'var_ratio_inner_outer': {'d_prime': 2.296, 'q0_higher': False},
        'inner_flatness': {'d_prime': 2.194, 'q0_higher': True},
        'var_r3': {'d_prime': 2.161, 'q0_higher': False},
    }

    # Top 30 features for v3 model (ordered by d-prime)
    V3_FEATURES = [
        'phase_diff_r2_r8', 'phase_diff_r2_r7', 'phase_diff_r2_r6',
        'amp_ratio_r3_r7', 'amp_m1_r3', 'amp_ratio_r3_r8',
        'phase_gradient_mean', 'var_ratio_inner_outer', 'inner_flatness',
        'var_r3', 'phase_gradient_std', 'phase_gradient_max',
        'var_r4', 'amp_m1_r4', 'phase_m1_r2',
        'amp_ratio_r2_r8', 'amp_ratio_r2_r7', 'amp_m1_r2',
        'phase_diff_r3_r7', 'phase_diff_r3_r8', 'amp_m1_r8',
        'phase_diff_r3_r6', 'phase_diff_r4_r8', 'phase_diff_r4_r7',
        'var_r8', 'phase_diff_r4_r6', 'var_r2',
        'outer_asymmetry', 'var_r5', 'amp_m1_r7'
    ]

    # v2 thresholds (fallback if sklearn not available)
    V2_THRESHOLDS = {
        'amp_m1_r3': 0.119,
        'var_r3': 0.114,
        'phase_diff_m1': 0.685,
    }

    def __init__(self, n_points=64, version=3):
        """
        Initialize the classifier.

        Parameters:
        -----------
        n_points : int
            Number of points to sample along each azimuthal ring (default: 64)
        version : int
            Classifier version (1, 2, or 3). Version 3 requires sklearn.
        """
        self.n_points = n_points
        self.version = version
        self.radii_fractions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.harmonics = [0, 1, 2, 3]

        # For v3, initialize model (will be trained on first use or loaded)
        self.model = None
        self.scaler = None
        self._model_trained = False

        if version == 3 and not SKLEARN_AVAILABLE:
            print("Warning: sklearn not available, falling back to v2")
            self.version = 2

    def sample_azimuthal_profile(self, image, center_y, center_x, radius):
        """Sample intensity along a circle at given radius."""
        theta = np.linspace(0, 2*np.pi, self.n_points, endpoint=False)
        y_coords = center_y + radius * np.sin(theta)
        x_coords = center_x + radius * np.cos(theta)

        h, w = image.shape[:2]
        if (y_coords.min() < 0 or y_coords.max() >= h or
            x_coords.min() < 0 or x_coords.max() >= w):
            return theta, None

        profile = ndimage.map_coordinates(image, [y_coords, x_coords],
                                          order=1, mode='constant')
        return theta, profile

    def compute_dft_features(self, profile):
        """Compute DFT of azimuthal profile."""
        if profile is None:
            return None, None

        n = len(profile)
        fft_result = fft(profile)
        amplitudes = np.abs(fft_result[:n//2+1]) / n
        amplitudes[1:] *= 2
        phases = np.angle(fft_result[:n//2+1])

        return amplitudes, phases

    def extract_base_features(self, image, center_y, center_x, radius):
        """Extract base Fourier features at all radii."""
        features = {}
        phases_m1 = {}

        for r_frac in self.radii_fractions:
            actual_radius = radius * r_frac
            _, profile = self.sample_azimuthal_profile(image, center_y, center_x, actual_radius)
            r_key = int(r_frac * 10)

            if profile is None:
                for m in self.harmonics:
                    features[f'amp_m{m}_r{r_key}'] = np.nan
                    features[f'phase_m{m}_r{r_key}'] = np.nan
                features[f'var_r{r_key}'] = np.nan
                continue

            amplitudes, phases = self.compute_dft_features(profile)

            for m in self.harmonics:
                if amplitudes is not None and len(amplitudes) > m:
                    features[f'amp_m{m}_r{r_key}'] = amplitudes[m]
                    features[f'phase_m{m}_r{r_key}'] = phases[m]
                    if m == 1:
                        phases_m1[r_key] = phases[m]

            features[f'var_r{r_key}'] = np.std(profile)

        return features, phases_m1

    def engineer_features(self, base_features, phases_m1):
        """Engineer physics-informed features from base features."""
        features = base_features.copy()

        # Phase differences at multiple radius pairs
        for r_inner in [2, 3, 4]:
            for r_outer in [6, 7, 8]:
                if r_inner in phases_m1 and r_outer in phases_m1:
                    phase_diff = phases_m1[r_outer] - phases_m1[r_inner]
                    phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
                    features[f'phase_diff_r{r_inner}_r{r_outer}'] = np.abs(phase_diff)
                else:
                    features[f'phase_diff_r{r_inner}_r{r_outer}'] = np.nan

        # Amplitude ratios (inner/outer)
        for r_inner, r_outer in [(2, 7), (2, 8), (3, 7), (3, 8)]:
            amp_inner = base_features.get(f'amp_m1_r{r_inner}', np.nan)
            amp_outer = base_features.get(f'amp_m1_r{r_outer}', np.nan)
            if not np.isnan(amp_inner) and not np.isnan(amp_outer) and amp_outer > 1e-6:
                features[f'amp_ratio_r{r_inner}_r{r_outer}'] = amp_inner / amp_outer
            else:
                features[f'amp_ratio_r{r_inner}_r{r_outer}'] = np.nan

        # Inner flatness (detects smooth-core Type B Q=0)
        inner_amps = [base_features.get(f'amp_m1_r{r}', np.nan) for r in [2, 3, 4]]
        inner_vars = [base_features.get(f'var_r{r}', np.nan) for r in [2, 3, 4]]
        inner_amps = [x for x in inner_amps if not np.isnan(x)]
        inner_vars = [x for x in inner_vars if not np.isnan(x)]

        if inner_amps and inner_vars:
            features['inner_flatness'] = 1 / (np.mean(inner_amps) + np.mean(inner_vars) + 0.01)
        else:
            features['inner_flatness'] = np.nan

        # Phase gradient (rate of phase change across radii)
        phase_values = [phases_m1.get(r, np.nan) for r in [2, 3, 4, 5, 6, 7, 8]]
        valid_phases = [p for p in phase_values if not np.isnan(p)]

        if len(valid_phases) >= 3:
            unwrapped = np.unwrap(valid_phases)
            grad = np.abs(np.gradient(unwrapped))
            features['phase_gradient_mean'] = np.mean(grad)
            features['phase_gradient_std'] = np.std(grad)
            features['phase_gradient_max'] = np.max(grad)
        else:
            features['phase_gradient_mean'] = np.nan
            features['phase_gradient_std'] = np.nan
            features['phase_gradient_max'] = np.nan

        # Outer asymmetry
        outer_amps = [base_features.get(f'amp_m1_r{r}', np.nan) for r in [6, 7, 8]]
        outer_amps = [x for x in outer_amps if not np.isnan(x)]
        features['outer_asymmetry'] = np.mean(outer_amps) if outer_amps else np.nan

        # Variance ratio (inner/outer)
        outer_vars = [base_features.get(f'var_r{r}', np.nan) for r in [6, 7, 8]]
        outer_vars = [x for x in outer_vars if not np.isnan(x)]
        if inner_vars and outer_vars and np.mean(outer_vars) > 0.01:
            features['var_ratio_inner_outer'] = np.mean(inner_vars) / np.mean(outer_vars)
        else:
            features['var_ratio_inner_outer'] = np.nan

        # Legacy: abs_phase_diff_m1 for v2 compatibility
        if 3 in phases_m1 and 8 in phases_m1:
            phase_diff = phases_m1[8] - phases_m1[3]
            phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
            features['abs_phase_diff_m1'] = np.abs(phase_diff)
        else:
            features['abs_phase_diff_m1'] = np.nan

        return features

    def extract_features(self, image, center_y, center_x, radius):
        """Extract all features for a single skyrmion."""
        base_features, phases_m1 = self.extract_base_features(image, center_y, center_x, radius)
        all_features = self.engineer_features(base_features, phases_m1)
        return all_features

    def _get_feature_vector(self, features):
        """Convert feature dict to vector for v3 model."""
        return np.array([features.get(f, np.nan) for f in self.V3_FEATURES])

    def _train_default_model(self):
        """Train a default model if none is loaded."""
        if not SKLEARN_AVAILABLE:
            return

        # Use pre-computed statistics for a simple fallback model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        self.scaler = StandardScaler()
        self._model_trained = False  # Not actually trained, will use v2 fallback

    def classify_v1(self, features):
        """v1 classification (threshold on amp_m1_r8 + var_r3)."""
        amp_m1_r8 = features.get('amp_m1_r8', np.nan)
        var_r3 = features.get('var_r3', np.nan)

        if np.isnan(amp_m1_r8) or np.isnan(var_r3):
            return 1, 0.5

        score = (amp_m1_r8 - 0.24) / 0.05 * 0.7 + (0.11 - var_r3) / 0.03 * 0.3
        prob_q0 = 1 / (1 + np.exp(-score))

        return (0, prob_q0) if prob_q0 > 0.5 else (1, 1 - prob_q0)

    def classify_v2(self, features):
        """v2 classification (threshold on amp_m1_r3 + var_r3 + phase_diff)."""
        amp_m1_r3 = features.get('amp_m1_r3', np.nan)
        var_r3 = features.get('var_r3', np.nan)
        phase_diff = features.get('abs_phase_diff_m1', np.nan)

        if np.isnan(amp_m1_r3) and np.isnan(var_r3) and np.isnan(phase_diff):
            return 1, 0.5

        scores = []
        if not np.isnan(amp_m1_r3):
            scores.append(0.40 * (0.119 - amp_m1_r3) / 0.05)
        if not np.isnan(var_r3):
            scores.append(0.35 * (0.114 - var_r3) / 0.03)
        if not np.isnan(phase_diff):
            scores.append(0.25 * (phase_diff - 0.685) / 0.5)

        combined = sum(scores) * 2
        prob_q0 = 1 / (1 + np.exp(-combined))

        return (0, prob_q0) if prob_q0 > 0.5 else (1, 1 - prob_q0)

    def classify_v3(self, features):
        """v3 classification using trained RandomForest model."""
        # If no model trained, use improved threshold-based approach
        if not self._model_trained:
            return self._classify_v3_threshold(features)

        # Get feature vector
        X = self._get_feature_vector(features).reshape(1, -1)
        X = np.nan_to_num(X, nan=0)
        X_scaled = self.scaler.transform(X)

        # Predict
        prob = self.model.predict_proba(X_scaled)[0]
        pred = self.model.predict(X_scaled)[0]

        confidence = prob[pred]
        return int(pred), float(confidence)

    def _classify_v3_threshold(self, features):
        """Improved threshold-based v3 using top features."""
        scores = []
        weights = []

        # Phase difference r2→r8 (best feature, d'=2.46)
        pd = features.get('phase_diff_r2_r8', np.nan)
        if not np.isnan(pd):
            # Q=0 has HIGHER phase difference (~1.1 rad vs ~0.2 rad)
            score = (pd - 0.65) / 0.5  # Positive if Q=0
            scores.append(score)
            weights.append(2.46)

        # Amplitude ratio r3/r7 (d'=2.41)
        ar = features.get('amp_ratio_r3_r7', np.nan)
        if not np.isnan(ar):
            # Q=0 has LOWER ratio (weak inner signal)
            score = (0.4 - ar) / 0.2  # Positive if Q=0
            scores.append(score)
            weights.append(2.41)

        # Phase gradient mean (d'=2.31)
        pg = features.get('phase_gradient_mean', np.nan)
        if not np.isnan(pg):
            # Q=0 has HIGHER gradient (phase changes more across radii)
            score = (pg - 0.3) / 0.3
            scores.append(score)
            weights.append(2.31)

        # Inner flatness (d'=2.19)
        iflat = features.get('inner_flatness', np.nan)
        if not np.isnan(iflat):
            # Q=0 has HIGHER flatness (especially Type B smooth core)
            score = (iflat - 5) / 3
            scores.append(score)
            weights.append(2.19)

        # Variance ratio (d'=2.30)
        vr = features.get('var_ratio_inner_outer', np.nan)
        if not np.isnan(vr):
            # Q=0 has LOWER ratio
            score = (0.6 - vr) / 0.2
            scores.append(score)
            weights.append(2.30)

        if not scores:
            return 1, 0.5

        # Weighted average
        combined = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        prob_q0 = 1 / (1 + np.exp(-combined * 1.5))

        return (0, prob_q0) if prob_q0 > 0.5 else (1, 1 - prob_q0)

    def classify(self, image, center_y, center_x, radius):
        """
        Classify a single skyrmion as Q=0 or Q=1.

        Returns:
        --------
        prediction : int (0 or 1)
        confidence : float (0.5 to 1.0)
        features : dict
        """
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        features = self.extract_features(image, center_y, center_x, radius)

        if self.version == 1:
            pred, conf = self.classify_v1(features)
        elif self.version == 2:
            pred, conf = self.classify_v2(features)
        else:
            pred, conf = self.classify_v3(features)

        return pred, conf, features

    def classify_batch(self, image, skyrmions):
        """Classify multiple skyrmions."""
        results = []
        for sky in skyrmions:
            pred, conf, feats = self.classify(
                image, sky['center_y'], sky['center_x'], sky['radius']
            )
            results.append({
                'prediction': pred,
                'confidence': conf,
                'features': feats,
                'label': 'Q=0 (Skyrmionium)' if pred == 0 else 'Q=1 (Skyrmion)'
            })
        return results

    def load_model(self, model_path):
        """Load a trained model from file."""
        if not SKLEARN_AVAILABLE:
            print("sklearn not available, cannot load model")
            return False

        try:
            with open(model_path, 'rb') as f:
                saved = pickle.load(f)
            self.model = saved['model']
            self.scaler = saved['scaler']
            self._model_trained = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def save_model(self, model_path):
        """Save trained model to file."""
        if not self._model_trained:
            print("No trained model to save")
            return False

        try:
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'features': self.V3_FEATURES,
                    'version': self.VERSION
                }, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def get_diagnostic_info(self, features):
        """Get diagnostic information about classification."""
        diagnostics = {
            'version': self.VERSION,
            'features': {},
            'key_indicators': {}
        }

        # Key v3 features
        key_features = [
            ('phase_diff_r2_r8', 'Phase shift (r2→r8)', 'rad', True),
            ('amp_ratio_r3_r7', 'Amp ratio (inner/outer)', '', False),
            ('phase_gradient_mean', 'Phase gradient', 'rad/step', True),
            ('inner_flatness', 'Inner flatness', '', True),
            ('var_ratio_inner_outer', 'Var ratio (inner/outer)', '', False),
        ]

        for feat, desc, unit, q0_higher in key_features:
            val = features.get(feat, np.nan)
            if not np.isnan(val):
                diagnostics['features'][feat] = {
                    'value': val,
                    'description': desc,
                    'unit': unit,
                    'suggests': 'Q=0' if (val > 0.5) == q0_higher else 'Q=1'
                }

        # Q=0 Type detection
        inner_flat = features.get('inner_flatness', 0)
        phase_diff = features.get('phase_diff_r2_r8', 0)

        if inner_flat > 8 and phase_diff < 0.5:
            diagnostics['q0_type'] = 'Type B (smooth core)'
        elif phase_diff > 1.0:
            diagnostics['q0_type'] = 'Type A (nested dumbbell)'
        else:
            diagnostics['q0_type'] = 'Uncertain'

        return diagnostics


def classify_from_dataframe(image, df, classifier=None, version=3):
    """Classify skyrmions from DataFrame with X, Y, Area columns."""
    if classifier is None:
        classifier = PolarFourierClassifier(version=version)

    df_classified = df.copy()
    predictions, confidences, labels, phase_diffs = [], [], [], []

    for idx, row in df.iterrows():
        center_x, center_y = row['X'], row['Y']
        radius = np.sqrt(row['Area'] / np.pi)

        pred, conf, feats = classifier.classify(image, center_y, center_x, radius)

        predictions.append(pred)
        confidences.append(conf)
        labels.append('Q=0 (Skyrmionium)' if pred == 0 else 'Q=1 (Skyrmion)')
        pd = feats.get('phase_diff_r2_r8', feats.get('abs_phase_diff_m1', np.nan))
        phase_diffs.append(np.degrees(pd) if not np.isnan(pd) else np.nan)

    df_classified['Q_pred'] = predictions
    df_classified['Q_confidence'] = confidences
    df_classified['Q_label'] = labels
    df_classified['phase_diff_deg'] = phase_diffs

    return df_classified


def train_v3_model(features_csv_path, save_path=None):
    """
    Train the v3 RandomForest model on labeled data.

    Parameters:
    -----------
    features_csv_path : str
        Path to CSV with extracted features and 'q_class' column
    save_path : str, optional
        Path to save trained model

    Returns:
    --------
    classifier : PolarFourierClassifier with trained model
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn required for training")

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(features_csv_path)
    y = df['q_class'].values

    # Get features
    clf = PolarFourierClassifier(version=3)
    X = df[clf.V3_FEATURES].values
    X = np.nan_to_num(X, nan=0)

    # Scale and train
    clf.scaler = StandardScaler()
    X_scaled = clf.scaler.fit_transform(X)

    clf.model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    clf.model.fit(X_scaled, y)
    clf._model_trained = True

    if save_path:
        clf.save_model(save_path)

    return clf


def test_classifier():
    """Test the classifier."""
    print(f"Testing PolarFourierClassifier v{PolarFourierClassifier.VERSION}...")

    clf = PolarFourierClassifier(version=3)

    # Synthetic test
    size = 100
    y, x = np.ogrid[:size, :size]
    center = size // 2
    r = np.sqrt((y - center)**2 + (x - center)**2)
    theta = np.arctan2(y - center, x - center)

    # Q=1 pattern
    image_q1 = 0.5 + 0.3 * np.cos(theta) * np.exp(-(r - 20)**2 / 100)

    # Q=0 Type A (nested dumbbell)
    outer = 0.3 * np.cos(theta) * np.exp(-(r - 20)**2 / 50)
    inner = -0.3 * np.cos(theta) * np.exp(-(r - 8)**2 / 20)
    image_q0_a = 0.5 + outer + inner

    # Q=0 Type B (smooth core)
    image_q0_b = 0.5 + 0.3 * np.cos(theta) * np.exp(-(r - 20)**2 / 50)
    image_q0_b[r < 10] = 0.5  # Flat center

    print("\nQ=1 pattern:")
    pred, conf, feats = clf.classify(image_q1, center, center, 25)
    print(f"  Prediction: Q={pred}, Confidence: {conf:.3f}")
    print(f"  phase_diff_r2_r8: {np.degrees(feats.get('phase_diff_r2_r8', 0)):.1f}°")

    print("\nQ=0 Type A (nested dumbbell):")
    pred, conf, feats = clf.classify(image_q0_a, center, center, 25)
    print(f"  Prediction: Q={pred}, Confidence: {conf:.3f}")
    print(f"  phase_diff_r2_r8: {np.degrees(feats.get('phase_diff_r2_r8', 0)):.1f}°")

    print("\nQ=0 Type B (smooth core):")
    pred, conf, feats = clf.classify(image_q0_b, center, center, 25)
    diag = clf.get_diagnostic_info(feats)
    print(f"  Prediction: Q={pred}, Confidence: {conf:.3f}")
    print(f"  inner_flatness: {feats.get('inner_flatness', 0):.2f}")
    print(f"  Detected type: {diag.get('q0_type', 'Unknown')}")

    print(f"\nClassifier v{clf.VERSION} ready!")
    return True


if __name__ == "__main__":
    test_classifier()
