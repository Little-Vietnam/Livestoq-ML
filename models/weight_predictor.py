"""
Weight Prediction Module
=========================
Predicts the live weight of a cow from its body dimensions using
multiple estimation approaches:

  1. **Schaeffer's Formula** (allometric regression)
     W = a × (chest_girth)^b × (body_length)^c

  2. **Heart-Girth Formula** (widely used in livestock science)
     W = (chest_girth² × body_length) / k

  3. **Multi-Measurement Regression** (Gradient Boosting)
     Uses all 8 body measurements as features

  4. **Ensemble** – weighted average of all methods

All weights are in kilograms (kg).
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
import json
import os


@dataclass
class WeightResult:
    """Result of weight prediction."""
    predicted_weight_kg: float              # Final ensemble weight
    weight_range_kg: Tuple[float, float]    # (min, max) confidence interval
    method_weights: Dict[str, float]        # Weight from each method
    method_confidences: Dict[str, float]    # Confidence per method
    breed: str = "generic"
    breed_factor: float = 1.0
    bcs_estimate: float = 5.0              # Body Condition Score (1-9)
    details: Dict = field(default_factory=dict)


class WeightPredictor:
    """
    Predicts cow live weight from body dimensions using empirical
    formulas and a trained ML model.
    """

    def __init__(self, weight_config):
        self.cfg = weight_config
        self._gb_model = None
        self._scaler = None

    def predict(
        self,
        dimensions: Dict[str, float],
        breed: str = "generic",
    ) -> WeightResult:
        """
        Predict the cow's weight from extracted body dimensions.

        Parameters
        ----------
        dimensions : dict
            Must contain keys like 'body_length_cm', 'chest_girth_cm', etc.
        breed : str
            Breed name for breed-specific correction.

        Returns
        -------
        WeightResult
        """
        breed_factor = self.cfg.breed_factors.get(
            breed.lower(), self.cfg.breed_factors["generic"]
        )

        # Collect predictions from each method
        predictions = {}
        confidences = {}

        # Method 1: Schaeffer's Formula
        w1, c1 = self._schaeffer_formula(dimensions)
        if w1 > 0:
            predictions["schaeffer"] = w1 * breed_factor
            confidences["schaeffer"] = c1

        # Method 2: Heart-Girth Formula
        w2, c2 = self._heart_girth_formula(dimensions)
        if w2 > 0:
            predictions["heart_girth"] = w2 * breed_factor
            confidences["heart_girth"] = c2

        # Method 3: Body-Length × Girth² Formula (Crevat-Lagneau)
        w3, c3 = self._crevat_lagneau_formula(dimensions)
        if w3 > 0:
            predictions["crevat_lagneau"] = w3 * breed_factor
            confidences["crevat_lagneau"] = c3

        # Method 4: Multi-measurement regression
        w4, c4 = self._regression_predict(dimensions)
        if w4 > 0:
            predictions["regression"] = w4 * breed_factor
            confidences["regression"] = c4

        # Method 5: Body-surface-area based
        w5, c5 = self._bsa_method(dimensions)
        if w5 > 0:
            predictions["bsa"] = w5 * breed_factor
            confidences["bsa"] = c5

        # ── Ensemble ────────────────────────────────────────────────────
        if not predictions:
            return WeightResult(
                predicted_weight_kg=0.0,
                weight_range_kg=(0.0, 0.0),
                method_weights={},
                method_confidences={},
                breed=breed,
                breed_factor=breed_factor,
            )

        final_weight = self._ensemble(predictions, confidences)

        # Apply field correction factor (2/3 scaling)
        final_weight = final_weight * (2 / 3)

        # Estimate BCS from body proportions
        bcs = self._estimate_bcs(dimensions)

        # Confidence interval (±15%)
        weight_min = final_weight * 0.85
        weight_max = final_weight * 1.15

        return WeightResult(
            predicted_weight_kg=round(final_weight, 1),
            weight_range_kg=(round(weight_min, 1), round(weight_max, 1)),
            method_weights={k: round(v, 1) for k, v in predictions.items()},
            method_confidences=confidences,
            breed=breed,
            breed_factor=breed_factor,
            bcs_estimate=bcs,
            details={
                "input_dimensions": dimensions,
                "num_methods_used": len(predictions),
            },
        )

    # ── Formula Methods ──────────────────────────────────────────────────

    def _schaeffer_formula(
        self, dims: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Schaeffer's allometric formula:
        W = a × CG^b × BL^c

        Where CG = chest girth (cm), BL = body length (cm)
        """
        cg = dims.get("chest_girth_cm", 0)
        bl = dims.get("body_length_cm", 0)

        if cg <= 0 or bl <= 0:
            return 0.0, 0.0

        weight = self.cfg.schaeffer_a * (cg ** self.cfg.schaeffer_b) * (bl ** self.cfg.schaeffer_c)

        # Sanity bounds
        weight = np.clip(weight, 50, 1500)
        confidence = 0.7 if (cg > 100 and bl > 80) else 0.4

        return float(weight), confidence

    def _heart_girth_formula(
        self, dims: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Classic Heart-Girth formula (widely used for cattle):
        W (lbs) = (CG² × BL) / 300
        W (kg)  = W_lbs × 0.453592

        Where CG and BL are in inches.
        """
        cg_cm = dims.get("chest_girth_cm", 0)
        bl_cm = dims.get("body_length_cm", 0)

        if cg_cm <= 0 or bl_cm <= 0:
            return 0.0, 0.0

        # Convert to inches
        cg_in = cg_cm / 2.54
        bl_in = bl_cm / 2.54

        weight_lbs = (cg_in ** 2 * bl_in) / 300.0
        weight_kg = weight_lbs * 0.453592

        weight_kg = np.clip(weight_kg, 50, 1500)
        confidence = 0.65

        return float(weight_kg), confidence

    def _crevat_lagneau_formula(
        self, dims: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Crevat-Lagneau formula:
        W (kg) = 80 × (CG in meters)³

        Simpler but less accurate. Uses chest girth only.
        """
        cg_cm = dims.get("chest_girth_cm", 0)

        if cg_cm <= 0:
            return 0.0, 0.0

        cg_m = cg_cm / 100.0
        weight_kg = 80.0 * (cg_m ** 3)

        weight_kg = np.clip(weight_kg, 50, 1500)
        confidence = 0.55

        return float(weight_kg), confidence

    def _regression_predict(
        self, dims: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Gradient-Boosting regression using all available measurements.
        If no trained model exists, uses a hand-crafted linear model
        based on literature coefficients.
        """
        # Try loading a trained model
        if self._gb_model is not None:
            return self._predict_with_trained_model(dims)

        # Fallback: literature-based multivariate linear model
        # W = β0 + β1×CG + β2×BL + β3×BH + β4×CD + β5×BW
        # Coefficients from published cattle weight estimation studies
        coefficients = {
            "intercept": -850.0,
            "chest_girth_cm": 3.8,
            "body_length_cm": 2.1,
            "body_height_cm": 1.5,
            "chest_depth_cm": 2.5,
            "body_width_cm": 1.8,
            "abdominal_girth_cm": 1.2,
        }

        weight = coefficients["intercept"]
        features_used = 0

        for feat, coef in coefficients.items():
            if feat == "intercept":
                continue
            val = dims.get(feat, 0)
            if val > 0:
                weight += coef * val
                features_used += 1

        if features_used < 2:
            return 0.0, 0.0

        weight = np.clip(weight, 50, 1500)
        confidence = min(0.6, 0.2 + features_used * 0.1)

        return float(weight), confidence

    def _predict_with_trained_model(
        self, dims: Dict[str, float]
    ) -> Tuple[float, float]:
        """Use a trained sklearn model for prediction."""
        feature_order = [
            "body_length_cm", "body_width_cm", "tube_girth_cm",
            "body_height_cm", "chest_width_cm", "abdominal_girth_cm",
            "chest_depth_cm", "chest_girth_cm",
        ]
        X = np.array([[dims.get(f, 0) for f in feature_order]])

        if self._scaler is not None:
            X = self._scaler.transform(X)

        weight = float(self._gb_model.predict(X)[0])
        weight = np.clip(weight, 50, 1500)
        return weight, 0.7

    def _bsa_method(
        self, dims: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Body Surface Area method:
        BSA (m²) ≈ 0.09 × W^0.67  (Brody's equation)
        → W ≈ (BSA / 0.09)^(1/0.67)

        BSA estimated from body dimensions:
        BSA ≈ π × (BL/100) × (CG/100) / 4   (cylinder approximation)
        """
        bl = dims.get("body_length_cm", 0) / 100.0  # meters
        cg = dims.get("chest_girth_cm", 0) / 100.0  # meters

        if bl <= 0 or cg <= 0:
            return 0.0, 0.0

        # Approximate BSA using cylinder model
        radius = cg / (2 * np.pi)
        bsa = 2 * np.pi * radius * bl + 2 * np.pi * radius ** 2

        # Solve Brody's equation for weight
        weight = (bsa / 0.09) ** (1.0 / 0.67)

        weight = np.clip(weight, 50, 1500)

        return float(weight), 0.45

    # ── Ensemble ─────────────────────────────────────────────────────────

    def _ensemble(
        self,
        predictions: Dict[str, float],
        confidences: Dict[str, float],
    ) -> float:
        """
        Weighted ensemble: combine predictions using confidence-based
        weights with outlier rejection.
        """
        values = np.array(list(predictions.values()))
        confs = np.array([confidences[k] for k in predictions.keys()])

        # Reject outliers (predictions > 2 std from median)
        median = np.median(values)
        std = np.std(values) if len(values) > 1 else median * 0.3
        mask = np.abs(values - median) < 2 * std

        if mask.sum() == 0:
            mask = np.ones_like(mask, dtype=bool)

        filtered_values = values[mask]
        filtered_confs = confs[mask]

        # Confidence-weighted average
        total_conf = filtered_confs.sum()
        if total_conf > 0:
            weighted_avg = (filtered_values * filtered_confs).sum() / total_conf
        else:
            weighted_avg = filtered_values.mean()

        return float(weighted_avg)

    # ── Body Condition Score ─────────────────────────────────────────────

    def _estimate_bcs(self, dims: Dict[str, float]) -> float:
        """
        Rough BCS (1-9 scale) estimate from body proportions.
        Uses chest depth / body height ratio as a proxy.
        """
        cd = dims.get("chest_depth_cm", 0)
        bh = dims.get("body_height_cm", 0)
        cg = dims.get("chest_girth_cm", 0)
        bl = dims.get("body_length_cm", 0)

        if bh > 0 and cd > 0:
            ratio = cd / bh
            # Higher ratio → more filled out → higher BCS
            bcs = 1.0 + ratio * 10.0
            bcs = np.clip(bcs, 1.0, 9.0)
        elif cg > 0 and bl > 0:
            ratio = cg / bl
            bcs = 1.0 + ratio * 5.0
            bcs = np.clip(bcs, 1.0, 9.0)
        else:
            bcs = 5.0

        return round(float(bcs), 1)

    # ── Model I/O ────────────────────────────────────────────────────────

    def save_model(self, path: str):
        """Save the trained regression model."""
        if self._gb_model is not None:
            import joblib
            joblib.dump({
                "model": self._gb_model,
                "scaler": self._scaler,
            }, path)

    def load_model(self, path: str):
        """Load a pre-trained regression model."""
        if os.path.exists(path):
            import joblib
            data = joblib.load(path)
            self._gb_model = data["model"]
            self._scaler = data.get("scaler")
            print(f"[Weight] Loaded model from {path}")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Train a gradient-boosting regression model on collected data.

        Parameters
        ----------
        X : np.ndarray – (n_samples, 8) dimension features.
        y : np.ndarray – (n_samples,) ground-truth weights in kg.
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._gb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self._gb_model.fit(X_scaled, y)

        # Cross-validation
        scores = cross_val_score(
            self._gb_model, X_scaled, y,
            cv=min(5, len(y)), scoring="r2",
        )
        print(f"[Weight] Model trained. CV R²: {scores.mean():.3f} ± {scores.std():.3f}")

        if feature_names:
            importances = self._gb_model.feature_importances_
            for name, imp in sorted(
                zip(feature_names, importances),
                key=lambda x: -x[1],
            ):
                print(f"  {name}: {imp:.3f}")
