from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import shap

from api.schemas.request import InferenceRequest
from api.schemas.response import FeatureContribution, InferenceResponse
from api.services.inference_frame_service import InferenceFrameService
from feature_engineering.OpenFE.openfe import tree_to_formula

SHAP_SEED = 42


class ModelService:
    """Service for model-backed inference operations."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        """Initialize the model service and the SHAP explainer.

        Args:
            bundle (dict[str, Any]): Loaded inference bundle.
        """
        self.bundle = bundle
        self.model = bundle.get("model")
        self.fitted_column_transformer = bundle.get("fitted_column_transformer")
        self.row_wise_features = bundle.get("row_wise_features")
        self.column_wise_features = bundle.get("column_wise_features")
        self.best_features = bundle.get("best_features", bundle.get("selected_features"))
        self.categorical_mapping = bundle.get("categorical_mapping") or {}
        self.inference_frame_service = InferenceFrameService(bundle)

        background = InferenceFrameService.build_frame(
            X_pd=bundle["shap_background"],
            row_wise_features=self.row_wise_features,
            column_wise_features=self.column_wise_features,
            column_transformer=self.fitted_column_transformer,
            best_features=self.best_features,
            categorical_mapping=self.categorical_mapping,
        )
        self.feature_columns = list(background.columns)
        self.categorical_columns = [col for col in self.feature_columns if col in self.categorical_mapping]
        self.success_idx = list(self.model.classes_).index(1)
        self.display_names = self._build_display_names()
        self.explainer = shap.PermutationExplainer(
            self._predict_success,
            shap.maskers.Independent(self._encode(background), max_samples=len(background)),
        )
        self.explainer(self._encode(background.iloc[[0]]))

    def is_model_available(self) -> bool:
        """Check if model is loaded and available.

        Returns:
            bool: True if model is available.
        """
        return self.model is not None

    def predict(self, payload: InferenceRequest) -> InferenceResponse:
        """Run prediction and SHAP explanation for a single inference payload.

        Args:
            payload (InferenceRequest): Inference payload.

        Raises:
            ValueError: If model does not support required prediction interface.
            ValueError: If prediction output shape is invalid.

        Returns:
            InferenceResponse: Prediction response payload.
        """
        model_input = self.inference_frame_service.build_from_payload(payload)
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Loaded model does not support predict_proba")

        proba_raw = self.model.predict_proba(model_input)
        proba_array = np.asarray(proba_raw)
        if proba_array.ndim != 2:
            raise ValueError("Invalid predict_proba output")

        class_idx = int(np.argmax(proba_array[0]))
        probability = float(proba_array[0, class_idx])
        classes = getattr(self.model, "classes_", None)
        prediction = classes[class_idx] if classes is not None else class_idx

        np.random.seed(SHAP_SEED)
        explanation = self.explainer(self._encode(model_input))
        shap_values = explanation.values[0]
        contributions = [
            FeatureContribution(
                feature=self.display_names[col],
                value=self._to_json_value(model_input[col].iloc[0]),
                contribution=float(shap_values[idx]),
            )
            for idx, col in enumerate(self.feature_columns)
        ]
        contributions.sort(key=lambda item: abs(item.contribution), reverse=True)

        return InferenceResponse(
            prediction=prediction,
            probability=probability,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=type(self.model).__name__,
            base_value=float(explanation.base_values[0]),
            contributions=contributions,
        )

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        """Encode a model-ready frame as a float array with categorical codes for the SHAP masker.

        Args:
            X (pd.DataFrame): Model-ready frame.

        Returns:
            np.ndarray: Numeric array in `feature_columns` order.
        """
        X_encoded = X[self.feature_columns].copy()
        for col in self.categorical_columns:
            X_encoded[col] = X_encoded[col].cat.codes
        return X_encoded.to_numpy(dtype=float)

    def _predict_success(self, X: np.ndarray) -> np.ndarray:
        """Predict P(success) for an encoded array produced by `_encode`.

        Args:
            X (np.ndarray): Encoded feature array.

        Returns:
            np.ndarray: Success probability per row.
        """
        X_decoded = pd.DataFrame(X, columns=self.feature_columns)
        for col in self.categorical_columns:
            X_decoded[col] = pd.Categorical.from_codes(
                X_decoded[col].astype(int), categories=self.categorical_mapping[col]
            )
        return self.model.predict_proba(X_decoded)[:, self.success_idx]

    def _build_display_names(self) -> dict[str, str]:
        """Map model column names to their OpenFE formulas where the column is an alias.

        Returns:
            dict[str, str]: Model column name to display name.
        """
        display_names = {col: col for col in self.feature_columns}
        for idx, node in enumerate(self.row_wise_features or []):
            alias = f"autoFE_f_{idx}"
            if alias in display_names:
                display_names[alias] = tree_to_formula(node)
        feature_name_mapping = getattr(self.fitted_column_transformer, "feature_name_mapping", {}) or {}
        for formula, safe_name in feature_name_mapping.items():
            if safe_name in display_names:
                display_names[safe_name] = formula
        return display_names

    @staticmethod
    def _to_json_value(value: Any) -> float | str | None:
        """Convert a single frame cell to a JSON-safe value.

        Args:
            value (Any): Frame cell value.

        Returns:
            float | str | None: Float for numeric values, string for categories, None when missing.
        """
        if pd.isna(value):
            return None
        if isinstance(value, str):
            return value
        return float(value)
