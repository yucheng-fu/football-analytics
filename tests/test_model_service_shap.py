from pathlib import Path

import numpy as np
import pytest

from api.schemas.request import InferenceRequest
from api.services.model_service import ModelService
from utils.inference_utils import load_inference_bundle_from_local_artifacts

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "src" / "api" / "artifacts" / "lightgbm"


@pytest.fixture(scope="module")
def model_service():
    bundle = load_inference_bundle_from_local_artifacts(artifact_dir=str(ARTIFACT_DIR), model_type="lightgbm")
    return ModelService(bundle)


@pytest.fixture
def payload():
    return InferenceRequest(
        start_x=60.0,
        start_y=40.0,
        end_x=78.0,
        end_y=35.0,
        length=22.5,
        height="Ground Pass",
        angle=0.25,
        duration=1.35,
        body_part="Right Foot",
        under_pressure=1,
    )


def test_shap_contributions_sum_to_success_probability(model_service, payload):
    response = model_service.predict(payload)
    success_probability = response.probability if response.prediction == 1 else 1 - response.probability
    total = response.base_value + sum(item.contribution for item in response.contributions)
    assert total == pytest.approx(success_probability, abs=1e-6)


def test_shap_contributions_cover_every_model_feature_sorted_by_magnitude(model_service, payload):
    response = model_service.predict(payload)
    magnitudes = [abs(item.contribution) for item in response.contributions]
    assert len(response.contributions) == len(model_service.feature_columns)
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_shap_display_names_resolve_openfe_aliases(model_service, payload):
    response = model_service.predict(payload)
    names = {item.feature for item in response.contributions}
    assert not any(name.startswith("autoFE_f_") or name.startswith("ofe_col_") for name in names)
    assert np.isclose(
        next(item.value for item in response.contributions if item.feature == "under_pressure"),
        1.0,
    )
