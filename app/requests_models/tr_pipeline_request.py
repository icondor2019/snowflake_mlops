from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class TrainingPipelineRequest(BaseModel):
    """Request model for training pipeline execution."""
    
    snapshot_date: datetime = Field(
        ..., 
        description="Snapshot date to query features from feature store"
    )
    model_name: str = Field(
        ..., 
        description="Name of the model to train/update in registry"
    )
    feature_view_name: str = Field(
        default="training_features",
        description="Name of the feature view to query"
    )
    target_column: str = Field(
        default="target",
        description="Name of the target column in features"
    )
    test_size: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Test set size fraction"
    )
    random_state: int = Field(
        default=42,
        description="Random state for reproducibility"
    )
    enable_hyperparameter_tuning: bool = Field(
        default=False,
        description="Whether to perform hyperparameter tuning or use champion params"
    )
    tuning_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Hyperparameter tuning configuration (grid/random search params)"
    )
    hyperparameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Specific hyperparameters to use for training (overrides champion params if provided)"
    )
    tags: Optional[Dict[str, str]] = Field(
        default=None,
        description="Tags to apply to the trained model (e.g., {'env': 'prod'})"
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of this training run"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "snapshot_date": "2026-03-04T00:00:00",
                "model_name": "RandomForestModel",
                "feature_view_name": "training_features",
                "target_column": "target",
                "test_size": 0.2,
                "enable_hyperparameter_tuning": False,
            }
        }
