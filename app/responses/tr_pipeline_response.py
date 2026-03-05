from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class ModelComparisonResult(BaseModel):
    """Result of comparing challenger model with champion."""
    
    status: str = Field(..., description="Comparison status (compared/no_champion/error)")
    champion_version: Optional[str] = Field(None, description="Name of champion version")
    challenger_version: Optional[str] = Field(None, description="Name of challenger version")
    metric_name: Optional[str] = Field(None, description="Metric used for comparison")
    champion_score: Optional[float] = Field(None, description="Champion metric score")
    challenger_score: Optional[float] = Field(None, description="Challenger metric score")
    improvement: Optional[float] = Field(None, description="Absolute improvement")
    improvement_pct: Optional[float] = Field(None, description="Percentage improvement")
    challenger_is_better: Optional[bool] = Field(None, description="Whether challenger is better")
    recommendation: str = Field(..., description="Recommendation (promote_challenger/keep_champion)")


class TrainingPipelineResponse(BaseModel):
    """Response model for training pipeline execution."""
    
    status: str = Field(
        ..., 
        description="Pipeline execution status (success/failed)"
    )
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Pipeline execution results"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if pipeline failed"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of response"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "data": {
                    "model_name": "RandomForestModel",
                    "trained_version": "v2",
                    "snapshot_date": "2026-03-04T00:00:00",
                    "metrics": {
                        "accuracy": 0.92,
                        "f1_score": 0.89,
                        "precision": 0.91,
                        "recall": 0.87
                    },
                    "comparison_result": {
                        "status": "compared",
                        "champion_version": "v1",
                        "challenger_version": "v2",
                        "metric_name": "accuracy",
                        "champion_score": 0.90,
                        "challenger_score": 0.92,
                        "improvement": 0.02,
                        "improvement_pct": 2.22,
                        "challenger_is_better": True,
                        "recommendation": "promote_challenger"
                    },
                    "promoted_to_champion": True,
                    "training_time_seconds": 45.32,
                    "features_count": 150,
                    "training_samples": 8000,
                    "test_samples": 2000
                },
                "timestamp": "2026-03-04T10:30:15"
            }
        }
