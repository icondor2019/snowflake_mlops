import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
import time
import os
from loguru import logger

from utils.snowflake_mlops import SnowflakeMLOpsManager
from app.requests_models.tr_pipeline_request import TrainingPipelineRequest
from app.responses.tr_pipeline_response import TrainingPipelineResponse, ModelComparisonResult


class TrainingPipelineException(Exception):
    """Custom exception for training pipeline errors."""
    pass


class TrainingPipeline:
    """
    Machine Learning Training Pipeline for model training, evaluation, and registration.
    
    Orchestrates the complete ML workflow:
    1. Load features from feature store with snapshot_date
    2. Retrieve champion model parameters
    3. Train new model (with optional hyperparameter tuning)
    4. Compare with champion model
    5. Register model with metrics and metadata
    6. Promote to champion if better
    """
    
    # Default hyperparameters for Random Forest
    DEFAULT_RF_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
    }
    
    # Hyperparameter tuning grid
    RF_TUNING_GRID = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    
    def __init__(self, 
                 session,
                 experiment_name: str = "ml_training_pipeline",
                 database: str = "AI_PROJECT"):
        """
        Initialize training pipeline with Snowflake components.
        
        Args:
            session: Snowflake Snowpark session
            experiment_name: Name for MLOps experiment tracking
            database: Snowflake database name
            schema: Snowflake schema for features
        """
        if not session:
            raise TrainingPipelineException("Snowflake session is required")
        
        self.session = session
        self.database = database
        self.schema_models = os.getenv("SNOWFLAKE_SCHEMA_MODELS", "mlops")
        self.schema_features = os.getenv("SNOWFLAKE_SCHEMA_FEATURES", "features")
        
        # Initialize Snowflake MLOps manager
        self.mlops_manager = SnowflakeMLOpsManager(
            session=session,
            experiment_name=experiment_name,
            database=database,
            schema=self.schema_models
        )
        
        # Pipeline state
        self._features_df: Optional[pd.DataFrame] = None
        self._X_train: Optional[pd.DataFrame] = None
        self._X_test: Optional[pd.DataFrame] = None
        self._y_train: Optional[pd.Series] = None
        self._y_test: Optional[pd.Series] = None
        self._trained_model: Optional[Any] = None
        self._model_metrics: Optional[Dict[str, float]] = None
        self._comparison_result: Optional[Dict[str, Any]] = None
        
        logger.info(f"TrainingPipeline initialized with experiment: {experiment_name}")
    
    def get_features(self, 
                    request: TrainingPipelineRequest) -> pd.DataFrame:
        """
        Get features from feature store using snapshot_date.
        
        Args:
            request: TrainingPipelineRequest with snapshot_date and feature_view_name
            
        Returns:
            DataFrame with features for the snapshot date
        """
        logger.info(f"Loading features from {request.feature_view_name} for snapshot_date {request.snapshot_date}")
        
        try:
            # Load data from feature view using MLOps manager
            self._features_df = self.mlops_manager.get_feature_store_view(
                feature_vw_name=request.feature_view_name,
                version="v1"
            )
            
            if self._features_df is None or len(self._features_df) == 0:
                raise TrainingPipelineException(
                    f"No features found in {request.feature_view_name}"
                )
            
            logger.info(f"Loaded {len(self._features_df)} samples with {len(self._features_df.columns)} features")
            
            return self._features_df
            
        except Exception as e:
            raise TrainingPipelineException(f"Failed to get features: {str(e)}")
    
    def get_champion_model(self, model_name: str) -> Dict[str, Any]:
        """
        Retrieve champion model and its hyperparameters from registry.
        
        Args:
            model_name: Name of the model in registry
            
        Returns:
            Dictionary with champion model info and parameters
        """
        logger.info(f"Retrieving champion model: {model_name}")
        
        try:
            # Get champion version
            champion_version = self.mlops_manager.get_champion_version(
                model_name=model_name,
                champion_tag_name="stage",
                champion_tag_value="champion"
            )
            
            if not champion_version:
                logger.warning(f"No champion found for {model_name}, will use default parameters")
                return {
                    "version_name": None,
                    "parameters": self.DEFAULT_RF_PARAMS,
                    "is_default": True,
                }
            
            # Get champion model parameters
            champion_params = self.mlops_manager.get_model_params(
                model_name=model_name,
                version_name=champion_version
            )
            
            logger.info(f"Retrieved champion version {champion_version}")
            
            return {
                "version_name": champion_version,
                "parameters": champion_params or self.DEFAULT_RF_PARAMS,
                "is_default": False,
            }
            
        except Exception as e:
            logger.warning(f"Failed to retrieve champion model: {str(e)}, using defaults")
            return {
                "version_name": None,
                "parameters": self.DEFAULT_RF_PARAMS,
                "is_default": True,
            }
    
    def prepare_data(self,
                    request: TrainingPipelineRequest) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare training and test datasets.
        
        Args:
            request: TrainingPipelineRequest with test_size and random_state
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Preparing train/test split with test_size={request.test_size}")
        
        if self._features_df is None or len(self._features_df) == 0:
            raise TrainingPipelineException("Features not loaded. Call get_features() first.")
        
        try:
            # Separate features and target
            target_col = request.target_column.lower()
            if target_col not in self._features_df.columns:
                raise TrainingPipelineException(
                    f"Target column '{request.target_column}' not found in features"
                )
            
            X = self._features_df.drop(columns=[target_col])
            y = self._features_df[target_col]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=request.test_size,
                random_state=request.random_state
            )
            
            self._X_train = X_train
            self._X_test = X_test
            self._y_train = y_train
            self._y_test = y_test
            
            logger.info(f"Data prepared: train={len(X_train)}, test={len(X_test)}, features={len(X.columns)}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            raise TrainingPipelineException(f"Failed to prepare data: {str(e)}")
    
    def train_model(self,
                   request: TrainingPipelineRequest,
                   champion_params: Dict[str, Any]) -> Any:
        """
        Train model with specified parameters or with hyperparameter tuning.
        
        Args:
            request: TrainingPipelineRequest with hyperparameter settings
            champion_params: Champion model parameters
            
        Returns:
            Trained model object
        """
        logger.info(f"Starting model training (enable_tuning={request.enable_hyperparameter_tuning})")
        
        if self._X_train is None:
            raise TrainingPipelineException("Data not prepared. Call prepare_data() first.")
        
        try:
            # Use provided hyperparameters or champion parameters
            if request.hyperparameters:
                params = request.hyperparameters
                logger.info(f"Using provided hyperparameters: {params}")
            else:
                params = champion_params
                logger.info(f"Using champion hyperparameters: {params}")
            
            if request.enable_hyperparameter_tuning:
                # Perform hyperparameter tuning
                logger.info("Performing hyperparameter tuning with GridSearchCV")
                
                tuning_config = request.tuning_params or {
                    'cv': 5,
                    'n_jobs': -1,
                    'scoring': 'accuracy' if self._y_train.dtype == 'int64' else 'neg_mean_squared_error'
                }
                
                rf = RandomForestClassifier(**params)
                grid_search = GridSearchCV(
                    rf,
                    self.RF_TUNING_GRID,
                    cv=tuning_config.get('cv', 5),
                    n_jobs=tuning_config.get('n_jobs', -1),
                    scoring=tuning_config.get('scoring', 'accuracy')
                )
                
                grid_search.fit(self._X_train, self._y_train)
                self._trained_model = grid_search.best_estimator_
                
                logger.info(f"Tuning completed. Best params: {grid_search.best_params_}")
                logger.info(f"Best score: {grid_search.best_score_:.4f}")
                
            else:
                # Train with fixed parameters
                self._trained_model = RandomForestClassifier(**params)
                self._trained_model.fit(self._X_train, self._y_train)
                logger.info("Model trained with champion/provided parameters")
            
            return self._trained_model
            
        except Exception as e:
            raise TrainingPipelineException(f"Failed to train model: {str(e)}")
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute evaluation metrics on test set.
        
        Returns:
            Dictionary with classification or regression metrics
        """
        logger.info("Computing evaluation metrics")
        
        if self._trained_model is None or self._X_test is None:
            raise TrainingPipelineException("Model not trained or data not prepared")
        
        try:
            y_pred = self._trained_model.predict(self._X_test)
            
            # Classification metrics
            metrics = {
                'accuracy': accuracy_score(self._y_test, y_pred),
                'precision': precision_score(self._y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(self._y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(self._y_test, y_pred, average='weighted', zero_division=0),
            }
            
            self._model_metrics = metrics
            logger.info(f"Metrics computed: {metrics}")
            
            return metrics
            
        except Exception as e:
            raise TrainingPipelineException(f"Failed to compute metrics: {str(e)}")
    
    def compare_with_champion(self, model_name: str, request: TrainingPipelineRequest) -> Dict[str, Any]:
        """
        Compare trained model with champion model.
        
        Args:
            model_name: Name of the model in registry
            request: TrainingPipelineRequest
            
        Returns:
            Comparison result dictionary
        """
        logger.info("Comparing with champion model")
        
        if self._model_metrics is None:
            raise TrainingPipelineException("Metrics not computed. Call compute_metrics() first.")
        
        try:
            # Use accuracy as the comparison metric
            comparison = self.mlops_manager.compare_with_champion(
                model_name=model_name,
                challenger_version=f"{model_name}_v{int(time.time())}",  # Temp version name
                metric_name='accuracy',
                champion_tag_name='stage',
                champion_tag_value='champion',
                higher_is_better=True
            )
            
            self._comparison_result = comparison
            logger.info(f"Comparison complete: {comparison.get('recommendation')}")
            
            return comparison
            
        except Exception as e:
            logger.warning(f"Comparison with champion failed: {str(e)}")
            # Return neutral comparison result
            return {
                "status": "no_champion",
                "recommendation": "promote_challenger",
                "message": "No champion to compare against"
            }
    
    def register_model(self,
                      model_name: str,
                      request: TrainingPipelineRequest,
                      version_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Register trained model in Snowflake registry with metrics and metadata.
        
        Args:
            model_name: Name for the model in registry
            request: TrainingPipelineRequest with snapshot_date and metadata
            version_name: Optional version name (auto-generated if not provided)
            
        Returns:
            Registration result dictionary
        """
        logger.info(f"Registering model: {model_name}")
        
        if self._trained_model is None or self._model_metrics is None:
            raise TrainingPipelineException("Model not trained or metrics not computed")
        
        try:
            # Generate version name
            if not version_name:
                version_name = f"v{int(time.time())}"
            
            # Use MLOps manager to log and register the model
            metrics = self.mlops_manager.log_run(
                run_name=f"{model_name}_{version_name}",
                model=self._trained_model,
                X_train=self._X_train,
                y_train=self._y_train,
                X_test=self._X_test,
                y_test=self._y_test,
                params={},
                metrics_fn=lambda y_true, y_pred: self._model_metrics,
                log_model=True,
                model_name=model_name,
                version_name=version_name,
                save_metrics_to_registry=True,
                tags={'stage': 'challenger'} if request.tags is None else request.tags
            )
            
            # Log metadata (snapshot_date)
            metadata = {
                'snapshot_date': request.snapshot_date.isoformat(),
                'feature_view': request.feature_view_name,
                'description': request.description or 'Training pipeline run'
            }
            
            self.mlops_manager.log_model_metadata(
                model_name=model_name,
                version_name=version_name,
                metadata=metadata
            )
            
            logger.info(f"Model registered: {model_name} version {version_name}")
            
            return {
                "status": "registered",
                "model_name": model_name,
                "version_name": version_name,
                "metrics": metrics,
            }
            
        except Exception as e:
            raise TrainingPipelineException(f"Failed to register model: {str(e)}")
    
    def run_pipeline(self, request: TrainingPipelineRequest) -> TrainingPipelineResponse:
        """
        Execute the complete training pipeline in order.
        
        Steps:
        1. Load features from feature store
        2. Get champion model parameters
        3. Prepare train/test data
        4. Train model (with optional tuning)
        5. Compute metrics
        6. Compare with champion
        7. Register model
        
        Args:
            request: TrainingPipelineRequest with all pipeline parameters
            
        Returns:
            TrainingPipelineResponse with results
        """
        logger.info(f"Starting training pipeline for model: {request.model_name}")
        start_time = time.time()
        
        try:
            # Step 1: Get features
            self.get_features(request)
            
            # Step 2: Get champion parameters
            champion_info = self.get_champion_model(request.model_name)
            
            # Step 3: Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(request)
            
            # Step 4: Train model
            self.train_model(request, champion_info['parameters'])
            
            # Step 5: Compute metrics
            metrics = self.compute_metrics()
            
            # Step 6: Compare with champion
            comparison = self.compare_with_champion(request.model_name, request)
            
            # Step 7: Register model
            version_name = f"{request.model_name}_v{int(time.time() * 1000)}"
            registration = self.register_model(
                request.model_name,
                request,
                version_name=version_name
            )
            
            elapsed_time = time.time() - start_time
            
            # Build response
            response_data = {
                "model_name": request.model_name,
                "trained_version": version_name,
                "snapshot_date": request.snapshot_date.isoformat(),
                "metrics": metrics,
                "comparison_result": comparison,
                "training_time_seconds": round(elapsed_time, 2),
                "features_count": len(self._X_train.columns) if self._X_train is not None else 0,
                "training_samples": len(self._X_train) if self._X_train is not None else 0,
                "test_samples": len(self._X_test) if self._X_test is not None else 0,
            }
            
            logger.info(f"Pipeline completed successfully in {elapsed_time:.2f}s")
            
            return TrainingPipelineResponse(
                status="success",
                data=response_data,
                timestamp=datetime.utcnow()
            )
            
        except TrainingPipelineException as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return TrainingPipelineResponse(
                status="failed",
                error=str(e),
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Unexpected error in pipeline: {str(e)}")
            return TrainingPipelineResponse(
                status="failed",
                error=f"Unexpected error: {str(e)}",
                timestamp=datetime.utcnow()
            )