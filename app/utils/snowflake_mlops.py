from snowflake.ml.experiment import ExperimentTracking
from snowflake.ml.model.model_signature import infer_signature
from snowflake.ml.registry import Registry
from snowflake.ml.feature_store import FeatureStore
from typing import Optional, Dict, Any, List
from loguru import logger
from datetime import datetime


class SnowflakeMLOpsManager:
    def __init__(self, session, database: str = None, schema: str = None):
        """
        session: Snowpark session
        experiment_name: nombre del experimento
        database: database for registry (optional, uses session's current)
        schema: schema for registry (optional, uses session's current)
        """
        self.session = session
        self.database = database
        self.schema = schema
        self.registry = Registry(
            session=session,
            database_name=database,
            schema_name=schema
        )

    def log_run(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        params: dict,
        metrics_fn: callable,
        log_model: bool = False,
        model_name: str = None,
        version_name: str = None,
        alias: str = None,
        experiment_name: str = None,
        run_name: str = None
        ):
        """
        Execute the entire ML workflow and log everything in one go.

        params: hyperparameters to log
        metrics_fn: function that takes (y_true, y_pred) and returns a dict of metrics to log
        log_model: if True, the trained model will be logged in Snowflake
        model_name: name for the model in registry
        version_name: version name (auto-generated if None)
        save_metrics_to_registry: if True, saves metrics to model version for comparison
        alias: alias to apply to the model version, only to promote champions
        """
        self.exp = ExperimentTracking(session=self.session)
        run_context = f"{model_name}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}"

        if experiment_name is None:
            experiment_name = f"experiment_{run_context}"
        self.exp.set_experiment(experiment_name)

        if run_name is None:
            run_name = f"run_{run_context}"

        with self.exp.start_run(run_name):
            self.exp.log_params(params)

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            metrics = metrics_fn(y_test, y_pred)
            self.exp.log_metrics(metrics)

            if log_model:
                sig = infer_signature(X_train, y_train)
                mv = self.exp.log_model(
                    model,
                    model_name=model_name,
                    version_name=version_name,
                    signatures={"predict": sig},
                    metrics=metrics,
                )
                reg_version = mv.version_name
                reg_model_name = mv.model_name
                logger.debug(f"Registeed_model: name: {reg_model_name}, version:{reg_version}")
                if alias:
                    self.session.sql(f"""
                                    ALTER MODEL AI_PROJECT.MLOPS.{reg_model_name} VERSION {reg_version} SET ALIAS = {alias}
                                """).collect()
            return mv

    def get_model_by_version(
        self,
        model_name: str,
        version: str = "champion",
        ) -> Optional[str]:
        """Get a specific model version object from registry by name and version or alias."""
        model_ref = self.registry.get_model(model_name)
        mv = model_ref.version("champion")
        logger.debug(f"Champion model version: {mv.version_name}")
        return mv

    def get_feature_store_view(self, 
                                  feature_vw_name: str, 
                                  version: str = "v1",
                                  warehouse: str = 'ai_project_wh',
                                  limit: int = None):
        """Get feature view data as pandas dataframe.
        
        Args:
            feature_vw_name: Feature view name
            version: Feature view version
            limit: Maximum number of rows to return
            
        Returns:
            Pandas DataFrame with feature data
        """

        fs = FeatureStore(
            session=self.session, 
            database="AI_PROJECT", 
            name="FEATURES",
            default_warehouse=warehouse
        )
        
        fv = fs.get_feature_view(name=feature_vw_name, version=version)
        if limit:
            df = fv.feature_df.limit(limit).to_pandas()
        else:
            df = fv.feature_df.to_pandas()

        return df

    def create_tag(self, model_name: str, tag_name: str, tag_value: str) -> Dict[str, Any]:
        """Create or update a tag on a model in the registry.
        
        Args:
            model_name: Name of the model
            tag_name: Tag name/key
            tag_value: Tag value
            
        Returns:
            Dictionary with tag creation status
        """
        model_ref = self.registry.get_model(model_name)
        model_ref.set_tag(tag_name, tag_value)
        
        return {
            "status": "created",
            "model_name": model_name,
            "tag_name": tag_name,
            "tag_value": tag_value,
        }

    def log_model_metadata(self, 
                          model_name: str, 
                          version_name: str, 
                          metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Log custom metadata to a model version (e.g., snapshot_date, data_version).
        
        Args:
            model_name: Name of the model
            version_name: Version to tag with metadata
            metadata: Dictionary of metadata key-value pairs to log
            
        Returns:
            Dictionary with metadata logging status
        """
        model_ref = self.registry.get_model(model_name)
        model_version = model_ref.version(version_name)
        
        # Log metadata as tags with 'meta_' prefix
        for key, value in metadata.items():
            tag_key = f"meta_{key}"
            model_version.set_tag(tag_key, str(value))
        
        return {
            "status": "logged",
            "model_name": model_name,
            "version_name": version_name,
            "metadata_keys": list(metadata.keys()),
        }