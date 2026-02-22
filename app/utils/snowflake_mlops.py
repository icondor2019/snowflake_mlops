from snowflake.ml.experiment import ExperimentTracking
from snowflake.ml.model.model_signature import infer_signature
from snowflake.ml.registry import Registry
from typing import Optional, Dict, Any, List


class SnowflakeMLOpsManager:
    def __init__(self, session, experiment_name: str, database: str = None, schema: str = None):
        """
        session: Snowpark session
        experiment_name: nombre del experimento
        database: database for registry (optional, uses session's current)
        schema: schema for registry (optional, uses session's current)
        """
        self.session = session
        self.database = database
        self.schema = schema
        self.exp = ExperimentTracking(session=session)
        self.exp.set_experiment(experiment_name)
        self.registry = Registry(
            session=session,
            database_name=database,
            schema_name=schema
        )

    def log_run(
        self,
        run_name: str,
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
        save_metrics_to_registry: bool = True,
        tags: Dict[str, str] = None,
    ):
        """
        Execute the entire ML workflow and log everything in one go.

        params: hyperparameters to log
        metrics_fn: function that takes (y_true, y_pred) and returns a dict of metrics to log
        log_model: if True, the trained model will be logged in Snowflake
        model_name: name for the model in registry
        version_name: version name (auto-generated if None)
        save_metrics_to_registry: if True, saves metrics to model version for comparison
        tags: dict of tags to apply to the model (e.g., {"stage": "challenger"})
        """

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
                    model_name=model_name or "model",
                    version_name=version_name,
                    signatures={"predict": sig},
                    metrics=metrics if save_metrics_to_registry else None,
                )

                if tags and mv:
                    model_ref = self.registry.get_model(model_name or "model")
                    for tag_name, tag_value in tags.items():
                        model_ref.set_tag(tag_name, tag_value)

            return metrics

    def compare_with_champion(
        self,
        model_name: str,
        challenger_version: str,
        metric_name: str,
        champion_tag_name: str = "stage",
        champion_tag_value: str = "champion",
        higher_is_better: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare a challenger model version against the champion.
        
        Args:
            model_name: name of the model in registry
            challenger_version: version name of the challenger model
            metric_name: metric to compare (e.g., "accuracy", "f1_score")
            champion_tag_name: tag name used to identify champion
            champion_tag_value: tag value for champion
            higher_is_better: True if higher metric is better (accuracy), 
                              False if lower is better (rmse, mae)
        
        Returns:
            dict with comparison results and recommendation
        """
        model_ref = self.registry.get_model(model_name)
        
        champion_version = None
        for version in model_ref.versions():
            version_name = version.version_name
            try:
                mv = model_ref.version(version_name)
                metrics = mv.show_metrics()
                if metrics.get("_tags", {}).get(champion_tag_name) == champion_tag_value:
                    champion_version = version_name
                    break
            except:
                pass
        
        if not champion_version:
            tags = model_ref.show_tags()
            if tags.get(champion_tag_name) == champion_tag_value:
                champion_version = model_ref.default.version_name
        
        if not champion_version:
            return {
                "status": "no_champion",
                "message": f"No champion found with tag {champion_tag_name}={champion_tag_value}",
                "recommendation": "promote_challenger",
                "challenger_version": challenger_version,
            }
        
        champion_mv = model_ref.version(champion_version)
        challenger_mv = model_ref.version(challenger_version)
        
        champion_metrics = champion_mv.show_metrics()
        challenger_metrics = challenger_mv.show_metrics()
        
        champion_score = champion_metrics.get(metric_name)
        challenger_score = challenger_metrics.get(metric_name)
        
        if champion_score is None or challenger_score is None:
            return {
                "status": "error",
                "message": f"Metric '{metric_name}' not found in one or both versions",
                "champion_metrics": champion_metrics,
                "challenger_metrics": challenger_metrics,
            }
        
        if higher_is_better:
            is_better = challenger_score > champion_score
            improvement = challenger_score - champion_score
            improvement_pct = (improvement / champion_score * 100) if champion_score != 0 else 0
        else:
            is_better = challenger_score < champion_score
            improvement = champion_score - challenger_score
            improvement_pct = (improvement / champion_score * 100) if champion_score != 0 else 0
        
        return {
            "status": "compared",
            "champion_version": champion_version,
            "challenger_version": challenger_version,
            "metric_name": metric_name,
            "champion_score": champion_score,
            "challenger_score": challenger_score,
            "improvement": improvement,
            "improvement_pct": round(improvement_pct, 2),
            "challenger_is_better": is_better,
            "recommendation": "promote_challenger" if is_better else "keep_champion",
            "champion_metrics": champion_metrics,
            "challenger_metrics": challenger_metrics,
        }

    def promote_to_champion(
        self,
        model_name: str,
        version_name: str,
        champion_tag_name: str = "stage",
        champion_tag_value: str = "champion",
        set_as_default: bool = True,
    ) -> Dict[str, Any]:
        """
        Promote a model version to champion status.
        
        Args:
            model_name: name of the model
            version_name: version to promote
            champion_tag_name: tag name for champion status
            champion_tag_value: tag value for champion
            set_as_default: also set this version as default
        
        Returns:
            dict with promotion result
        """
        model_ref = self.registry.get_model(model_name)
        
        model_ref.set_tag(champion_tag_name, champion_tag_value)
        
        if set_as_default:
            model_ref.default = version_name
        
        return {
            "status": "promoted",
            "model_name": model_name,
            "version_name": version_name,
            "tag": f"{champion_tag_name}={champion_tag_value}",
            "is_default": set_as_default,
        }

    def get_champion_version(
        self,
        model_name: str,
        champion_tag_name: str = "stage",
        champion_tag_value: str = "champion",
    ) -> Optional[str]:
        """Get the current champion version name."""
        model_ref = self.registry.get_model(model_name)
        tags = model_ref.show_tags()
        if tags.get(champion_tag_name) == champion_tag_value:
            return model_ref.default.version_name
        return None
