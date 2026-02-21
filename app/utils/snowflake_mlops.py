from snowflake.ml.experiment import ExperimentTracking
from snowflake.ml.model.model_signature import infer_signature


class SnowflakeMLOpsManager:
    def __init__(self, session, experiment_name: str):
        """
        session: Snowpark session
        experiment_name: nombre del experimento
        """
        self.session = session
        self.exp = ExperimentTracking(session=session)
        self.exp.set_experiment(experiment_name)

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
    ):
        """
        Execute the entire ML workflow and log everything in one go.

        params: hyperparameters to log
        metrics_fn: function that takes (y_true, y_pred) and returns a dict of metrics to log
        log_model: if True, the trained model will be logged in Snowflake
        """

        with self.exp.start_run(run_name):
            # 1. Log params
            self.exp.log_params(params)

            # 2. Train
            model.fit(X_train, y_train)

            # 3. Predict
            y_pred = model.predict(X_test)

            # 4. Metrics
            metrics = metrics_fn(y_test, y_pred)
            self.exp.log_metrics(metrics)

            # 5. Optional: log model
            if log_model:
                sig = infer_signature(X_train, y_train)
                self.exp.log_model(
                    model,
                    model_name=model_name or "model",
                    signatures={"predict": sig},
                )

            return metrics
