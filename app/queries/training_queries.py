CREATE_TRAINING_LOG_TABLE = """CREATE TABLE IF NOT EXISTS model_training_log (
    uuid STRING DEFAULT UUID_STRING(),
    created_at TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
    model_name STRING,
    version_name STRING,
    metrics VARIANT,
    training_parameters VARIANT,
    champion_version_name STRING,
    champion_metrics VARIANT,
    comparison_metrics VARIANT,
    potential_challenger_promoted BOOLEAN DEFAULT FALSE,
    comment STRING
    )"""


INSERT_TRAINING_LOG = """INSERT INTO {table_name}
    (MODEL_NAME, VERSION_NAME, METRICS, TRAINING_PARAMETERS, 
    CHAMPION_VERSION_NAME, CHAMPION_METRICS, COMPARISON_METRICS,
    POTENTIAL_CHALLENGER_PROMOTED, COMMENT)
    SELECT ?, ?, PARSE_JSON(?), PARSE_JSON(?), ?, PARSE_JSON(?), PARSE_JSON(?), ?, ?"""

