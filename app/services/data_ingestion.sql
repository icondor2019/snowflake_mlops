use role ai_cortex_developer_role;
use database ai_project;

CREATE SCHEMA IF NOT EXISTS AI_PROJECT.RAW;
USE SCHEMA AI_PROJECT.RAW;

-- File format for INFER_SCHEMA (needs PARSE_HEADER)
CREATE OR REPLACE FILE FORMAT csv_format_infer
    TYPE = 'CSV'
    FIELD_DELIMITER = ','
    PARSE_HEADER = TRUE
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    NULL_IF = ('', 'NULL', 'null')
    EMPTY_FIELD_AS_NULL = TRUE
    TRIM_SPACE = TRUE;

-- File format for COPY INTO (needs PARSE_HEADER for MATCH_BY_COLUMN_NAME)
CREATE OR REPLACE FILE FORMAT csv_format_load
    TYPE = 'CSV'
    FIELD_DELIMITER = ','
    PARSE_HEADER = TRUE
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    NULL_IF = ('', 'NULL', 'null')
    EMPTY_FIELD_AS_NULL = TRUE
    TRIM_SPACE = TRUE;

-- File format for JSON files
CREATE OR REPLACE FILE FORMAT json_format
    TYPE = 'JSON'
    STRIP_OUTER_ARRAY = TRUE
    ALLOW_DUPLICATE = FALSE
    STRIP_NULL_VALUES = FALSE
    IGNORE_UTF8_ERRORS = FALSE;

-- Internal stage for data ingestion
CREATE OR REPLACE STAGE raw_data_stage
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for raw data ingestion (CSV and JSON files)';

-- Check files in the STAGE
LIST @ai_project.raw.raw_data_stage;

-- ============================================
-- Create tables from staged CSV files
-- ============================================

-- 1. H8_SUMMARY_HOURS table
SELECT * FROM TABLE(
    INFER_SCHEMA(
        LOCATION => '@raw_data_stage/h8_summary_hours.csv',
        FILE_FORMAT => 'csv_format_infer'
    )
);

CREATE OR REPLACE TABLE h8_summary_hours
    USING TEMPLATE (
        SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))
        FROM TABLE(
            INFER_SCHEMA(
                LOCATION => '@raw_data_stage/h8_summary_hours.csv',
                FILE_FORMAT => 'csv_format_infer'
            )
        )
    );

COPY INTO h8_summary_hours 
    FROM @raw_data_stage/h8_summary_hours.csv 
    FILE_FORMAT = csv_format_load
    MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;

-- 2. POINTS_OF_INTEREST_AGGREGATED table
SELECT * FROM TABLE(
    INFER_SCHEMA(
        LOCATION => '@raw_data_stage/points_of_interest_aggregated.csv',
        FILE_FORMAT => 'csv_format_infer'
    )
);

CREATE OR REPLACE TABLE points_of_interest_aggregated
    USING TEMPLATE (
        SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))
        FROM TABLE(
            INFER_SCHEMA(
                LOCATION => '@raw_data_stage/points_of_interest_aggregated.csv',
                FILE_FORMAT => 'csv_format_infer'
            )
        )
    );

COPY INTO points_of_interest_aggregated 
    FROM @raw_data_stage/points_of_interest_aggregated.csv 
    FILE_FORMAT = csv_format_load
    MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;

-- 3. TEST table
SELECT * FROM TABLE(
    INFER_SCHEMA(
        LOCATION => '@raw_data_stage/test.csv',
        FILE_FORMAT => 'csv_format_infer'
    )
);

CREATE OR REPLACE TABLE test
    USING TEMPLATE (
        SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))
        FROM TABLE(
            INFER_SCHEMA(
                LOCATION => '@raw_data_stage/test.csv',
                FILE_FORMAT => 'csv_format_infer'
            )
        )
    );

COPY INTO test 
    FROM @raw_data_stage/test.csv 
    FILE_FORMAT = csv_format_load
    MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;

-- 4. TRAIN table
SELECT * FROM TABLE(
    INFER_SCHEMA(
        LOCATION => '@raw_data_stage/train.csv',
        FILE_FORMAT => 'csv_format_infer'
    )
);

CREATE OR REPLACE TABLE train
    USING TEMPLATE (
        SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))
        FROM TABLE(
            INFER_SCHEMA(
                LOCATION => '@raw_data_stage/train.csv',
                FILE_FORMAT => 'csv_format_infer'
            )
        )
    );

COPY INTO train 
    FROM @raw_data_stage/train.csv 
    FILE_FORMAT = csv_format_load
    MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;

-- Verify tables created
SHOW TABLES IN SCHEMA AI_PROJECT.RAW;

select * from ai_project.features.train_features