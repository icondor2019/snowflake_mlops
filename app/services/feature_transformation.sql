-- Feature Transformation SQL
-- Replicates feature_store_service.py logic in pure SQL
-- Full refresh ELT pattern
-- All columns standardized to UPPERCASE

USE ROLE ai_cortex_developer_role;
USE DATABASE ai_project;
USE SCHEMA raw;

CREATE SCHEMA IF NOT EXISTS ai_project.features;

-- ============================================
-- STEP 1: Load POI coordinates from JSON stage
-- ============================================
CREATE OR REPLACE TEMPORARY TABLE poi_coordinates AS
SELECT 
    f.key AS CATEGORY,
    p.value[0]::FLOAT AS POI_LAT,
    p.value[1]::FLOAT AS POI_LON
FROM @raw_data_stage/point_interest_coordinates.json (FILE_FORMAT => 'json_format'),
LATERAL FLATTEN(input => $1) f,
LATERAL FLATTEN(input => f.value) p;

-- Check POI categories
SELECT DISTINCT CATEGORY FROM poi_coordinates;

-- ============================================
-- STEP 2: Get hex centroids (H3 cell to lat/lon)
-- ============================================
CREATE OR REPLACE TEMPORARY TABLE train_with_coords AS
SELECT 
    t."hex_id" AS HEX_ID,
    t."cost_of_living" AS COST_OF_LIVING,
    ST_Y(H3_CELL_TO_POINT(t."hex_id")) AS LAT,
    ST_X(H3_CELL_TO_POINT(t."hex_id")) AS LON
FROM ai_project.raw.train t;

-- ============================================
-- STEP 3: Calculate minimum distance to each POI category
-- ============================================
CREATE OR REPLACE TEMPORARY TABLE train_with_distances AS
SELECT 
    t.*,
    p.CATEGORY,
    HAVERSINE(t.LAT, t.LON, p.POI_LAT, p.POI_LON) AS DISTANCE_KM
FROM train_with_coords t
CROSS JOIN poi_coordinates p;

select * from train_with_distances limit 11;

-- ============================================
-- STEP 4: Pivot to get min distance per category
-- ============================================
CREATE OR REPLACE TABLE ai_project.features.train_features AS
WITH min_distances AS (
    SELECT 
        HEX_ID,
        COST_OF_LIVING,
        LAT,
        LON,
        CATEGORY,
        MIN(DISTANCE_KM) AS MIN_DISTANCE
    FROM train_with_distances
    GROUP BY HEX_ID, COST_OF_LIVING, LAT, LON, CATEGORY
),
pivoted AS (
    SELECT 
        HEX_ID,
        COST_OF_LIVING,
        LAT,
        LON,
        MAX(CASE WHEN CATEGORY = 'supermarket' THEN MIN_DISTANCE END) AS DIST_TO_SUPERMARKET,
        MAX(CASE WHEN CATEGORY = 'hospital' THEN MIN_DISTANCE END) AS DIST_TO_HOSPITAL,
        MAX(CASE WHEN CATEGORY = 'school' THEN MIN_DISTANCE END) AS DIST_TO_SCHOOL,
        MAX(CASE WHEN CATEGORY = 'park' THEN MIN_DISTANCE END) AS DIST_TO_PARK,
        MAX(CASE WHEN CATEGORY = 'restaurant' THEN MIN_DISTANCE END) AS DIST_TO_RESTAURANT,
        MAX(CASE WHEN CATEGORY = 'bank' THEN MIN_DISTANCE END) AS DIST_TO_BANK,
        MAX(CASE WHEN CATEGORY = 'cafe' THEN MIN_DISTANCE END) AS DIST_TO_CAFE,
        MAX(CASE WHEN CATEGORY = 'fuel' THEN MIN_DISTANCE END) AS DIST_TO_FUEL
    FROM min_distances
    GROUP BY HEX_ID, COST_OF_LIVING, LAT, LON
)
SELECT 
    p.HEX_ID,
    p.COST_OF_LIVING,
    p.LAT,
    p.LON,
    p.DIST_TO_SUPERMARKET,
    p.DIST_TO_HOSPITAL,
    p.DIST_TO_SCHOOL,
    p.DIST_TO_PARK,
    p.DIST_TO_RESTAURANT,
    p.DIST_TO_BANK,
    p.DIST_TO_CAFE,
    p.DIST_TO_FUEL,
    
    -- Hourly summary features
    h."total_hours" AS TOTAL_HOURS,
    h."unique_devices" AS UNIQUE_DEVICES,
    h."unique_days" AS UNIQUE_DAYS,
    h."work_h1" AS WORK_H1,
    h."work_h2" AS WORK_H2,
    h."work_h3" AS WORK_H3,
    h."work_h4" AS WORK_H4,
    h."wn_h1" AS WN_H1,
    h."wn_h2" AS WN_H2,
    h."wn_h3" AS WN_H3,
    h."wn_h4" AS WN_H4,
    
    -- POI count features
    poi."bank" AS BANK,
    poi."atm" AS ATM,
    poi."cafe" AS CAFE,
    poi."restaurant" AS RESTAURANT,
    poi."post_office" AS POST_OFFICE,
    poi."fuel" AS FUEL,
    poi."education" AS EDUCATION,
    poi."fire_station" AS FIRE_STATION,
    poi."negative" AS NEGATIVE,
    poi."cinema" AS CINEMA,
    poi."internet_cafe" AS INTERNET_CAFE,
    poi."doctors" AS DOCTORS,
    poi."car_wash" AS CAR_WASH,
    poi."police" AS POLICE,
    poi."supermaxi" AS SUPERMAXI,
    poi."bco_pichincha" AS BCO_PICHINCHA,
    
    CURRENT_TIMESTAMP() AS REFRESHED_AT

FROM pivoted p
LEFT JOIN ai_project.raw.h8_summary_hours h ON p.HEX_ID = h."hex_id"
LEFT JOIN ai_project.raw.points_of_interest_aggregated poi ON p.HEX_ID = poi."hex_id";

-- ============================================
-- STEP 5: Create TEST features (same logic)
-- ============================================
CREATE OR REPLACE TEMPORARY TABLE test_with_coords AS
SELECT 
    t."hex_id" AS HEX_ID,
    t."cost_of_living" AS COST_OF_LIVING,
    ST_Y(H3_CELL_TO_POINT(t."hex_id")) AS LAT,
    ST_X(H3_CELL_TO_POINT(t."hex_id")) AS LON
FROM ai_project.raw.test t;

CREATE OR REPLACE TEMPORARY TABLE test_with_distances AS
SELECT 
    t.*,
    p.CATEGORY,
    HAVERSINE(t.LAT, t.LON, p.POI_LAT, p.POI_LON) AS DISTANCE_KM
FROM test_with_coords t
CROSS JOIN poi_coordinates p;

CREATE OR REPLACE TABLE ai_project.features.test_features AS
WITH min_distances AS (
    SELECT 
        HEX_ID,
        COST_OF_LIVING,
        LAT,
        LON,
        CATEGORY,
        MIN(DISTANCE_KM) AS MIN_DISTANCE
    FROM test_with_distances
    GROUP BY HEX_ID, COST_OF_LIVING, LAT, LON, CATEGORY
),
pivoted AS (
    SELECT 
        HEX_ID,
        COST_OF_LIVING,
        LAT,
        LON,
        MAX(CASE WHEN CATEGORY = 'supermarket' THEN MIN_DISTANCE END) AS DIST_TO_SUPERMARKET,
        MAX(CASE WHEN CATEGORY = 'hospital' THEN MIN_DISTANCE END) AS DIST_TO_HOSPITAL,
        MAX(CASE WHEN CATEGORY = 'school' THEN MIN_DISTANCE END) AS DIST_TO_SCHOOL,
        MAX(CASE WHEN CATEGORY = 'park' THEN MIN_DISTANCE END) AS DIST_TO_PARK,
        MAX(CASE WHEN CATEGORY = 'restaurant' THEN MIN_DISTANCE END) AS DIST_TO_RESTAURANT,
        MAX(CASE WHEN CATEGORY = 'bank' THEN MIN_DISTANCE END) AS DIST_TO_BANK,
        MAX(CASE WHEN CATEGORY = 'cafe' THEN MIN_DISTANCE END) AS DIST_TO_CAFE,
        MAX(CASE WHEN CATEGORY = 'fuel' THEN MIN_DISTANCE END) AS DIST_TO_FUEL
    FROM min_distances
    GROUP BY HEX_ID, COST_OF_LIVING, LAT, LON
)
SELECT 
    p.HEX_ID,
    p.COST_OF_LIVING,
    p.LAT,
    p.LON,
    p.DIST_TO_SUPERMARKET,
    p.DIST_TO_HOSPITAL,
    p.DIST_TO_SCHOOL,
    p.DIST_TO_PARK,
    p.DIST_TO_RESTAURANT,
    p.DIST_TO_BANK,
    p.DIST_TO_CAFE,
    p.DIST_TO_FUEL,
    
    -- Hourly summary features
    h."total_hours" AS TOTAL_HOURS,
    h."unique_devices" AS UNIQUE_DEVICES,
    h."unique_days" AS UNIQUE_DAYS,
    h."work_h1" AS WORK_H1,
    h."work_h2" AS WORK_H2,
    h."work_h3" AS WORK_H3,
    h."work_h4" AS WORK_H4,
    h."wn_h1" AS WN_H1,
    h."wn_h2" AS WN_H2,
    h."wn_h3" AS WN_H3,
    h."wn_h4" AS WN_H4,
    
    -- POI count features
    poi."bank" AS BANK,
    poi."atm" AS ATM,
    poi."cafe" AS CAFE,
    poi."restaurant" AS RESTAURANT,
    poi."post_office" AS POST_OFFICE,
    poi."fuel" AS FUEL,
    poi."education" AS EDUCATION,
    poi."fire_station" AS FIRE_STATION,
    poi."negative" AS NEGATIVE,
    poi."cinema" AS CINEMA,
    poi."internet_cafe" AS INTERNET_CAFE,
    poi."doctors" AS DOCTORS,
    poi."car_wash" AS CAR_WASH,
    poi."police" AS POLICE,
    poi."supermaxi" AS SUPERMAXI,
    poi."bco_pichincha" AS BCO_PICHINCHA,
    
    CURRENT_TIMESTAMP() AS REFRESHED_AT

FROM pivoted p
LEFT JOIN ai_project.raw.h8_summary_hours h ON p.HEX_ID = h."hex_id"
LEFT JOIN ai_project.raw.points_of_interest_aggregated poi ON p.HEX_ID = poi."hex_id";

-- ============================================
-- VERIFY
-- ============================================
SELECT 'TRAIN' AS DATASET, COUNT(*) AS ROW_COUNT FROM ai_project.features.train_features
UNION ALL
SELECT 'TEST' AS DATASET, COUNT(*) AS ROW_COUNT FROM ai_project.features.test_features;

SELECT * FROM ai_project.features.train_features LIMIT 10;
