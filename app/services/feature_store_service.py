import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F
import json


class FeatureStoreService:
    
    POI_STAGE_PATH = "@AI_PROJECT.RAW.raw_data_stage/point_interest_coordinates.json"
    
    def __init__(
        self,
        session: Session,
        database: str = "AI_PROJECT",
        schema: str = "RAW",
    ):
        if not session:
            raise ValueError("Snowflake session is required")
        
        self.session = session
        self.database = database
        self.schema = schema
        
        self._raw_data: pd.DataFrame = None
        self._features: pd.DataFrame = None
        self._target: pd.Series = None
        self._feature_names: List[str] = []
        self._poi_data: Dict[str, Any] = None

    def load_data(
        self,
        table_name: str,
        target_column: str = None,
        filters: Dict[str, Any] = None,
    ) -> pd.DataFrame:
        full_table = f"{self.database}.{self.schema}.{table_name}"
        df = self.session.table(full_table)
        
        if filters:
            for col, val in filters.items():
                df = df.filter(F.col(col) == val)
        
        self._raw_data = df.to_pandas()
        self._raw_data.columns = self._raw_data.columns.str.lower()
        
        if target_column and target_column.lower() in self._raw_data.columns:
            self._target = self._raw_data[target_column.lower()]
        
        return self._raw_data

    def load_poi_data(self, stage_path: str = None) -> None:
        poi_path = stage_path or self.POI_STAGE_PATH
        
        result = self.session.sql(f"""
            SELECT $1 as data 
            FROM {poi_path} (FILE_FORMAT => 'json_format')
        """).collect()
        
        if result:
            self._poi_data = json.loads(result[0]['DATA'])
        else:
            raise ValueError(f"Could not load POI data from {poi_path}")

    def create_temporal_features(
        self,
        datetime_column: str,
        features: List[str] = None,
    ) -> pd.DataFrame:
        if self._raw_data is None:
            raise ValueError("Load data first using load_data")
        
        df = self._raw_data.copy()
        col = datetime_column.lower() if datetime_column.lower() in df.columns else datetime_column
        
        if col not in df.columns:
            raise ValueError(f"Column {datetime_column} not found in data")
        
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])
        
        default_features = ['hour', 'day_of_week', 'month', 'is_weekend', 'quarter', 'day_of_month']
        features_to_create = features or default_features
        
        if 'hour' in features_to_create:
            df['hour'] = df[col].dt.hour
        if 'day_of_week' in features_to_create:
            df['day_of_week'] = df[col].dt.dayofweek
        if 'month' in features_to_create:
            df['month'] = df[col].dt.month
        if 'is_weekend' in features_to_create:
            df['is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
        if 'quarter' in features_to_create:
            df['quarter'] = df[col].dt.quarter
        if 'day_of_month' in features_to_create:
            df['day_of_month'] = df[col].dt.day
        if 'week_of_year' in features_to_create:
            df['week_of_year'] = df[col].dt.isocalendar().week
        if 'is_month_start' in features_to_create:
            df['is_month_start'] = df[col].dt.is_month_start.astype(int)
        if 'is_month_end' in features_to_create:
            df['is_month_end'] = df[col].dt.is_month_end.astype(int)
        
        self._raw_data = df
        return df

    def create_distance_features(
        self,
        lat_column: str,
        lon_column: str,
        poi_categories: List[str] = None,
    ) -> pd.DataFrame:
        if self._raw_data is None:
            raise ValueError("Load data first")
        if self._poi_data is None:
            raise ValueError("Load POI data first using load_poi_data")
        
        from app.utils.geospatial_tools import haversine_distance
        
        df = self._raw_data.copy()
        lat_col = lat_column.lower() if lat_column.lower() in df.columns else lat_column
        lon_col = lon_column.lower() if lon_column.lower() in df.columns else lon_column
        
        categories = poi_categories or list(self._poi_data.keys())
        
        for category in categories:
            if category not in self._poi_data:
                continue
            
            poi_coords = self._poi_data[category]
            col_name = f'dist_to_{category.lower().replace(" ", "_")}'
            
            distances = []
            for _, row in df.iterrows():
                min_dist = float('inf')
                for poi in poi_coords:
                    dist = haversine_distance(
                        row[lat_col], row[lon_col],
                        poi['lat'], poi['lon']
                    )
                    min_dist = min(min_dist, dist)
                distances.append(min_dist)
            
            df[col_name] = distances
        
        self._raw_data = df
        return df

    def create_aggregation_features(
        self,
        group_columns: List[str],
        agg_column: str,
        agg_functions: List[str] = None,
    ) -> pd.DataFrame:
        if self._raw_data is None:
            raise ValueError("Load data first")
        
        df = self._raw_data.copy()
        agg_funcs = agg_functions or ['mean', 'std', 'min', 'max', 'count']
        
        for func in agg_funcs:
            col_name = f"{agg_column}_{func}_by_{'_'.join(group_columns)}"
            agg_values = df.groupby(group_columns)[agg_column].transform(func)
            df[col_name] = agg_values
        
        self._raw_data = df
        return df

    def prepare_feature_set(
        self,
        feature_columns: List[str],
        target_column: str = None,
    ) -> tuple:
        if self._raw_data is None:
            raise ValueError("Load data first")
        
        df = self._raw_data.copy()
        features = [c.lower() if c.lower() in df.columns else c for c in feature_columns]
        features = [c for c in features if c in df.columns]
        
        X = df[features].fillna(0)
        self._features = X
        self._feature_names = features
        
        if target_column:
            target_col = target_column.lower() if target_column.lower() in df.columns else target_column
            y = df[target_col] if target_col in df.columns else None
            self._target = y
            return X, y
        
        return X, None

    def save_to_snowflake(
        self,
        table_name: str,
        output_schema: str = None,
    ) -> str:
        if self._features is None:
            raise ValueError("Prepare features first using prepare_feature_set")
        
        target_schema = output_schema or self.schema
        full_table = f"{self.database}.{target_schema}.{table_name}"
        
        feature_df = self._features.copy()
        feature_df['created_at'] = datetime.now()
        
        if self._target is not None:
            feature_df['_target'] = self._target.values
        
        snow_df = self.session.create_dataframe(feature_df)
        snow_df.write.mode("overwrite").save_as_table(full_table)
        
        return full_table

    def get_features(self) -> pd.DataFrame:
        return self._features

    def get_target(self) -> pd.Series:
        return self._target

    def get_feature_names(self) -> List[str]:
        return self._feature_names

    def run_service(
        self,
        source_table: str,
        output_table: str,
        feature_columns: List[str],
        target_column: str = None,
        datetime_column: str = None,
        lat_column: str = None,
        lon_column: str = None,
        output_schema: str = None,
    ) -> Dict[str, Any]:
        result = {
            "status": "started",
            "steps_completed": [],
            "errors": [],
        }
        
        try:
            self.load_data(source_table, target_column)
            result["steps_completed"].append("data_loading")
        except Exception as e:
            result["errors"].append(f"Data loading failed: {str(e)}")
            result["status"] = "failed"
            return result
        
        try:
            if datetime_column:
                self.create_temporal_features(datetime_column)
                result["steps_completed"].append("temporal_features")
        except Exception as e:
            result["errors"].append(f"Temporal features failed: {str(e)}")
        
        try:
            if lat_column and lon_column:
                self.load_poi_data()
                self.create_distance_features(lat_column, lon_column)
                result["steps_completed"].append("distance_features")
        except Exception as e:
            result["errors"].append(f"Distance features failed: {str(e)}")
        
        try:
            self.prepare_feature_set(feature_columns, target_column)
            result["steps_completed"].append("prepare_features")
        except Exception as e:
            result["errors"].append(f"Prepare features failed: {str(e)}")
            result["status"] = "failed"
            return result
        
        try:
            table_created = self.save_to_snowflake(output_table, output_schema)
            result["steps_completed"].append("save_to_snowflake")
            result["table"] = table_created
        except Exception as e:
            result["errors"].append(f"Save to Snowflake failed: {str(e)}")
        
        result["status"] = "completed" if not result["errors"] else "completed_with_errors"
        result["feature_count"] = len(self._feature_names)
        result["row_count"] = len(self._features)
        
        return result
