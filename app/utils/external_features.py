"""
External Features Processing Module

Handles the extraction and aggregation of external geospatial features
from OpenStreetMap (OSM) points of interest data for predictive modeling.

Sections:
    2.2.1: Load and categorize Ecuadorian important points
    2.2.2: Extract coordinate lists from different POI types
    2.2.3: Calculate volume (count) of POIs per hex cell
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from loguru import logger


class ExternalFeaturesProcessor:
    """
    Processes external geospatial features from OSM points of interest data.
    
    Workflow:
        1. Load and categorize points of interest
        2. Extract coordinate lists for specific POI types
        3. Aggregate POI counts by H3 hex cells
    """
    
    def __init__(self):
        """Initialize POI category definitions."""
        self._define_amenity_categories()
        self._define_shop_categories()
    
    def _define_amenity_categories(self):
        """Define amenity type lists for filtering."""
        self.health_amenities = ["doctors", "veterinary"]
        self.security_amenities = ["police", "fire"]
        self.leisure_amenities = ["restaurant", "internet_cafe"]
        self.education_amenities = ["school", "college"]
        self.car_amenities = ["parking_entrance", "parking", "bus_station"]
        self.public_amenities = ["shelter", "post_office", "townhall", "marketplace"]
        self.negative_amenities = [
            "waste_disposal", "love_hotel", "prison", "gambling", 
            "stripclub", "sanitary_dump_station", "casino", "grave_yard"
        ]
        self.sea_amenities = [
            "boat_rental", "scuba_diving", "ferry_terminal", 
            "boat_storage", "dive_centre"
        ]
    
    def _define_shop_categories(self):
        """Define shop type lists for filtering."""
        self.groceries_and_food = [
            "bakery", "greengrocer", "butcher", "alcohol", "beverages",
            "convenience", "hardware", "department_store", "laundry"
        ]
        self.lux_shop = ["florist", "gift", "confectionery"]
        self.tech_shop = ["electronics", "mobile_phone", "beauty", "optician", "shoes"]
        self.cars_shop = ["car_parts", "tyres"]
        self.other_shop = ["yes"]
    
    # ========================================================================
    # 2.2.1: Load and categorize Ecuadorian important points
    # ========================================================================
    
    def load_and_prepare_points(self, geojson_path: str, csv_path: str = None) -> pd.DataFrame:
        """
        Load OSM points of interest and prepare base dataframe.
        
        Args:
            geojson_path: Path to OSM GeoJSON file
            csv_path: Optional path to pre-processed CSV file
        
        Returns:
            DataFrame with columns: amenity, shop, hex_id, geometry
        """
        try:
            # Load from GeoJSON
            gdf = gpd.read_file(geojson_path)
            
            # Assign H3 cell (resolution 8) to each point
            gdf["hex_id"] = gdf["geometry"].apply(
                lambda point: h3.latlng_to_cell(point.y, point.x, 8)
            )
            
            # Select relevant columns
            point_cols = ["amenity", "shop", "hex_id", "geometry"]
            df_points = gdf[point_cols].copy()
            
            logger.info(f"Loaded {len(df_points)} points of interest")
            return df_points
            
        except Exception as e:
            logger.error(f"Error loading points of interest: {e}")
            raise
    
    def classify_points_by_cost_zone(
        self, 
        points_df: pd.DataFrame, 
        train_df: pd.DataFrame,
        high_cost_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Classify points into cost zones based on training data distribution.
        
        Args:
            points_df: DataFrame with hex_id column
            train_df: Training data with hex_id and cost_of_living columns
            high_cost_threshold: Threshold for high cost classification
        
        Returns:
            DataFrame with cost_of_living column (1=high, 2=low, 0=unknown)
        """
        df = points_df.copy()
        
        # Identify high and low cost zones from training data
        high_cost_hex = train_df[train_df["cost_of_living"] > high_cost_threshold]["hex_id"].unique()
        low_cost_hex = train_df[train_df["cost_of_living"] <= high_cost_threshold]["hex_id"].unique()
        
        # Classify points
        df["cost_of_living"] = np.where(
            df["hex_id"].isin(high_cost_hex), 1,
            np.where(df["hex_id"].isin(low_cost_hex), 2, 0)
        )
        
        logger.info(
            f"Classified points: {(df['cost_of_living']==1).sum()} high-cost, "
            f"{(df['cost_of_living']==2).sum()} low-cost, "
            f"{(df['cost_of_living']==0).sum()} unknown"
        )
        return df
    
    # ========================================================================
    # 2.2.2: Extract list of coordinates from important points
    # ========================================================================
    
    def extract_coordinate_lists(self, geojson_path: str) -> dict:
        """
        Extract latitude-longitude coordinate tuples for different POI types.
        
        Args:
            geojson_path: Path to OSM GeoJSON file
        
        Returns:
            Dictionary with POI type names as keys and coordinate lists as values
        """
        try:
            gdf = gpd.read_file(geojson_path)
            
            # Convert geometry to (lat, lon) tuples
            gdf["lat_lon"] = gdf["geometry"].apply(lambda p: (p.y, p.x))
            
            coordinates = {}
            
            # Negative places (to avoid)
            negative_for_coordinates = ["waste_disposal", "prison"]
            coordinates["negative_points"] = gdf[
                gdf["amenity"].isin(negative_for_coordinates)
            ]["lat_lon"].tolist()
            
            # Supermarkets
            coordinates["supermaxi_points"] = gdf[
                gdf["name"].str.contains("supermaxi|megamaxi", case=False, na=False)
            ]["lat_lon"].tolist()
            
            # Car-related shops
            cars_shop_filter = ["car_parts", "tyres"]
            coordinates["car_points"] = gdf[
                gdf["shop"].isin(cars_shop_filter)
            ]["lat_lon"].tolist()
            
            # Education
            education_filter = ["university", "kindergarten"]
            coordinates["education_points"] = gdf[
                gdf["amenity"].isin(education_filter)
            ]["lat_lon"].tolist()
            
            # Transport
            transport_filter = ["bus_station", "parking", "taxi"]
            coordinates["transport_points"] = gdf[
                gdf["amenity"].isin(transport_filter)
            ]["lat_lon"].tolist()
            
            # Security
            security_filter = ["fire_station"]
            coordinates["security_points"] = gdf[
                gdf["amenity"].isin(security_filter)
            ]["lat_lon"].tolist()
            
            # Log extraction results
            for poi_type, coords in coordinates.items():
                logger.info(f"Extracted {len(coords)} {poi_type}")
            
            return coordinates
            
        except Exception as e:
            logger.error(f"Error extracting coordinate lists: {e}")
            raise
    
    # ========================================================================
    # 2.2.3: Calculate volume of important points per hex_id
    # ========================================================================
    
    def add_poi_binary_flags(self, points_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary indicator columns for different POI types.
        
        Args:
            points_df: DataFrame with 'amenity' and 'shop' columns
        
        Returns:
            DataFrame with binary indicator columns for each POI category
        """
        df = points_df.copy()
        
        # Shop-based indicators
        df["groceries_shop"] = (df["shop"].isin(self.groceries_and_food)).astype(int)
        df["lux_shop"] = (df["shop"].isin(self.lux_shop)).astype(int)
        df["tech_shop"] = (df["shop"].isin(self.tech_shop)).astype(int)
        df["car_shop"] = (df["shop"].isin(self.cars_shop)).astype(int)
        df["other_shop"] = (df["shop"].isin(self.other_shop)).astype(int)
        
        # Amenity-based indicators
        df["health"] = (df["amenity"].isin(self.health_amenities)).astype(int)
        df["security"] = (df["amenity"].isin(self.security_amenities)).astype(int)
        df["financial"] = (df["amenity"].isin(self.financial_amenities)).astype(int)
        df["leisure"] = (df["amenity"].isin(self.leisure_amenities)).astype(int)
        df["education"] = (df["amenity"].isin(self.education_amenities)).astype(int)
        df["cars"] = (df["amenity"].isin(self.car_amenities)).astype(int)
        df["negative"] = (df["amenity"].isin(self.negative_amenities)).astype(int)
        df["sea"] = (df["amenity"].isin(self.sea_amenities)).astype(int)
        
        return df
    
    def aggregate_poi_by_hex(self, points_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate POI counts by H3 hex cell.
        
        Args:
            points_df: DataFrame with hex_id and binary indicator columns
        
        Returns:
            DataFrame with POI counts per hex_id
        """
        # Define aggregation operations
        poi_columns = [
            "groceries_shop", "lux_shop", "tech_shop", "car_shop", "other_shop",
            "health", "security", "financial", "leisure", "education",
            "cars", "negative", "sea"
        ]
        
        operations = {col: "sum" for col in poi_columns}
        
        # Group by hex_id and sum
        df_aggregated = points_df.groupby("hex_id").agg(operations).reset_index()
        
        logger.info(
            f"Aggregated {len(points_df)} points into {len(df_aggregated)} hex cells"
        )
        return df_aggregated
    
    def process_external_features(
        self,
        geojson_path: str,
        train_df: pd.DataFrame,
        output_csv: str = None
    ) -> pd.DataFrame:
        """
        Complete processing pipeline for external features.
        
        Combines sections 2.2.1, 2.2.2, and 2.2.3.
        
        Args:
            geojson_path: Path to OSM GeoJSON file
            train_df: Training DataFrame with cost_of_living information
            output_csv: Optional path to save aggregated features
        
        Returns:
            DataFrame with POI counts per hex_id
        """
        logger.info("Starting external features processing...")
        
        # 2.2.1: Load and prepare points
        df_points = self.load_and_prepare_points(geojson_path)
        
        # 2.2.1: Classify by cost zone
        df_points = self.classify_points_by_cost_zone(df_points, train_df)
        
        # 2.2.2: Extract coordinates (for reference, can be used in distance calculations)
        coordinates = self.extract_coordinate_lists(geojson_path)
        
        # 2.2.3: Add binary flags
        df_points = self.add_poi_binary_flags(df_points)
        
        # 2.2.3: Aggregate by hex
        df_aggregated = self.aggregate_poi_by_hex(df_points)
        
        # Save if requested
        if output_csv:
            df_aggregated.to_csv(output_csv, index=False)
            logger.info(f"Saved aggregated features to {output_csv}")
        
        logger.info("External features processing completed")
        return df_aggregated
