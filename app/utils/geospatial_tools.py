import h3
from geopy.distance import geodesic
from loguru import logger
import traceback


class GeoSpatialTools:
    def __init__(self):
        pass

    def get_h3_center(self, hex_id):
        return h3.cell_to_latlng(hex_id)

    def calculate_h3_distance(self, row, hex_1='hex_id', hex_2='h_3'):
        hex_center = self.get_h3_center(row[hex_1])
        h3_center = self.get_h3_center(row[hex_2])
        return geodesic(hex_center, h3_center).km

    def calculate_coord_distance(self, coord1, coord2):
        return geodesic(coord1, coord2).km

    def hex_distance_from_coordinates(self, hex, coor_list):
        coor_base = self.get_h3_center(hex)
        cap_distances = []
        for coordenada in coor_list:
            distance_ = self.calculate_coord_distance(coor_base, coordenada)
            cap_distances.append(distance_)
        try:
            return round(min(cap_distances), 2)
        except Exception as e:
            logger.error(e)
            logger.error(traceback.format_exc())
            return 999999

    def nearest_hex_8_in_parent(self, raw_df, par_resolution):
        parent_name = f'h_{par_resolution}'
        df = raw_df.copy()
        df[parent_name] = df['hex_id'].apply(lambda x: h3.cell_to_parent(x, res=par_resolution))
        for idx, row in df.iterrows():
            current_hex = row['hex_id']
            current_hex_parent = row[parent_name]

            # Filter possible candidates within the same hexagon
            candidates = df[(df[parent_name] == current_hex_parent) & ~(df['missing'].isnull())]['hex_id'].tolist()

            # Remove itself from the candidate list
            candidates = [h for h in candidates if h != current_hex]

            if candidates:
                # Find the nearest hexagon by comparing distances
                nearest_hex = min(candidates, key=lambda h: h3.grid_distance(current_hex, h))
                df.at[idx, f'nn_h8_{parent_name}'] = nearest_hex
        return df
