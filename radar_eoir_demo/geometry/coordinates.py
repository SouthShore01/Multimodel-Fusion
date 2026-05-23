import numpy as np

EARTH_RADIUS = 6378137.0


def enu_to_latlon(x, y, origin_lat, origin_lon):
    dlat = (y / EARTH_RADIUS) * (180.0 / np.pi)
    dlon = (x / (EARTH_RADIUS * np.cos(np.deg2rad(origin_lat)))) * (180.0 / np.pi)
    return origin_lat + dlat, origin_lon + dlon


def latlon_to_enu(lat, lon, origin_lat, origin_lon):
    dy = (lat - origin_lat) * np.pi / 180.0 * EARTH_RADIUS
    dx = (lon - origin_lon) * np.pi / 180.0 * EARTH_RADIUS * np.cos(np.deg2rad(origin_lat))
    return dx, dy
