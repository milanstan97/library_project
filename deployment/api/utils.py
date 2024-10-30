import json
import os
import pickle
from typing import Optional, Tuple

import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from sklearn.calibration import CalibratedClassifierCV

wd = os.path.dirname(os.path.realpath(__file__))
dir = os.path.realpath(os.path.join(wd, ".."))


def load_model(model_name: str) -> CalibratedClassifierCV:
    """
    Load a model from a pickle file.

    Args:
        model_name (str): Name of the model.

    Returns:
        model (CalibratedClassifierCV): Trained and calibrated model.
    """
    model_path = os.path.join(dir, f"{model_name}.pkl")

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    return model


def load_library_map() -> dict:
    """
    Load a JSON file containing a map of library names to addresses.

    Returns:
        dict: Dictionary containing library name and address mappings.
    """
    map_path = os.path.join(wd, "library_name_address_map.json")

    with open(map_path, "r") as file:
        library_map = json.load(file)

    return library_map


def get_coordinates(address: str) -> Optional[Tuple[float, float]]:
    """
    Get latitude and longitude for a given address.

    Args:
        address (str): The address to geocode.

    Returns:
        Optional[Tuple[float, float]]: A tuple of (latitude, longitude)
        if found, else None.
    """
    geolocator = Nominatim(user_agent="data-science", timeout=10)

    try:
        location = geolocator.geocode(address)
        if location:
            lat, lon = location.latitude, location.longitude
        else:
            print("Address not found")
            return None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return lat, lon


def calculate_distance(library_address: str, customer_address: str) -> float:
    """
    Calculate the distance in kilometers between two addresses.

    Args:
        library_address (str): Address of the library.
        customer_address (str): Address of the customer.

    Returns:
        float: Distance between the two addresses in kilometers,
        or NaN if coordinates are not found.
    """
    coords_library = get_coordinates(library_address)
    coords_customer = get_coordinates(customer_address)

    if coords_library and coords_customer:
        distance = geodesic(coords_library, coords_customer).km
        return distance

    return np.nan
