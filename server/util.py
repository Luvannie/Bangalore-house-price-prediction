import json
import pickle
import numpy as np


__locations = None
__data_columns = None
__model = None

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __locations
    global __data_columns
    global __model
    with open("server/artifacts/columns.json","r") as f:
        __data_columns = json.load(f)['data columns']
        __locations = __data_columns[3:]
    with open('server/artifacts/bengaluru_home_prices_model.pickle',"rb") as f:
        __model = pickle.load(f)
    print("loading saved artifacts...done")


def get_estimate_price(location,sqft,BHK,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = BHK
    if loc_index >= 0:
        x[loc_index] = 1
    return round(__model.predict([x])[0])

if __name__ == "__main__":
    load_saved_artifacts()
   