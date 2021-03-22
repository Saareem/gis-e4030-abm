import pandas as pd

def shelves_to_df(combine = False):
    shelves = pd.read_json("shelves.json", orient = "records")

    shelves["x1"] = shelves["x"] - shelves["fixtureWidth"] / 2
    shelves["y1"] = shelves["y"] + shelves["fixtureDepth"] / 2
    shelves["x2"] = shelves["x"] + shelves["fixtureWidth"] / 2
    shelves["y2"] = shelves["y"] - shelves["fixtureDepth"] / 2

    return shelves