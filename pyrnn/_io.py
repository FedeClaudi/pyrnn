import json
from pathlib import Path


def save_json(filepath, content, append=False, topcomment=None):
    """
    Saves content to a json file
    :param filepath: path to a file (must include .json)
    :param content: dictionary of stuff to save
    """
    fp = Path(filepath)
    if fp.suffix not in (".json"):
        raise ValueError(f"Filepath {fp} not valid should point to json file")

    with open(filepath, "w") as json_file:
        json.dump(content, json_file, indent=4)


def load_json(filepath):
    """
    Load a json file
    :param filepath: path to json file
    """
    fp = Path(filepath)
    if not fp.exists():
        raise ValueError("Unrecognized file path: {}".format(filepath))

    with open(filepath) as f:
        data = json.load(f)
    return data
