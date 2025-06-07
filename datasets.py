"""
This module provides streamable regression datasets for benchmarking adaptive online learning models.

The selection of datasets is inspired by the work:
    Sun, Y., Gomes, H. M., Pfahringer, B., & Bifet, A. (2025). Evaluation for Regression Analyses on Evolving Data Streams. 
    arXiv preprint arXiv:2502.07213. https://arxiv.org/abs/2502.07213
"""
from pathlib import Path
import csv
from river.stream import iter_csv, iter_arff


class BaseDataset:
    """Base class for any dataset that returns a stream of (x, y) pairs."""

    def load_stream(self):
        """Creates a generator object for streaming the data"""
        raise NotImplementedError("Subclasses must implement `load_as_stream`.")


class Abalone(BaseDataset):
    """
    UCI Abalone Dataset

    This dataset aims to predict the age of abalone from physical measurements.
    It includes attributes like length, diameter, weight, and categorical sex.

    Reference:
        Nash, W., Sellers, T., Talbot, S., Cawthorn, A., & Ford, W. (1994).
        *Abalone [Dataset]*. UCI Machine Learning Repository.
        https://doi.org/10.24432/C55C7W
    """
    def __init__(self, path="datasets/abalone/abalone.data") -> None:
        self.path = Path(path)

    def __len__(self):
        return 4177

    def load_stream(self):
        with open(self.path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                sex = row[0]
                features = {
                    f"sex_{sex}": 1,
                    "length": float(row[1]),
                    "diameter": float(row[2]),
                    "height": float(row[3]),
                    "whole_weight": float(row[4]),
                    "shucked_weight": float(row[5]),
                    "viscera_weight": float(row[6]),
                    "shell_weight": float(row[7]),
                }
                target = float(row[8])
                yield features, target


class BikeSharing(BaseDataset):
    """
    UCI Bike Sharing Dataset

    This dataset contains the hourly and daily count of rental bikes between the years 2011 and 2012
    in the Capital Bikeshare system with the corresponding weather and seasonal information.

    Reference:
        Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository.
        https://doi.org/10.24432/C5W894
    """
    def __init__(self, path="datasets/bike+sharing+dataset/hour.csv") -> None:
        self.path = Path(path)

    def __len__(self):
        return 17389

    def load_stream(self):
        with open(self.path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                target = float(row["cnt"])
                features = {
                    key: float(value) if key not in ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"] 
                    else int(value)
                    for key, value in row.items()
                    if key not in {"instant", "dteday", "casual", "registered", "cnt"}
                }

                yield features, target


class Superconductivty(BaseDataset):
    """
    UCI Superconductivty Dataset

    This dataset contains features extracted from superconductors, aiming to predict their critical temperatures.
    It includes 81 features derived from the chemical formula of superconductors.

    Reference:
        Hamidieh, K. (2018). Superconductivty Data [Dataset]. UCI Machine Learning Repository. 
        https://doi.org/10.24432/C53P47.
    """
    def __init__(self, path="datasets/superconductivty+data/train.csv") -> None:
        self.path = Path(path)

    def __len__(self):
        return 21263

    def load_stream(self):
        with open(self.path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    target = float(row["critical_temp"])
                    features = {
                        key: float(value)
                        for key, value in row.items()
                        if key != "critical_temp"
                    }
                    yield features, target
                except ValueError:  # skip malformed rows
                    continue


class House8L(BaseDataset):
    """
    House8L Dataset

    This dataset was constructed based on data provided by the U.S. Census Bureau's 1990 Summary Tape File 1 (STF 1).
    The data encompasses various demographic and housing statistics aggregated at the State-Place level across all U.S. states.
    Most of the raw counts were transformed into appropriate proportions to facilitate analysis. The dataset is part of a collection 
    designed to predict the median house price in a region based on its demographic composition and housing market characteristics. 

    Original source:
        DELVE repository of data.

    Source:
        Collection of regression datasets by Luís Torgo (ltorgo@ncc.up.pt) at http://www.ncc.up.pt/~ltorgo/Regression/DataSets.html

    OpenML link:
        https://www.openml.org/search?type=data&status=active&id=218&sort=runs
    """
    def __init__(self, path="datasets/dataset_2204_house_8L.arff") -> None:
        self.path = Path(path)

    def __len__(self):
        return 22784

    def load_stream(self):
        return iter_arff(self.path, target="price")
