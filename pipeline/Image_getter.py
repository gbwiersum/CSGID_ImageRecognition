from ImageRecognition.utils.OracleDBConnector import OracleDBConnector
import pandas as pd

OracleDBConnector()


def image_getter():
    df = pd.DataFrame(columns=["ImagePath", "HumanScore", "AIScore"])
    # TODO: get columns
    return df
