#from utils.OracleDBConnector import OracleDBConnector
import pandas as pd
import glob

#OracleDBConnector()


def get_human_scored():
    df = pd.DataFrame(columns=["ImagePath", "HumanScore", "AIScore"])
    # TODO: get columns where HumanScore != None
    return df


def get_unscored():
    plate_list = glob.glob('../Minstrel-Screens/*/*')
    plates = pd.DataFrame(plate_list, columns=["filepath"])
    # TODO: get columns where AIScore = None
    return plates
