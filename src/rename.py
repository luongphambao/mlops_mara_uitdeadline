import pandas as pd 
import numpy as np

data=pd.read_csv("data/result_test.csv")
#drop label
data=data.drop(["label"],axis=1)
#rename prediction_label to label
data=data.rename(columns={"prediction_label":"label"})
#rename prediction_score to prob
data=data.rename(columns={"prediction_score":"prob"})
data.to_csv("data/result_test.csv",index=False)