from pipeline import *
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from catboost import CatBoostClassifier

np.random.seed(101)
df = pd.read_parquet('data/raw_data/phase-1/prob-2/raw_train.parquet')

df = df.sample(frac=1)
train = df.drop(columns=['label'])

label = df['label']
X_train, X_val, y_train, y_val = train_test_split(train, label, test_size=0.2, random_state=101)

cats = ["feature1",
        "feature3",
        "feature4",
        "feature6",
        "feature7",
        "feature8",
        "feature9",
        "feature10",
        "feature11",
        "feature12",
        "feature14",
        "feature15",
        "feature16",
        "feature17",
        "feature19",
        "feature20"]
from sklearn.pipeline import FeatureUnion, Pipeline
columns = ['feature{}'.format(i) for i in range(1, 21)]
s = Pipeline(
    steps=(
        ('feature_select', FeatureSelector(columns)),
        ('woe', WOE(cats=cats, nums=[]))
    )
)
s.fit(X_train, y_train)
import pickle
with open('pipeline_2.pkl', 'wb') as f:
    pickle.dump(s, f)
X_train = s.transform(X_train)
X_val = s.transform(X_val)
 
model = CatBoostClassifier(
    iterations=4000,
    random_seed=101,
    learning_rate=0.02,
    eval_metric='AUC'
)
model.fit(X_train, y_train, eval_set=(X_val, y_val))
predict=model.predict(X_val)
from sklearn.metrics import roc_auc_score,accuracy_score
print(roc_auc_score(y_val, predict))
print(accuracy_score(y_val, predict))