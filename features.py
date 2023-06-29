# %%file data_preprocessing_prob2.py
# To be used for creating pipelines and personalizing them
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import RobustScaler
import pandas as pd

# Building a function to standardize columns

def feature_name_standardize(df: pd.DataFrame):
    df_ = df.copy()
    df_.columns = [i.replace(" ", "_").lower() for i in df_.columns]
    return df_

# Building a function to drop features

def drop_feature(df: pd.DataFrame, features: list = []):
    df_ = df.copy()
    if len(features) != 0:
        df_ = df_.drop(columns=features)

    return df_

# Building a function to treat incorrect value

def mask_value(df: pd.DataFrame, feature: str = None, value_to_mask: str = None, masked_value: str = None):
    df_ = df.copy()
    if feature != None and value_to_mask != None:
        if feature in df_.columns:
            df_[feature] = df_[feature].astype('object')
            df_.loc[df_[df_[feature] == value_to_mask].index, feature] = masked_value
            df_[feature] = df_[feature].astype('category')

    return df_

# Building a custom imputer

def impute_category_unknown(df: pd.DataFrame, fill_value: str):
    df_ = df.copy()
    for col in df_.select_dtypes(include='category').columns.tolist():
        df_[col] = df_[col].astype('object')
        df_[col] = df_[col].fillna('Unknown')
        df_[col] = df_[col].astype('category')
    return df_

# Building a custom data preprocessing class with fit and transform methods for standardizing column names

class FeatureNamesStandardizer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """All SciKit-Learn compatible transformers and classifiers have the
        same interface. `fit` always returns the same object."""
        return self

    def transform(self, X):
        """Returns dataframe with column names in lower case with underscores in place of spaces."""
        X_ = feature_name_standardize(X)
        return X_


# Building a custom data preprocessing class with fit and transform methods for dropping columns

class ColumnDropper(TransformerMixin):
    def __init__(self, features: list):
        self.features = features

    def fit(self, X, y=None):
        """All SciKit-Learn compatible transformers and classifiers have the
        same interface. `fit` always returns the same object."""
        return self

    def transform(self, X):
        """Given a list of columns, returns a dataframe without those columns."""
        X_ = drop_feature(X, features=self.features)
        return X_



# Building a custom data preprocessing class with fit and transform methods for custom value masking

class CustomValueMasker(TransformerMixin):
    def __init__(self, feature: str, value_to_mask: str, masked_value: str):
        self.feature = feature
        self.value_to_mask = value_to_mask
        self.masked_value = masked_value

    def fit(self, X, y=None):
        """All SciKit-Learn compatible transformers and classifiers have the
        same interface. `fit` always returns the same object."""
        return self

    def transform(self, X):
        """Return a dataframe with the required feature value masked as required."""
        X_ = mask_value(X, self.feature, self.value_to_mask, self.masked_value)
        return X_


# Building a custom class to one-hot encode using pandas
class PandasOneHot(TransformerMixin):
    def __init__(self, columns: list = None):
        self.columns = columns

    def fit(self, X, y=None):
        """All SciKit-Learn compatible transformers and classifiers have the
        same interface. `fit` always returns the same object."""
        return self

    def transform(self, X):
        """Return a dataframe with the required feature value masked as required."""
        X_ = pd.get_dummies(X, columns = self.columns, drop_first=True)
        return X_

# Building a custom class to fill nulls with Unknown
class FillUnknown(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """All SciKit-Learn compatible transformers and classifiers have the
        same interface. `fit` always returns the same object."""
        return self

    def transform(self, X):
        """Return a dataframe with the required feature value masked as required."""
        X_ = impute_category_unknown(X, fill_value='Unknown')
        return X_


if __name__ == "__main__":
  # To Standardize feature names
  feature_name_standardizer = FeatureNamesStandardizer()

  X_train = feature_name_standardizer.fit_transform(X_train)
  X_val = feature_name_standardizer.transform(X_val)
  X_test = feature_name_standardizer.transform(X_test)


  # To impute categorical Nulls to Unknown
  cat_columns = X_train.select_dtypes(include="category").columns.tolist()
  imputer = FillUnknown()

  X_train[cat_columns] = imputer.fit_transform(X_train[cat_columns])
  X_val[cat_columns] = imputer.transform(X_val[cat_columns])
  X_test[cat_columns] = imputer.transform(X_test[cat_columns])

  # To encode the data
  one_hot = PandasOneHot()

  X_train = one_hot.fit_transform(X_train)
  X_val = one_hot.transform(X_val)
  X_test = one_hot.transform(X_test)


  # Scale the numerical columns
  robust_scaler = RobustScaler(with_centering=False, with_scaling=True)
  num_columns = [
        "feature2",
        "feature5",
        "feature13",
        "feature18"
      ]

  X_train[num_columns] = pd.DataFrame(
      robust_scaler.fit_transform(X_train[num_columns]),
      columns=num_columns,
      index=X_train.index,
  )
  X_val[num_columns] = pd.DataFrame(
      robust_scaler.transform(X_val[num_columns]), columns=num_columns, index=X_val.index
  )
  X_test[num_columns] = pd.DataFrame(
      robust_scaler.transform(X_test[num_columns]),
      columns=num_columns,
      index=X_test.index,
  )
