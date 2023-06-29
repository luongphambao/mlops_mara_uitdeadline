import pandas as pd

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

from prometheus_client import Summary

REQUEST_TIME=Summary('request_processing_time', 'Time spend processing request')

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model')])
class MLOPS_Predictor(BentoService):
    """
    A minimum prediction service exposing a Scikit-learn model
    """

    @REQUEST_TIME.time()
    @api(input=DataframeInput(), mb_max_latency=200, mb_max_batch_size=200,batch=True)
    def predict(self, df: pd.DataFrame):
        """
        An inference API named `predict` with Dataframe input adapter, which codifies
        how HTTP requests or CSV files are converted to a pandas Dataframe object as the
        inference API function input
        """
        return self.artifacts.model.predict(df)