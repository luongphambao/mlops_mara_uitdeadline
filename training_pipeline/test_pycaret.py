import pandas as pd
from pycaret.classification import *

data_path="../data/raw_data/phase-2/prob-2/raw_train.parquet"
data=pd.read_parquet(data_path)
clf1 = setup(data, target = 'label', log_experiment = True, experiment_name = 'phase-2-prob-2',use_gpu=False)
best_model = compare_models()
# finalize the model
final_best = finalize_model(best_model)
# save model to disk
# feature_fig=plot_model(final_best, plot='feature')
# feature_fig.savefig("phase-2-prob-2feature_importance.png")
save_model(final_best, 'phase-2-prob-2-model')