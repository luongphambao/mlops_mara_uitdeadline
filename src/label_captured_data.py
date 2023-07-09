import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from problem_config import ProblemConfig, ProblemConst, get_prob_config
import os 
import json 
try:
    from problem_config import ProblemConst, create_prob_config
    from raw_data_processor import RawDataProcessor
    from utils import AppConfig, AppPath
except:
    from src.problem_config import ProblemConst, create_prob_config
    from src.raw_data_processor import RawDataProcessor
    from src.utils import AppConfig, AppPath
def label_captured_data(prob_config: ProblemConfig):
    train_x = pd.read_parquet(prob_config.train_x_path).to_numpy()
    train_y = pd.read_parquet(prob_config.train_y_path).to_numpy()
    test_x = pd.read_parquet(prob_config.test_x_path).to_numpy()
    test_y = pd.read_parquet(prob_config.test_y_path).to_numpy()
    ml_type = prob_config.ml_type

    logging.info("Load captured data")
    captured_x = pd.DataFrame()
    print(len(os.listdir(prob_config.captured_data_dir)))
    # for file_path in prob_config.captured_data_dir.glob("*.json"):
    #     captured_data =json.load(open(file_path))
    #     rows = captured_data["rows"]
    #     columns = captured_data["columns"]
    #     captured_data = pd.DataFrame(rows, columns=columns)
    #     captured_x = pd.concat([captured_x, captured_data])
    # #reindex feature1 to feature41
    # captured_x = captured_x.reindex(columns=["feature" + str(i) for i in range(1, 42)])
    if prob_config.prob_id == "prob-1":
        captured_x=pd.read_csv("data/data_captured_phase-2_prob-1.csv")
    else:
        captured_x=pd.read_csv("data/data_captured_phase-2_prob-2.csv")
    #drop duplicates
    df_raw=captured_x.copy()
    print(captured_x.shape)
    #captured_x = captured_x.drop_duplicates()
    #drop nan
    #captured_x = captured_x.dropna()
    category_index = RawDataProcessor.load_category_index(prob_config)
    captured_x = RawDataProcessor.apply_category_features(
            raw_df=captured_x,
            categorical_cols=prob_config.categorical_cols,
            category_index=category_index,
        )
    np_captured_x = captured_x.to_numpy()
    n_captured = len(np_captured_x)
    n_samples = len(train_x) + n_captured
    logging.info(f"Loaded {n_captured} captured samples, {n_samples} train + captured")

    logging.info("Initialize and fit the clustering model")
    n_cluster = int(n_samples /10) * len(np.unique(train_y))
    print(n_cluster)
    kmeans_model = MiniBatchKMeans(
        n_clusters=n_cluster, random_state=prob_config.random_state
    ).fit(train_x)

    logging.info("Predict the cluster assignments for the new data")
    kmeans_clusters = kmeans_model.predict(np_captured_x)

    logging.info(
        "Assign new labels to the new data based on the labels of the original data in each cluster"
    )
    new_labels = []
    for i in range(n_cluster):
        mask = kmeans_model.labels_ == i  # mask for data points in cluster i
        cluster_labels = train_y[mask]  # labels of data points in cluster i
        if len(cluster_labels) == 0:
            # If no data points in the cluster, assign a default label (e.g., 0)
            new_labels.append(0)
        else:
            # For a linear regression problem, use the mean of the labels as the new label
            # For a logistic regression problem, use the mode of the labels as the new label
            if ml_type == "regression":
                new_labels.append(np.mean(cluster_labels.flatten()))
            else:
                new_labels.append(
                    np.bincount(cluster_labels.flatten().astype(int)).argmax()
                )

    approx_label = [new_labels[c] for c in kmeans_clusters]
    approx_label_df = pd.DataFrame(approx_label, columns=[prob_config.target_col])
    df_raw["label"]=approx_label
    df_raw.to_csv("data/data_captured_phase-2_prob-1_kmean.csv",index=False)
    captured_x.to_parquet(prob_config.captured_x_path, index=False)
    approx_label_df.to_parquet(prob_config.uncertain_y_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE2)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    label_captured_data(prob_config)
