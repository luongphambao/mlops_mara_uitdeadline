# MLOps Marathon 2023 - UIT_Deadline

This repository is the solution UIT_Deadline (based on sample solution)

## Quickstart

1.  Prepare environment

    ```bash
    pip install -r requirements.txt
    make mlflow_up
    ```

2.  Scripts command

    -   Download data for each phase to `./data/raw_data` dir 

    -   Process data

        ```bash
        python src/raw_data_processor.py --phase-id phase-1 --prob-id prob-1
        ```
    -   Train model

        ```bash
        export MLFLOW_TRACKING_URI=http://localhost:5000
        python src/model_trainer.py --phase-id phase-1 --prob-id prob-1 --model-name xgb --add-captured-data true
        ```
        Register model: Go to mlflow UI at <http://localhost:5000> and register a new model named **phase-1_prob-1_model-1**
        
    -  Save capture data
        ```bash
        python3 src/create_capture_data.py
        ```
    -   Predict model trained with captured data
        ```bash
        python3 src/model_predict.py 1 #prob1
        ```
    -  Convert model to bentoml batching model
        ```bash
        python3 src/save_model_serving.py
        ```        
    -  Deployment
        ```bash
        make predictor_up
        ```
    -  System monitoring
        ```bash
       bash run.sh prom-graf up
        ```
    -  Load test
        ```bash
        locust -f src/load_test.py
        ```
    

