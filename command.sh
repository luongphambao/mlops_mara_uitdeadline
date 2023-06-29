#data processing
python src/raw_data_processor.py --phase-id phase-1 --prob-id prob-2
#model training
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/model_trainer.py --phase-id phase-1 --prob-id prob-1
#model predictor
# run model predictor
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/model_predictor.py --config-path data/model_config/phase-1/prob-1/model-1.yaml --port 8000
set MLFLOW_TRACKING_URI=http://localhost:5000
# curl in another terminal
curl -X POST http://localhost:8000/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json

# stop the predictor above
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/model_predictor.py --config-path data/model_config/phase-1/prob-2/model-1.yaml --port 8000
curl -X POST http://localhost:8000/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json

USE_NGROK=True python3 src/model_predictor.py --config-path1 data/model_config/phase-1/prob-1/model-1.yaml --config-path2 data/model_config/phase-1/prob-2/model-1.yaml --port 8000
	curl -X POST https://aaa5-123-21-154-4.ngrok.io/phase-1/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json

curl -X POST http://localhost:8000/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
curl -X POST http://localhost:8000/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/dead4937-38d1-4f99-9e76-ada0a99460e9.json

locust -f src/load_test.py
 BENTOML_CONFIG=configuration.yaml bentoml serve service:svc --development --reload --port 8000