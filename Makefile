# teardown
teardown:
	make predictor_down
	make mlflow_down

# mlflow
mlflow_up:
	docker-compose -f deployment/mlflow/docker-compose.yml up -d

mlflow_down:
	docker-compose -f deployment/mlflow/docker-compose.yml down

# predictor
predictor_up:
	bash deployment/deploy.sh run_predictor data/model_config/phase-1/prob-1/model-2.yaml data/model_config/phase-1/prob-2/model-1.yaml 5040
predictor_down:
	PORT=8000 docker-compose -f deployment/model_predictor/docker-compose.yml down

predictor_restart:
	PORT=8000 docker-compose -f deployment/model_predictor/docker-compose.yml stop
	PORT=8000 docker-compose -f deployment/model_predictor/docker-compose.yml start

predictor_curl:
	
	curl -X POST http://localhost:8000/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/248ba825-81b4-4601-9a41-b409e512b74b.json
	curl -X POST http://localhost:8000/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/755d777f-5978-445a-8636-71e843ec5098.json
	curl -X POST http://localhost:8000/phase-1/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-2/0a7a806e-afd7-4d06-87e9-9a9d5e702f20.json
	curl -X POST http://localhost:8000/phase-1/prob-2/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-2/0b3cd47f-e17a-49cd-906a-8b7795427c06.json
#39.9442612402