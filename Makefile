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
	PORT=5040 docker-compose -f deployment/model_predictor/docker-compose.yml down

predictor_restart:
	PORT=5040 docker-compose -f deployment/model_predictor/docker-compose.yml stop
	PORT=5040 docker-compose -f deployment/model_predictor/docker-compose.yml start

predictor_curl:

	curl -X POST http://13.229.118.139:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/captured_data/phase-1/prob-1/a99572de-fefc-4bdf-a930-34373d36e544.json
	curl -X POST http://13.229.118.139:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/captured_data/phase-1/prob-1/a49ce035-153a-4a1f-9d9e-51d3b4ab8302.json
	curl -X POST http://13.229.118.139:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/captured_data/phase-1/prob-1/461ea294-279e-4f69-afb1-7aa3893373e6.json 
	curl -X POST http://13.229.118.139:5040/phase-1/prob-2/predict -H "Content-Type: application/json" -d @data/captured_data/phase-1/prob-2/6025e8a9-c5d5-4c5b-994f-02f086e21326.json 
	curl -X POST http://13.229.118.139:5040/phase-1/prob-2/predict -H "Content-Type: application/json" -d @data/captured_data/phase-1/prob-2/aebb376b-06ee-42b7-a122-52ed3b2cabd4.json
	curl -X POST http://13.229.118.139:5040/phase-1/prob-2/predict -H "Content-Type: application/json" -d @data/captured_data/phase-1/prob-2/60e87f55-9b93-4c0e-94e7-bae8beabdea1.json  

