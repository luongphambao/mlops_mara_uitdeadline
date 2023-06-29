#!/bin/bash
python src/model_predictor.py --config-path data/model_config/phase-1/prob-2/model-1.yaml --port 8000
python3 src/model_predictor.py --config-path1 data/model_config/phase-1/prob-1/model-2.yaml --config-path2 data/model_config/phase-1/prob-2/model-1.yaml --port 8000