from model_predictor import main

if __name__ == "__main__":
    default_config_path = (
        AppPath.MODEL_CONFIG_DIR
        / ProblemConst.PHASE1
        / ProblemConst.PROB1
        / "model-1.yaml"
    ).as_posix()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path-1", type=str, default=default_config_path)
    parser.add_argument("--port-1", type=int, default=PREDICTOR_API_PORT)
    parser.add_argument("--config-path-2", type=str, default=default_config_path)
    parser.add_argument("--port-2", type=int, default=PREDICTOR_API_PORT)
    args = parser.parse_args()
    main({ 'config_path': args.config_path_1, 'port': args.port_1 })
    main({ 'config_path': args.config_path_2, 'port': args.port_2 })