import wandb

def get_wandb_instance(config, args):
    group = args.model
    # replace key from args to key in config 
    args_keys = set(vars(args))
    config_keys = set(config.keys())

    common_keys = list(args_keys.intersection(config_keys))

    arg_dict = {}
    for key in vars(args):
        if vars(args).get(key):
            arg_dict[key] = vars(args).get(key)
    for key in common_keys:
        if key in config.keys() and key in arg_dict.keys():
            config[key] = arg_dict[key]
    run = wandb.init(
        entity="aiotlab",
        project="Air-Quality-Prediction-Benchmark", 
        group=group,
        config=config
    )
    return run, config