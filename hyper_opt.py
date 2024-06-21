from main import *
import os
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
import ray.tune as tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune import CLIReporter
from easydict import EasyDict as edict


def main_with_hypers(config, args):
    print(config)
    args = edict(vars(args))
    for each in config:
        args[each] = config[each]

    print(args)
    call_back = TuneReportCallback(
        {"auc": "val_auc",
         "hit@1": "val_hit_1"}, on="validation_end"
    )
    main(args, [call_back], remove_file=True)


def main_hyper_opt(args, num_samples=200, num_epochs=10):
    """

    :param args: 程序运行默认参数
    :param num_samples: 模型数量
    :param num_epochs: 总轮次
    :return:
    """
    config = {
        "n_gnn": tune.choice(list(range(1, 8))),
        "n_mlp": tune.choice(list(range(1, 10))),
        "hid_dim": tune.choice([8, 16, 32, 64, 128, 256, 512, 1024]),
        "lr": tune.loguniform(1e-5, 1e-1),
        "dropout": tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),

        "gamma": tune.choice([0.5 * each for each in range(21)]),
        "aug_add": tune.choice([0.1 * each if each < 10 else 0.99 for each in range(11)]),
        "edge_drop": tune.choice([0.1 * each if each < 10 else 0.99 for each in range(11)]),
        "info_loss_weight": tune.choice([0.1 * each if each < 10 else 0.99 for each in range(0, 10)]),
        "temperature": tune.choice([0.01 * each for each in range(1, 20)]),
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["auc", "hit@1", "training_iteration"])

    analysis = tune.run(
        tune.with_resources(
            tune.with_parameters(main_with_hypers, args=args),
            resources={"cpu": 2, "gpu": 2}
        ),
        # metric="auc",
        metric="hit@1",
        mode="max",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_hgnn")

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PreciseADR")
    register_args(parser)
    args = parse_args_and_yaml(parser)
    args = parser.parse_args()
    args.model_name = "PreciseADR_HGT"
    if "cuda" in args.device:
        args.device = args.device if torch.cuda.is_available() else "cpu"

    main_hyper_opt(args)
