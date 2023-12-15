from main import *
import datetime

if __name__ == "__main__":

    for dataset in ["gender", "age"]: #"all",
        for model_name in ["HGT"]:
            parser = argparse.ArgumentParser(description="PreciseADR")
            register_args(parser, config_file=f"config/{dataset}_HGT_config.yaml")
            args = parse_args_and_yaml(parser)
            seed = args.seed

            if "cuda" in args.device:
                args.device = args.device if torch.cuda.is_available() else "cpu"

            res_dict = {}
            try:
                for i in range(10):
                    args.model_name = model_name
                    args.seed = seed + i

                    res = main(args, model_wrapper=ContrastiveWrapper)
                    print(args)
                    print(dataset, args.model_name, res)
                    res_dict[f"{dataset}-{args.model_name}_run-{i}"] = res
            except Exception as e:
                print(e)
                raise e
            finally:
                print(res_dict)
                torch.save(res_dict, f"hetero_{args.dataset}.pth")
                with open(f'config/{dataset}_{model_name}_config_{datetime.datetime.now().strftime("%Y-%m-%d")}.yaml',
                          'w') as file:
                    yaml.dump(args.__dict__, file)
