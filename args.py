import os.path
import yaml

def register_args(parser, config_file="config/all_HGT_config.yaml"):
    # GNN related
    parser.add_argument("--model_name", type=str, default="AEHAG")
    parser.add_argument("--add_SE",  default=False, action="store_true")
    parser.add_argument("--batch_size", type=int, default=10240)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)

    parser.add_argument("--n_mlp", type=int, default=3)
    parser.add_argument("--n_gnn", type=int, default=3)
    parser.add_argument("--loss", type=str, default="ce")
    parser.add_argument("--info_loss_weight", type=float, default=0.5)
    parser.add_argument("--loss_weight", type=list, default=None)
    parser.add_argument("--num_neigh", type=int, default=10)


    # training related
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--shuffle_attr", type=bool, default=True)
    parser.add_argument("--num_train_per_class", type=int, default=10)
    parser.add_argument("--show_training", type=bool, default=False)
    # parser.add_argument("--use_scheduler", type=bool, default=False)
    parser.add_argument("--use_scheduler", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--in_drop", type=float, default=0.3)
    parser.add_argument("--aug_add", type=float, default=0.3)
    parser.add_argument("--edge_drop", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0, help="gamma in focal loss")

    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--eval_step", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--patient", type=int, default=5, )
    parser.add_argument("--lr", type=float, default=5e-4, )
    parser.add_argument("--weight_decay", type=float, default=5e-5, )
    parser.add_argument("--split", type=str, default="random")
    # parser.add_argument("--split", type=str, default="by_label")
    parser.add_argument("--n_data", type=int, default=0)
    parser.add_argument("--se_min_count", type=int, default=100)

    parser.add_argument("--log_dir", type=str, default='./log/', )
    parser.add_argument("--config", type=str, default=config_file)


def parse_args_and_yaml(given_parser):

    given_configs, remaining = given_parser.parse_known_args()
    if given_configs.config:
        if os.path.exists(given_configs.config):
            with open(given_configs.config, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
                given_parser.set_defaults(**cfg)
        else:
            raise RuntimeError(f"Config file {given_configs.config} not exists")

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = given_parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    # print(args_text)
    return args

