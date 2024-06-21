import torch
import numpy as np
from main_eval import *


base_path = os.path.dirname(__file__)

def add_disturbance(data, type="gender"):
    if type in ["gender", "all"]:
        male_mask = data["patient"].bow_feat[:, 2] == 1
        female_mask = data["patient"].bow_feat[:, 3] == 1

        data["patient"].bow_feat[male_mask, 2] = 0
        data["patient"].bow_feat[male_mask, 3] = 1

        data["patient"].bow_feat[female_mask, 2] = 1
        data["patient"].bow_feat[female_mask, 3] = 0

    if type in ["age", "all"]:
        # age_matrix = data["patient"].bow_feat[:, 34:48]
        # new_index = torch.randperm(age_matrix.size(1))
        # data["patient"].bow_feat[:, 34:48] = age_matrix[:, new_index]

        # 取反
        # age_matrix = data["patient"].bow_feat[:, 34:48]
        # new_ = 1 - age_matrix
        # data["patient"].bow_feat[:, 34:48] = new_

        # 全0
        # age_matrix = data["patient"].bow_feat[:, 34:48]
        # new_ = torch.zeros_like(age_matrix)
        # data["patient"].bow_feat[:, 34:48] = new_

        # 重新随机分配
        # age_matrix = data["patient"].bow_feat[:, 34:48]
        # num_age_groups = age_matrix.size(1)
        # new_ = torch.zeros_like(age_matrix)
        # for i in range(age_matrix.size(0)):
        #     new_age_group = np.random.randint(0, num_age_groups)
        #     new_[i, new_age_group] = 1
        # data["patient"].bow_feat[:, 34:48] = new_

        # 随机值
        age_matrix = data["patient"].bow_feat[:, 34:48]
        data["patient"].bow_feat[:, 34:48] = torch.randn_like(age_matrix)

    if type == "all":
        weight_vec = data["patient"].bow_feat[:, 4:34]
        new_index = torch.randperm(weight_vec.size(1))
        data["patient"].bow_feat[:, 4:34] = weight_vec[:, new_index]

    return data


def build_dataset_func2(args, disturbe_type="gender", run=0):
    # build dataset
    datamodule = build_dataset_func(args)
    # new seed for disturb
    seed_everything(run)
    datamodule.data = add_disturbance(datamodule.data, disturbe_type)
    # recover seed for testing
    seed_everything(args.seed)
    return datamodule


def main_disturb(args, type="gender", dataset_func=build_dataset_func2, n_device=1, remove_file=False, run=0):
    # 加入扰动，测试结果
    seed_everything(args.seed)
    datamodule = dataset_func(args, disturbe_type=type, run=run)

    model = ContrastiveWrapper.load_from_checkpoint(args.save_path)
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.eval_step,
        num_sanity_val_steps=0,
    )
    model.eval()
    max_test_score = trainer.test(model, dataloaders=datamodule.test_dataloader())
    print(max_test_score)
    res = trainer.predict(model, dataloaders=datamodule.test_dataloader())

    torch.save(res, args.save_path + type + str(run) + ".res")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PreciseADR")
    register_args(parser)
    args = parser.parse_args()
    if "cuda" in args.device:
        args.device = args.device if torch.cuda.is_available() else "cpu"
    args.filtered_SE = None

    args.n_mlp = 1
    args.hid_dim = 512
    args.batch_size = 10240
    args.n_data = 0
    args.lr = 5e-4
    args.weight_decay = 5e-4
    args.split = "random"
    args.use_scheduler = True

    info = {
        "gender": f"{base_path}/../../gender_HGT_seed_42_n_gnn_3_n_mlp:3_score:0.8353.ckpt",
        "age": f"{base_path}/../../age_HGT_seed_42_n_gnn_3_n_mlp:3_score:0.8256.ckpt",
    }
    args.split = "origin"
    print(args)
    res_list = []
    for model in info:
        args.dataset = model
        for type in ["origin", "all", "age", "gender"]:
        # for type in ["age"]:
            for i in range(1):
                # args.seed = i
                args.save_path = info[model]
                res = main_disturb(args, type=type, dataset_func=build_dataset_func2, n_device=1, run=i)

                print(args.model_name)
                print(res)
                res_list.append(res)
    print(args)
    print(res_list)
