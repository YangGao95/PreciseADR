import os
import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from root_utils import seed_everything, build_save_path

from models.hetero_cl import *
import argparse
from args import *

from dataset.please import DataModule
import yaml


def build_dataset_func(args):
    if args.dataset in ["gender", "age", "all"]:
        args.se_type = args.dataset
        datamodule = DataModule(n_layer=args.n_gnn, batch_size=args.batch_size,
                                split=args.split, n_data=args.n_data, add_SE=args.add_SE, args=args)
    else:
        raise RuntimeError(f"Don't support Dataset:{args.dataset}")

    datamodule.setup()
    data = datamodule.data

    args.i_start = datamodule.i_start
    args.i_end = datamodule.i_end
    args.d_start = datamodule.d_start
    args.d_end = datamodule.d_end

    # assign dataset infos
    args.metadata = data.metadata()
    args.in_dim_dict = {n_t: data[n_t].x.size(0) for n_t in data.metadata()[0]}
    args.out_dim = data["patient"].y.size(1)
    args.num_info = data["patient"].num_info
    args.num_node = data.num_nodes
    args.num_feat = args.in_dim = data["patient"].bow_feat.size(1)

    print("train data size:", datamodule.data["patient"].train_mask.sum())
    print("val data size:", datamodule.data["patient"].val_mask.sum())
    print("test data size:", datamodule.data["patient"].test_mask.sum())

    return datamodule


def main(args, other_callbacks=[], dataset_func=build_dataset_func, n_device=1, model_wrapper=ContrastiveWrapper,
         remove_file=False, device="gpu"):
    print(args.seed)
    seed_everything(args.seed)
    datamodule = dataset_func(args)

    model = model_wrapper(model_name=args.model_name, args=args)

    checkpoint_callback = ModelCheckpoint(monitor='val_auc', save_top_k=1,
                                          mode='max')
    early_stop_callback = EarlyStopping(monitor='val_auc', mode="max", patience=args.patient)
    callbacks = [checkpoint_callback, early_stop_callback]
    callbacks.extend(other_callbacks)
    trainer = Trainer(accelerator=device,
                      devices=n_device,
                      max_epochs=args.max_epochs,
                      # max_epochs=1,
                      # auto_select_gpus=True,
                      check_val_every_n_epoch=args.eval_step,
                      callbacks=callbacks,
                      num_sanity_val_steps=0,
                      )

    trainer.fit(model, datamodule)

    print(args.model_name)
    max_val_score = trainer.validate(dataloaders=datamodule.val_dataloader())

    val_score = max_val_score[0]['val_auc']
    args.save_path = build_save_path(args, val_score)
    trainer.save_checkpoint(args.save_path)

    max_test_score = trainer.test(dataloaders=datamodule.test_dataloader())

    if remove_file:
        os.remove(args.save_path)
    else:
        res = trainer.predict(model, dataloaders=datamodule.test_dataloader())
        torch.save(res, args.save_path + ".res")

    return max_val_score, max_test_score


def main_predict(args, dataset_func=build_dataset_func, model_wrapper=BasicModelWrapper, ckpt_path=None):
    seed_everything(args.seed)
    datamodule = dataset_func(args)

    model = model_wrapper.load_from_checkpoint(checkpoint_path=ckpt_path)

    trainer = Trainer(
        accelerator='gpu' if "cuda" in args.device else "cpu",
        devices=1,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.eval_step,
        num_sanity_val_steps=0,
    )
    args.save_path = ckpt_path
    res = trainer.predict(model, dataloaders=datamodule.test_dataloader())
    torch.save(res, args.save_path + ".res")

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PreciseADR")
    register_args(parser)
    args = parse_args_and_yaml(parser)
    dataset = args.dataset

    if "cuda" in args.device:
        args.device = args.device if torch.cuda.is_available() else "cpu"

    seed = args.seed
    res_dict = {}
    try:
        for i in range(1):
            for model_name in ["PreciseADR_HGT"]:
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
