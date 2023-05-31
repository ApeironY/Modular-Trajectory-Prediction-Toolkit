import argparse
import torch
import os
import sys
sys.path.append(os.getcwd())
from XOE.utils.data_process import get_arbitary_data, get_momentary_data
from XOE.utils.datasets import AOE_Trainer_Dataset, MOE_Trainer_Dataset
from XOE.utils.utility import create_folders
from train import pretrain_AOE, pretrain_MOE

parser = argparse.ArgumentParser()
parser.add_argument("-aoe", "--use_AOE", action="store_true")
parser.add_argument("-d", "--dataset", type=int, nargs="+")
parser.add_argument("-k", "-eval_topK", type=int, nargs="+")
parser.add_argument("-o", "--operations", type=int, nargs="+")
parser.add_argument("-dd", "--data_dir", type=str, default="../dataset")
parser.add_argument("-df", "--data_from", type=str, default="ethucy")
parser.add_argument("-sd", "--save_dir", type=str, default="pretrained_models")
parser.add_argument("-flip_aug", "--flip_aug", type=bool, default=False)
parser.add_argument("--grid_num", type=int, default=6)
parser.add_argument("--grid_size", type=int, default=1)
parser.add_argument("-rec", "--reconstruct", type=bool, default=True)
parser.add_argument("--mask_size", type=int, default=6)
parser.add_argument("-bs", "--bs", type=int, default=128)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--base_lr", type=float, default=0.0005)
parser.add_argument("--obs_len", type=int, default=8)
parser.add_argument("--pred_len", type=int, default=12)
parser.add_argument("--embedding_size", type=int, default=24)
parser.add_argument("--obs_feat_size", type=int, default=48)
parser.add_argument("--num_layer", type=int, default=3)
parser.add_argument("--scene_feat_size", type=int, default=48)
parser.add_argument("--pred_feat_size", type=int, default=48)

# AOE modules
parser.add_argument("--add_behavior", action="store_true", help="add behavior extractor for cross-patch feature aggregation in Arbitrary Observation Encoder (AOE). Works only when use_xoe is True and obs_len == -1")
parser.add_argument("--add_intention", action="store_true", help="add intention extractor for cross-patch feature aggregation in Arbitrary Observation Encoder (AOE). Works only when use_xoe is True and obs_len == -1")
parser.add_argument("--add_empirical", action="store_true", help="add empirical information in input formulation for in-patch feature aggregation. Works only when use_xoe is True and obs_len == -1")

dataset_reference = {
    "ethucy": ["eth", "hotel", "univ", "zara1", "zara2"],
    "SDD": ["SDD"]
}
policy = {"name": "Poly", "power": 0.95}
args = parser.parse_args()
args.policy = policy
skip = 1
create_folders([args.save_dir])

if args.use_AOE and args.operations is None:
    print("==> Please Specify the Training Steps for AOE")
    exit()

for idx in args.dataset:
    dataset_name = dataset_reference[args.data_from][idx]
    print("==> Dataset:", dataset_name)

    if args.use_AOE:
        processed_data_train = get_arbitary_data(args, dataset_name, split="train", skip=skip)
        processed_data_val = get_arbitary_data(args, dataset_name, split="val", skip=skip)

        for i in range(processed_data_train["trj_data"].shape[0]):
            processed_data_train["trj_data"][i] = torch.from_numpy(processed_data_train["trj_data"][i]).float()
            processed_data_train["normalized_trj_data"][i] = torch.from_numpy(processed_data_train["normalized_trj_data"][i]).float()

        for i in range(processed_data_val["trj_data"].shape[0]):
            processed_data_val["trj_data"][i] = torch.from_numpy(processed_data_val["trj_data"][i]).float()
            processed_data_val["normalized_trj_data"][i] = torch.from_numpy(processed_data_val["normalized_trj_data"][i]).float()

        train_set = AOE_Trainer_Dataset(
            args.pred_len,
            trj_data=processed_data_train["trj_data"],
            normalized_trj_data=processed_data_train["normalized_trj_data"],
            formulated_data=processed_data_train["formulated_data"],  # Type: List
            add_empirical=args.add_empirical
        )
        val_set = AOE_Trainer_Dataset(
            args.pred_len,
            trj_data=processed_data_val["trj_data"],
            normalized_trj_data=processed_data_val["normalized_trj_data"],
            formulated_data=processed_data_val["formulated_data"],  # Type: List
            add_empirical=args.add_empirical
        )

        for op in args.operations:
            pretrain_AOE(train_set, val_set, op, args, dataset_name)
    else:
        if args.add_behavior or args.add_intention or args.add_empirical:
            print("==> add_behavior, add_intention, add_empirical are for AOE...")
            args.add_behavior = False
            args.add_intention = False
            args.add_empirical = False

        processed_data_train = get_momentary_data(args, dataset_name, split="train", skip=skip)
        processed_data_val = get_momentary_data(args, dataset_name, split="val", skip=skip)

        train_set = MOE_Trainer_Dataset(
            args.pred_len,
            trj_data=torch.from_numpy(processed_data_train["trj_data"]).float(),
            normalized_trj_data=torch.from_numpy(processed_data_train["normalized_trj_data"]).float(),
            formulated_data=processed_data_train["formulated_data"],  # Type: List
        )
        val_set = MOE_Trainer_Dataset(
            args.pred_len,
            trj_data=torch.from_numpy(processed_data_val["trj_data"]).float(),
            normalized_trj_data=torch.from_numpy(processed_data_val["normalized_trj_data"]).float(),
            formulated_data=processed_data_val["formulated_data"],  # Type: List
        )

        pretrain_MOE(train_set, val_set, args, dataset_name)
