import argparse
import torch
import os
import sys
sys.path.append(os.getcwd())
from PCCSNet_XOE.models.auxiliary import PCCS_Dataset, PCCS_AOE_Dataset, PCCS_MOE_Dataset
from PCCSNet_XOE.utils.data_process import get_data, get_test_data
from XOE.utils.data_process import get_arbitary_data, get_momentary_data, get_momentary_test_data, get_arbitrary_test_data
from PCCSNet_XOE.utils.utility import create_folders
from PCCSNet_XOE.train_steps import train
from PCCSNet_XOE.train_steps_XOE import train_XOE


parser = argparse.ArgumentParser()
parser.add_argument("-aoe", "--use_AOE", action="store_true")
parser.add_argument("-moe", "--use_MOE", action="store_true")
parser.add_argument("-train", "--train", action="store_true")
parser.add_argument("-eval", "--eval", action="store_true")
parser.add_argument("-em", "--eval_mode", type=int, default=0)
parser.add_argument("-test", "--test", action="store_true")
parser.add_argument("-tf", "--test_filename", type=str, nargs="+")

parser.add_argument("-df", "--data_from", type=str, default="ethucy")
parser.add_argument("-d", "--dataset_ids", type=int, nargs="+")
parser.add_argument("-o", "--operations", type=int, nargs="+")
parser.add_argument("-k", "--eval_topK", type=int, nargs="+")

parser.add_argument("-sd", "--save_dir", type=str, default="PCCSNet_XOE/saved_models")
parser.add_argument("-dd", "--data_dir", type=str, default="../dataset")
parser.add_argument("--pretrain", action="store_true")
parser.add_argument("--pretrain_dir", type=str, default="pretrained_models")
parser.add_argument("-lp", "--load_pretrain", type=str, default="ep_99.pth")

parser.add_argument("--flip_aug", action="store_true")
parser.add_argument("--rotate", action="store_true")
parser.add_argument("-c", "--n_cluster", type=int, default=200)
parser.add_argument("-bs", "--bs", type=int, default=128)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--base_lr", type=float, default=0.0005)
parser.add_argument("--obs_len", type=int, default=8)
parser.add_argument("--pred_len", type=int, default=12)
parser.add_argument("--obs_feat_size", type=int, default=48)
parser.add_argument("--pred_feat_size", type=int, default=48)
parser.add_argument("--encoder_layer", type=int, default=3)
parser.add_argument("--grid_num", type=int, default=6)
parser.add_argument("--grid_size", type=int, default=1)
parser.add_argument("--embedding_size", type=int, default=24)

parser.add_argument("-reC", "--reC", action="store_true")
parser.add_argument("-pm", "--pm", type=float, default=1.0)

parser.add_argument("--print_every", type=int, default=1)
parser.add_argument("--rad", type=float, default=1.0, help="Radius for Modality Loss")
parser.add_argument("-no_ML", "--disable_modality_loss", action="store_true")
parser.add_argument("--retrain", action="store_true")
parser.add_argument("--FDE_prioritize", action="store_true")
parser.add_argument("-ntc", "--no_time_check", action="store_true")
parser.add_argument("--data_type", default="arbi", type=str)

# AOE modules
parser.add_argument("--add_behavior", action="store_true", help="add behavior extractor for cross-patch feature aggregation in Arbitrary Observation Encoder (AOE). Works only when use_xoe is True and obs_len == -1")
parser.add_argument("--add_intention", action="store_true", help="add intention extractor for cross-patch feature aggregation in Arbitrary Observation Encoder (AOE). Works only when use_xoe is True and obs_len == -1")
parser.add_argument("--add_empirical", action="store_true", help="add empirical information in input formulation for in-patch feature aggregation. Works only when use_xoe is True and obs_len == -1")

policy = {"name": "Poly", "power": 0.95}
args = parser.parse_args()
args.policy = policy

create_folders([args.save_dir, "PCCSNet_XOE/processed_scene"], args.data_dir)
dataset_reference = {
    "ethucy": ["eth", "hotel", "univ", "zara1", "zara2"],
    "SDD": ["SDD"]
}

# Main Process
if len(args.dataset_ids) == 0:
    print("==> Please Specify the Target Dataset")
    exit()

if args.train:
    args.operations = [0, 1, 2, 3, 4]
elif args.eval:
    args.operations = [5]
elif args.test:
    args.operations = [6]

if args.use_AOE and args.use_MOE:
    print("==> Please Specify use AOE or MOE")
    exit()

if args.use_AOE or args.use_MOE:
    if args.use_MOE:
        print("==> MOE only works when obs_len = 2, force changed")
        args.obs_len = 2
    args.disable_modality_loss = True
skip = 1

if args.operations is None:
    print("==> Please Specify the Training Steps")
    exit()

for idx in args.dataset_ids:
    dataset = dataset_reference[args.data_from][idx]
    print("==> Dataset:", dataset)
    print("==> Save Models to:", args.save_dir)

    if not args.test:
        if not args.use_AOE and not args.use_MOE:
            train_data, val_data, normalized_train_data, normalized_val_data, train_scene_split, = \
                get_data(dataset, args.data_from, obs_len=args.obs_len, pred_len=args.pred_len,
                         rotate=args.rotate, flip_aug=args.flip_aug, file_dir=args.data_dir)

            print("==> Generated %d Trajectories for Training, %d for Testing" % (train_data.shape[0], val_data.shape[0]))

            train_data = torch.from_numpy(train_data).float()
            val_data = torch.from_numpy(val_data).float()
            normalized_train_data = torch.from_numpy(normalized_train_data).float()
            normalized_val_data = torch.from_numpy(normalized_val_data).float()

            train_set = PCCS_Dataset(
                args.obs_len, args.pred_len,
                train_data, normalized_train_data,
                split_marks=train_scene_split
            )
            val_set = PCCS_Dataset(
                args.obs_len, args.pred_len,
                val_data, normalized_val_data
            )

        else:
            if args.use_AOE:
                processed_data_train, split_marks = get_arbitary_data(args, dataset, split="train", skip=skip, req_split=True)
                processed_data_val = get_arbitary_data(args, dataset, split="val", skip=skip)
                for i in range(processed_data_train["trj_data"].shape[0]):
                    processed_data_train["trj_data"][i] = torch.from_numpy(processed_data_train["trj_data"][i]).float()
                    processed_data_train["normalized_trj_data"][i] = torch.from_numpy(processed_data_train["normalized_trj_data"][i]).float()

                for i in range(processed_data_val["trj_data"].shape[0]):
                    processed_data_val["trj_data"][i] = torch.from_numpy(processed_data_val["trj_data"][i]).float()
                    processed_data_val["normalized_trj_data"][i] = torch.from_numpy(processed_data_val["normalized_trj_data"][i]).float()

                print("==> Generated %d Trajectories for Training, %d for Testing" % (processed_data_train["trj_data"].shape[0], processed_data_val["trj_data"].shape[0]))

                train_set = PCCS_AOE_Dataset(
                    pred_len=args.pred_len,
                    trj_data=processed_data_train["trj_data"],
                    normalized_trj_data=processed_data_train["normalized_trj_data"],
                    formulated_obs=processed_data_train["formulated_data"],  # Type: List
                    split_marks=split_marks, input_dim=6 if args.add_empirical else 5
                )
                val_set = PCCS_AOE_Dataset(
                    pred_len=args.pred_len,
                    trj_data=processed_data_val["trj_data"],
                    normalized_trj_data=processed_data_val["normalized_trj_data"],
                    formulated_obs=processed_data_val["formulated_data"],  # Type: List
                    input_dim=6 if args.add_empirical else 5
                )
            else:
                processed_data_train, split_marks = get_momentary_data(args, dataset, split="train", skip=skip, req_split=True)
                processed_data_val = get_momentary_data(args, dataset, split="val", skip=skip)
                print("==> Generated %d Trajectories for Training, %d for Testing" % (processed_data_train["trj_data"].shape[0], processed_data_val["trj_data"].shape[0]))

                train_set = PCCS_MOE_Dataset(
                    args.obs_len, args.pred_len,
                    trj_data=torch.from_numpy(processed_data_train["trj_data"]).float(),
                    normalized_trj_data=torch.from_numpy(processed_data_train["normalized_trj_data"]).float(),
                    formulated_obs=processed_data_train["formulated_data"],  # Type: List
                    split_marks=split_marks
                )
                val_set = PCCS_MOE_Dataset(
                    args.obs_len, args.pred_len,
                    trj_data=torch.from_numpy(processed_data_val["trj_data"]).float(),
                    normalized_trj_data=torch.from_numpy(processed_data_val["normalized_trj_data"]).float(),
                    formulated_obs=processed_data_val["formulated_data"],  # Type: List
                )

    else:
        assert len(args.test_filename) > 0, "Please specify the input file name"
        if not args.use_AOE and not args.use_MOE:
            data, normalized_data = get_test_data(args.test_filename, args.obs_len, rotate=args.rotate)

            data = torch.from_numpy(data).float()
            normalized_data = torch.from_numpy(normalized_data).float()

            train_set = None
            val_set = PCCS_Dataset(
                args.obs_len, args.pred_len,
                data, normalized_data
            )
        else:
            if args.use_AOE:
                processed_data_val = get_arbitrary_test_data(args.test_filename, args, skip=skip)
                train_set = None
                val_set = PCCS_AOE_Dataset(
                    args.pred_len,
                    trj_data=torch.from_numpy(processed_data_val["trj_data"]).float(),
                    normalized_trj_data=torch.from_numpy(processed_data_val["normalized_trj_data"]).float(),
                    formulated_obs=processed_data_val["formulated_data"],  # Type: List
                )
            else:
                processed_data_val = get_momentary_test_data(args.test_filename, args, skip=skip)
                train_set = None
                val_set = PCCS_MOE_Dataset(
                    args.obs_len, args.pred_len,
                    trj_data=torch.from_numpy(processed_data_val["trj_data"]).float(),
                    normalized_trj_data=torch.from_numpy(processed_data_val["normalized_trj_data"]).float(),
                    formulated_obs=processed_data_val["formulated_data"],  # Type: List
                )

    if not args.use_AOE and not args.use_MOE:
        for op in args.operations:
            train(train_set, val_set, op, dataset, args)
    else:
        for op in args.operations:
            train_XOE(train_set, val_set, op, dataset, args)
