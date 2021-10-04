import argparse
import torch
from models.auxiliary import PCCS_Dataset
from utils.data_process import get_data, get_test_data
from utils.utility import create_folders
from train_steps import train


parser = argparse.ArgumentParser()
parser.add_argument("-train", "--train", action="store_true")
parser.add_argument("-eval", "--eval", action="store_true")
parser.add_argument("-em", "--eval_mode", type=int, default=0)
parser.add_argument("-test", "--test", action="store_true")
parser.add_argument("-tf", "--test_filename", type=str, nargs="+")

parser.add_argument("-df", "--data_from", type=str, default="ethucy")
parser.add_argument("-d", "--dataset_ids", type=int, nargs="+")
parser.add_argument("-o", "--operations", type=int, nargs="+")
parser.add_argument("-k", "--eval_topK", type=int, nargs="+")

parser.add_argument("-sd", "--save_dir", type=str, default="saved_models")
parser.add_argument("-dd", "--data_dir", type=str, default="dataset")

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

parser.add_argument("--print_every", type=int, default=1)
parser.add_argument("--rad", type=float, default=1.0, help="Radius for Modality Loss")
parser.add_argument("-no_ML", "--disable_modality_loss", action="store_true")
parser.add_argument("--retrain", action="store_true")
parser.add_argument("--FDE_prioritize", action="store_true")
parser.add_argument("-ntc", "--no_time_check", action="store_true")

policy = {"name": "Poly", "power": 0.95}
args = parser.parse_args()
args.policy = policy

create_folders([args.save_dir, "processed_scene"], args.data_dir)
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

if args.operations is None:
    print("==> Please Specify the Training Steps")
    exit()

for idx in args.dataset_ids:
    dataset = dataset_reference[args.data_from][idx]
    print("==> Dataset:", dataset)
    print("==> Save Models to:", args.save_dir)

    if not args.test:
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
        assert len(args.test_filename) > 0, "Please specify the input file name"
        data, normalized_data = get_test_data(args.test_filename, args.obs_len, rotate=args.rotate)

        data = torch.from_numpy(data).float()
        normalized_data = torch.from_numpy(normalized_data).float()

        train_set = None
        val_set = PCCS_Dataset(
            args.obs_len, args.pred_len,
            data, normalized_data
        )

    for op in args.operations:
        train(train_set, val_set, op, dataset, args)
