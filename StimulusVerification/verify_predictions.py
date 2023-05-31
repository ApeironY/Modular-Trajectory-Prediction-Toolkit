import copy
import numpy as np
import torch
import os
import argparse
import pickle
from model.models import Social_Stimulus_Verifier, Context_Stimulus_Verifier
from utils import filter_full_social_data

parser = argparse.ArgumentParser()
parser.add_argument("-bm", "--base_model_name", type=str)
parser.add_argument("-d", "--dataset", type=str)
parser.add_argument("-w_c", "--weight_context", type=float, default=0.5)
parser.add_argument("-w_s", "--weight_social", type=float, default=0.5)
parser.add_argument("-md", "--model_dir", type=str, default='saved_models')
parser.add_argument("--data_folder", type=str, default='ethucy')
parser.add_argument("--context_model_name", type=str, default="context_12_mix_128_dim_0.25x")
parser.add_argument("--social_model_name", type=str, default="social_12_mix_128_dim_16.00x")
parser.add_argument("--num_mixtures_context", type=int, default=12)
parser.add_argument("--num_mixtures_social", type=int, default=12)
parser.add_argument("--scale_context", type=float, default=0.25)
parser.add_argument("--scale_social", type=float, default=16.0)
parser.add_argument("-pad_c", "--context_padding_size", type=int, default=50)
parser.add_argument("-icl_s", "--social_inclusion_thresh", type=float, default=2.5)
parser.add_argument("--feature_size", type=int, default=128)
parser.add_argument("--best_of", type=int, default=20)
parser.add_argument("--max_samples", type=int, default=200)
parser.add_argument("--score_bin_size", type=int, default=2)
parser.add_argument("--nms_thresh", type=float, default=0.3)
parser.add_argument("--base_model_output_dir", type=str, default='base_model_outputs')
parser.add_argument("-nc", "--no_cache", action='store_true')
parser.add_argument("-fp", "--FDE_prioritized", action='store_true')
args = parser.parse_args()
obs_len = 8
pred_len = 12

try:
    from verification_configs import config
    weight_context = config[args.dataset]['weight_context']
    weight_social = config[args.dataset]['weight_social']
    bin_size = config[args.dataset]['bin_size']
    nms_thresh = config[args.dataset]['nms_thresh']
except ModuleNotFoundError:
    weight_context = args.weight_context / (args.weight_context + args.weight_social)
    weight_social = args.weight_social / (args.weight_context + args.weight_social)
    bin_size = args.score_bin_size
    nms_thresh = args.nms_thresh

if not os.path.exists(os.path.join('scores_cache', args.dataset)):
    os.makedirs(os.path.join('scores_cache', args.dataset))

base_model_predictions = np.load(
    os.path.join(args.base_model_output_dir, args.base_model_name, '%s_200_predictions.npy' % args.dataset)
)  # num_trj * num_candidates * pred_len * 2, in original coordinates
gt_trajectories_raw = np.load(
    os.path.join(args.base_model_output_dir, args.base_model_name, '%s_GT_%d_%d.npy' % (args.dataset, obs_len, pred_len))
)  # num_trj * (obs_len + pred_len) * 2, in original coordinates, only used for evaluation
gt_trajectories = gt_trajectories_raw[..., -2:]
obs_trajectories = gt_trajectories[:, :obs_len]

all_verification_scores = []

# Context
original_obs_trajectories_ = copy.deepcopy(obs_trajectories)
base_model_predictions_ = copy.deepcopy(base_model_predictions)

cache_file_path = os.path.join('scores_cache', args.base_model_name,
                               "%s_context_%s_cache.npy" % (args.dataset, args.context_model_name))
if not args.no_cache and os.path.exists(cache_file_path):
    compatibility_scores = np.load(cache_file_path)  # num_trj * num_candidates
else:
    verifier_checkpoint_path = os.path.join(
        args.model_dir, 'context', args.dataset, args.context_model_name + '.pth'
    )
    input_channel = 1
    # Translate trajectories from meters to pixels
    from dataset_info import scene_range_reference
    if args.dataset != 'univ':
        scene_semantic_map = \
            np.load(os.path.join("..", "dataset", args.data_folder, "semantic_maps", args.dataset + ".npy"))[None]
        scene_range = scene_range_reference[args.dataset]

    else:
        scene_semantic_map = np.load(
            os.path.join("..", "dataset", args.data_folder, "semantic_maps", args.dataset + "-001.npy"))[None]
        scene_range = scene_range_reference[args.dataset + "-001"]

    _, pixels_y, pixels_x = scene_semantic_map.shape
    original_obs_trajectories_[..., 0] = (original_obs_trajectories_[..., 0] - scene_range[0][0]) / (
            scene_range[0][1] - scene_range[0][0]) * pixels_x
    original_obs_trajectories_[..., 1] = (original_obs_trajectories_[..., 1] - scene_range[1][0]) / (
            scene_range[1][1] - scene_range[1][0]) * pixels_y
    base_model_predictions_[..., 0] = (base_model_predictions_[..., 0] - scene_range[0][0]) / (
            scene_range[0][1] - scene_range[0][0]) * pixels_x
    base_model_predictions_[..., 1] = (base_model_predictions_[..., 1] - scene_range[1][0]) / (
            scene_range[1][1] - scene_range[1][0]) * pixels_y

    original_obs_trajectories_ += args.context_padding_size
    base_model_predictions_ += args.context_padding_size
    half_size = args.context_padding_size // 2
    compatibility_scores = []
    scene_semantic_map = np.pad(
        scene_semantic_map, ((0, 0), (args.context_padding_size, args.context_padding_size),
                             (args.context_padding_size, args.context_padding_size))
    )
    original_obs_trajectories_ = torch.from_numpy(original_obs_trajectories_).float().cuda()
    base_model_predictions_ = torch.from_numpy(base_model_predictions_).float().cuda()

    net = Context_Stimulus_Verifier(input_channel=input_channel, feat_dim=args.feature_size,
                                    num_mixtures=args.num_mixtures_context, scale=args.scale_context).cuda()

    net.load_state_dict(torch.load(verifier_checkpoint_path))
    print('Model Loaded', verifier_checkpoint_path)
    net.eval()
    # Begin Scoring
    with torch.no_grad():
        for i in range(len(obs_trajectories)):
            extended_trj = torch.cat(
                [original_obs_trajectories_[i, None].repeat(base_model_predictions_.shape[1], 1, 1),
                 base_model_predictions_[i]], dim=1)
            velocity_seq = extended_trj[:, -pred_len:] - extended_trj[:, -pred_len - 1:-1]  # num_candidates * 12 * 2
            all_trj_map = []
            invalid_trj_register = []
            for j in range(base_model_predictions_.shape[1]):
                single_trj_map = []
                for k in range(pred_len):
                    x, y = np.around(extended_trj[j][-pred_len - 1 + k].cpu().numpy()).astype(int)
                    single_trj_map.append(scene_semantic_map[:, y - half_size:y + half_size,
                                          x - half_size:x + half_size])
                try:
                    single_trj_map = np.stack(single_trj_map)
                except ValueError:
                    single_trj_map = np.zeros((12, 1, 50, 50))
                    invalid_trj_register.append(j)

                all_trj_map.append(single_trj_map)
            all_trj_map = np.stack(all_trj_map).astype(float)

            velocity_seq = velocity_seq.float().reshape(-1, 2).cuda()
            all_trj_map = torch.from_numpy(all_trj_map).float().reshape(-1, scene_semantic_map.shape[0],
                                                                        args.context_padding_size,
                                                                        args.context_padding_size)

            scores = net(velocity_seq, all_trj_map)
            scores = scores.reshape(-1, pred_len).mean(dim=-1)
            scores[invalid_trj_register] = -1e4
            compatibility_scores.append(scores.cpu().numpy())

    compatibility_scores = np.stack(compatibility_scores)
    np.save(cache_file_path, compatibility_scores)
    print('Saved cache file for future reference', cache_file_path, compatibility_scores.shape)

all_verification_scores.append(compatibility_scores * weight_context)
# ################

# Social
original_obs_trajectories_ = copy.deepcopy(obs_trajectories)
base_model_predictions_ = copy.deepcopy(base_model_predictions)
base_model_predictions_ -= original_obs_trajectories_[:, None, -1:, :]
base_model_predictions_ = torch.from_numpy(base_model_predictions_).float().cuda()

divisor = torch.arange(1, pred_len + 1).float().cuda()[None, None, :, None]
base_model_predictions_ /= divisor

cache_file_path = os.path.join('scores_cache', args.base_model_name,
                               "%s_social_%s_cache.npy" % (args.dataset, args.social_model_name))
if not args.no_cache and os.path.exists(cache_file_path):
    compatibility_scores = np.load(cache_file_path)  # num_trj * num_candidates
else:
    verifier_checkpoint_path = os.path.join(
        args.model_dir, 'social', args.dataset, args.social_model_name + '.pth'
    )

    filtered_social_file_name = args.dataset + "_filtered_social_thresh_" + str(
        args.social_inclusion_thresh) + "_val.pkl"
    filtered_social_file_name = os.path.join("social_data", "filtered", filtered_social_file_name)
    preprocessed_social_data_path = os.path.join("social_data", "preprocessed",
                                                 args.dataset + '_val_' + "social_info.pkl")
    preprocessed_social_signature_lookup_path = os.path.join("social_data", "preprocessed",
                                                             args.dataset + '_val_' + "social_info_signature_lookup.pkl")
    with open(preprocessed_social_data_path, "rb") as f:
        preprocessed_full_social_data = pickle.load(f)
    with open(preprocessed_social_signature_lookup_path, "rb") as f:
        preprocessed_full_social_signature_lookup = pickle.load(f)

    if os.path.exists(filtered_social_file_name):
        with open(filtered_social_file_name, "rb") as f:
            filtered_social_data = pickle.load(f)
    else:
        if not os.path.exists(os.path.join('social_data', 'filtered')):
            os.makedirs(os.path.join('social_data', 'filtered'))
        filtered_social_data = filter_full_social_data(preprocessed_full_social_data,
                                                       thresh=args.social_inclusion_thresh,
                                                       obs_len=obs_len)
        with open(filtered_social_file_name, "wb") as f:
            pickle.dump(filtered_social_data, f, protocol=4)
            print("Filtered Social Saved.")

    assert len(filtered_social_data) == len(base_model_predictions_), (len(filtered_social_data), len(base_model_predictions_))
    synchronized_order = []
    for i in range(len(filtered_social_data)):
        signature = tuple(gt_trajectories_raw[i, 0, :2])
        synchronized_order.append(preprocessed_full_social_signature_lookup[signature])

    reordered_social = []
    for idx in synchronized_order:
        reordered_social.append(filtered_social_data[idx])
    filtered_social_data = reordered_social

    net = Social_Stimulus_Verifier(feat_dim=args.feature_size, obs_len=obs_len, pred_len=pred_len,
                                   num_mixtures=args.num_mixtures_social, trj_scale=args.scale_social).cuda()
    net.load_state_dict(torch.load(verifier_checkpoint_path))
    print('Model Loaded', verifier_checkpoint_path)
    net.eval()

    compatibility_scores = []
    with torch.no_grad():
        for i in range(len(base_model_predictions)):
            sizes = []
            input_social = []
            for step_social in filtered_social_data[i]:
                sizes.append(len(step_social))
                input_social.append(torch.from_numpy(step_social))
            input_social = torch.cat(input_social, dim=0).float().cuda()
            scores = net(input_social, sizes, base_model_predictions_[i], test_mode=True)
            compatibility_scores.append(scores.cpu().numpy())

    compatibility_scores = np.stack(compatibility_scores)
    np.save(cache_file_path, compatibility_scores)
    print('Saved cache file for future reference', cache_file_path, compatibility_scores.shape)

all_verification_scores.append(compatibility_scores * weight_social)
# ################

verification_scores = np.sum(np.array(all_verification_scores), axis=0)

# Begin Evaluation
diff = base_model_predictions[:, :args.best_of] - gt_trajectories[:, None, obs_len:]
diff = np.linalg.norm(diff, axis=-1)
total_min_ade = total_min_fde = 0
for i in range(len(diff)):
    if not args.FDE_prioritized:
        idx = np.argmin(np.sum(diff[i], axis=-1))  # minADE
    else:
        idx = np.argmin(diff[i, :, -1])
    total_min_fde += diff[i, idx, -1]
    total_min_ade += np.sum(diff[i, idx])
print('==> ADE/FDE Before Verification %.4f %.4f' % (total_min_ade / len(diff) / pred_len, total_min_fde / len(diff)))


total_min_ade = total_min_fde = 0
base_model_predictions_copy = copy.deepcopy(base_model_predictions)
for i in range(len(diff)):
    verification_score = verification_scores[i][:args.max_samples] // bin_size
    verification_score = torch.tensor(verification_score).float() - \
                         torch.arange(0, len(verification_score)).float() / (2 * len(verification_score))

    order = torch.argsort(verification_score, descending=True).tolist()
    base_model_predictions_copy[i, :args.max_samples] = \
        base_model_predictions_copy[i, :args.max_samples][order]

    # NMS ###################
    selected_idx = [0]
    ignore_idx = []
    for idx in range(1, args.max_samples):
        if len(selected_idx) >= args.best_of:
            ignore_idx.append(idx)
        else:
            div_diff = base_model_predictions_copy[i, selected_idx] - base_model_predictions_copy[i,
                                                                      idx:idx + 1]
            div_diff = np.linalg.norm(div_diff, axis=-1)[..., -1]
            if np.min(div_diff) < nms_thresh:
                ignore_idx.append(idx)
            else:
                selected_idx.append(idx)
    new_order = selected_idx + ignore_idx
    base_model_predictions_copy[i, :args.max_samples] = base_model_predictions_copy[i, :args.max_samples][new_order]
    # #######################

diff = base_model_predictions_copy[:, :args.best_of] - gt_trajectories[:, None, obs_len:]
diff = np.linalg.norm(diff, axis=-1)
total_min_ade = total_min_fde = 0
for i in range(len(diff)):
    if not args.FDE_prioritized:
        idx = np.argmin(np.sum(diff[i], axis=-1))  # minADE
    else:
        idx = np.argmin(diff[i, :, -1])
    total_min_fde += diff[i, idx, -1]
    total_min_ade += np.sum(diff[i, idx])
print('==> ADE/FDE After Verification %.4f %.4f' % (total_min_ade / len(diff) / pred_len, total_min_fde / len(diff)))


