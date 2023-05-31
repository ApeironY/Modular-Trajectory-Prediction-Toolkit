from cmath import nan
import os
import argparse
import math
import pickle
import numpy as np
from dataset_info import file_names, train_val_reference

parser = argparse.ArgumentParser()
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--min_ped', default=0, type=int)
parser.add_argument('--data_folder', default='ethucy', type=str)
args = parser.parse_args()

for dataset_name in train_val_reference.keys():
    for split in ["train", "val"]:
        all_files = []
        for file_idx in train_val_reference[dataset_name][split]:
            all_files.append(os.path.join('..', 'dataset', args.data_folder, 'raw', file_names[file_idx] + '.csv'))

        social_info = {}
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        total_num_considered_ped = 0

        social_info_signature_lookup = dict()
        for path in all_files:
            print('Processing', path)
            data = np.genfromtxt(path, delimiter=",").transpose()  # N * (frame, id, x, y)
            all_frame_ids = list(np.unique(data[:, 0]))
            frame_data_by_idx = []
            for frame in all_frame_ids:
                frame_data_by_idx.append(data[data[:, 0] == frame, :])
            num_sequences_in_file = int(
                math.ceil((len(all_frame_ids) - args.obs_len - args.pred_len + 1) / args.skip)
            )
            for idx in range(0, num_sequences_in_file * args.skip + 1, args.skip):
                all_data_in_current_sequence = np.concatenate(
                    frame_data_by_idx[idx: idx + args.obs_len + args.pred_len], axis=0
                )
                all_peds_in_current_seq = np.unique(all_data_in_current_sequence[:, 1])

                num_considered_ped = 0
                idx_tmp = []
                location_tmp = []
                speed_tmp = []
                ped_seq_tmp = []
                for _, ped_id in enumerate(all_peds_in_current_seq):
                    current_sequence_of_ped = all_data_in_current_sequence[all_data_in_current_sequence[:, 1] == ped_id, :]
                    pad_front = all_frame_ids.index(current_sequence_of_ped[0, 0]) - idx
                    pad_end = all_frame_ids.index(current_sequence_of_ped[-1, 0]) - idx + 1
                    if pad_end - pad_front != args.obs_len + args.pred_len or \
                            current_sequence_of_ped.shape[0] != args.obs_len + args.pred_len:
                        continue

                    surrounding_ped_locations = []
                    surrounding_ped_speed = []
                    surrounding_ped_ids = []
                    for i in range(args.obs_len):
                        surrounding_ped_ids = np.union1d(surrounding_ped_ids, all_data_in_current_sequence[all_data_in_current_sequence[:, 0] == current_sequence_of_ped[i, 0], 1])
                    surrounding_ped_ids = surrounding_ped_ids[surrounding_ped_ids != ped_id]

                    if len(surrounding_ped_ids) > 0:
                        for i in range(args.obs_len):
                            tmp = []
                            surrounding_ped_data = all_data_in_current_sequence[all_data_in_current_sequence[:, 0] == current_sequence_of_ped[i, 0], :]
                            for surrounding_ped in surrounding_ped_ids:
                                if surrounding_ped_data[surrounding_ped_data[:, 1] == surrounding_ped, :].shape[0] == 0:
                                    tmp.append(np.array([[surrounding_ped, nan, nan]]))
                                else:
                                    tmp.append(surrounding_ped_data[surrounding_ped_data[:, 1] == surrounding_ped, 1:])
                            surrounding_ped_locations.append(np.concatenate(tmp))

                        surrounding_ped_locations = np.array(surrounding_ped_locations)
                        for i in range(args.obs_len):
                            tmp = []
                            for j in range(surrounding_ped_locations.shape[1]):
                                if i == 0:
                                    tmp.append(np.array([nan, nan]))
                                else:
                                    if surrounding_ped_locations[i, j, 1] == nan or surrounding_ped_locations[i - 1, j, 1] == nan:
                                        tmp.append(np.array([nan, nan]))
                                    else:
                                        tmp.append(surrounding_ped_locations[i, j, 1:] - surrounding_ped_locations[i - 1, j, 1:])
                            surrounding_ped_speed.append(tmp)
                        surrounding_ped_speed = np.array(surrounding_ped_speed)

                    _idx = num_considered_ped
                    idx_tmp.append(_idx)
                    location_tmp.append(surrounding_ped_locations)
                    speed_tmp.append(surrounding_ped_speed)
                    ped_seq_tmp.append(current_sequence_of_ped)
                    num_considered_ped += 1

                if num_considered_ped > args.min_ped:
                    for idx_2, spl, sps, cps in zip(idx_tmp, location_tmp, speed_tmp, ped_seq_tmp):
                        social_info[total_num_considered_ped + idx_2] = {}
                        social_info[total_num_considered_ped + idx_2]['location'] = spl
                        social_info[total_num_considered_ped + idx_2]['ped_seq'] = cps
                        social_info_signature_lookup[tuple(cps[0, :2])] = total_num_considered_ped + idx_2

                    total_num_considered_ped += num_considered_ped

        if not os.path.exists(os.path.join('social_data', 'preprocessed')):
            os.makedirs(os.path.join('social_data', 'preprocessed'))

        print("Total Num Considered Ped:", total_num_considered_ped)
        with open(os.path.join("social_data", "preprocessed", dataset_name + '_' + split + '_' + "social_info.pkl"), 'wb') as f:
            pickle.dump(social_info, f, protocol=4)
        with open(os.path.join("social_data", "preprocessed", dataset_name + '_' + split + '_' + "social_info_signature_lookup.pkl"), 'wb') as f:
            pickle.dump(social_info_signature_lookup, f, protocol=4)
        print('Saved')
