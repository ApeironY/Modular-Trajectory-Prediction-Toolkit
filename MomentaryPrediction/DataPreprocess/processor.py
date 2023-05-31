import copy
import numpy as np
import pandas as pd
import cv2
import os
import sys
sys.path.append(".")
import pickle
from DataPreprocess.processor_tools import get_social_and_scene_info


class eth_ucy_Processor:
    def __init__(self, grids_per_line=6, grid_size=1, add_empirical=False):
        self.file_names = ["eth", "hotel", "univ-001", "univ-003", "univ-example", "zara1", "zara2", "zara3"]
        self.scene_reference = {
            "eth": ((-8.5, 16), (-9.9, 21.5)),
            "hotel": ((-6.5, 5.9), (-10.4, 4.8)),
            "univ-001":  ((-0.6, 15.5), (-0.3, 14)),
            "univ-003": ((-0.6, 15.5), (-0.3, 14)),
            "univ-example": ((-0.6, 15.7), (-0.8, 14)),
            "zara1": ((-0.3, 15.5), (-0.7, 14.2)),
            "zara2": ((-0.4, 15.6), (-0.3, 16.7)),
            "zara3": ((-0.4, 15.6), (-0.3, 16.7))
        }
        self.grids_per_line = grids_per_line
        self.grid_size = grid_size
        self.add_empirical = add_empirical

    def gen_data(self, incl_rev=False):
        print("==> Generating Normal Data...")
        self._gen_data()
        if incl_rev:
            print("==> Generating Reversed Data...")
            self._gen_data(True)

    def _gen_data(self, reverse_frames=False):
        for name in self.file_names:
            file_name = name + "_g_" + str(self.grids_per_line) + "_s_" + str(self.grid_size) + "_rev_" + str(reverse_frames) + "_emp_" + str(self.add_empirical) + ".pkl"
            save_dir = os.path.join("../dataset", "ethucy", "processed_xoe", file_name)
            if os.path.exists(save_dir):
                print("File %s exists, continue" % save_dir)
                continue

            path = os.path.join("../dataset", "ethucy", "raw", name + ".csv")
            data = np.genfromtxt(path, delimiter=",").transpose()

            ped_ids = np.unique(data[:, 1])
            frames = np.unique(data[:, 0])
            trj_of_ped = {}
            full_data = {}

            normalized_frame_cnt = 0
            normalized_data = []
            for frame in sorted(frames, reverse=reverse_frames):
                data_of_frame = data[np.where(data[:, 0] == frame)]
                data_of_frame[:, 0] = normalized_frame_cnt
                normalized_data.append(data_of_frame)
                normalized_frame_cnt += 1

            normalized_data = np.concatenate(normalized_data, axis=0)
            data = normalized_data
            full_data["frame_num"] = normalized_frame_cnt

            peds_in_frame = {}
            for ped_id in ped_ids:
                ped_trj = data[np.where(data[:, 1] == ped_id)]
                for frame in ped_trj[:, 0]:
                    peds_in_frame[frame] = peds_in_frame.get(frame, []) + [ped_id]
                ped_speed = np.zeros_like(ped_trj[:, 2:])
                if len(ped_trj) > 1:
                    ped_speed[1:] = ped_trj[1:, 2:] - ped_trj[:-1, 2:]
                    ped_speed[0] = ped_speed[1]

                ped_trj = np.concatenate([ped_trj, ped_speed], axis=-1)[:, [0, 2, 3, 4, 5]]
                trj_of_ped[ped_id] = {"seq": ped_trj, "start_frame": ped_trj[0][0]}

            scene_mask = np.load(os.path.join("../dataset", "ethucy", "semantic_maps", name + ".npy"))
            formulated_data_of_ped = get_social_and_scene_info(copy.deepcopy(trj_of_ped), scene_mask,
                                                                              self.grids_per_line, self.grid_size,
                                                                              peds_in_frame, unit="meters", scene_range=self.scene_reference[name], add_empirical=self.add_empirical)

            full_data["trj_of_ped"] = trj_of_ped
            full_data["data_of_ped"] = formulated_data_of_ped

            with open(save_dir, "wb") as f:
                pickle.dump(full_data, f, protocol=4)
            print("Saved", save_dir)


class SDD_Processor:
    def __init__(self, grids_per_line=6, grid_size=40, add_empirical=False):
        self.scene_nums = {
            "bookstore": 7,
            "coupa": 4,
            "deathCircle": 5,
            "gates": 9,
            "hyang": 15,
            "little": 4,
            "nexus": 12,
            "quad": 4
        }
        self.grids_per_line = grids_per_line
        self.grid_size = grid_size
        self.add_empirical = add_empirical

    def gen_data(self, incl_rev=False):
        print("==> Generating Normal Data...")
        self._gen_data()
        if incl_rev:
            print("==> Generating Reversed Data...")
            self._gen_data(True)

    def _gen_data(self, reverse_frames=False):
        for scene_name, num in self.scene_nums.items():
            for idx in range(num):
                name = scene_name + "_%d" % idx
                file_name = name + "_g_" + str(self.grids_per_line) + "_s_" + str(self.grid_size) + "_rev_" + str(
                    reverse_frames) + "_emp_" + str(self.add_empirical) + ".pkl"
                save_dir = os.path.join("../dataset", "SDD", "processed_xoe", file_name)

                if os.path.exists(save_dir):
                    print(save_dir, "exists, skipped")
                    continue

                path = os.path.join("../dataset", "SDD", "raw", scene_name, "video%d" % idx, "annotations.txt")
                df = pd.read_csv(path, delimiter=" ", names=["ID", "xmin", "ymin", "xmax", "ymax", "frame",
                                                             "lost", "occluded", "generated", "label"])
                df = df[df["label"] == "Pedestrian"]
                df = df[df["frame"] % 12 == 0]

                data = np.array(df)[:, :-1].astype(float)
                data[:, 1] = (data[:, 1] + data[:, 3]) / 2
                data[:, 2] = (data[:, 2] + data[:, 4]) / 2
                data = data[:, [5, 0, 1, 2, 6, 7]]  # (frame, ID, x, y, lost, occluded)

                ped_ids = np.unique(data[:, 1])
                frames = np.unique(data[:, 0])

                trj_of_ped = {}
                full_data = {}

                normalized_frame_cnt = 0
                normalized_data = []
                for frame in sorted(frames, reverse=reverse_frames):
                    data_of_frame = data[np.where(data[:, 0] == frame)]
                    data_of_frame[:, 0] = normalized_frame_cnt
                    normalized_data.append(data_of_frame)
                    normalized_frame_cnt += 1

                normalized_data = np.concatenate(normalized_data, axis=0)
                data = normalized_data
                full_data["frame_num"] = normalized_frame_cnt

                peds_in_frame = {}
                for ped_id in ped_ids:
                    ped_trj = data[np.where(data[:, 1] == ped_id)]
                    for frame in ped_trj[:, 0]:
                        peds_in_frame[frame] = peds_in_frame.get(frame, []) + [ped_id]
                    ped_speed = np.zeros_like(ped_trj[:, 2:])
                    if len(ped_trj) > 1:
                        ped_speed[1:] = ped_trj[1:, 2:] - ped_trj[:-1, 2:]
                        ped_speed[0] = ped_speed[1]

                    ped_trj = np.concatenate([ped_trj, ped_speed], axis=-1)[:, [0, 2, 3, 6, 7]]  # (frame, x, y, vx, vy)
                    trj_of_ped[ped_id] = {"seq": ped_trj, "start_frame": ped_trj[0][0]}

                mask_path = os.path.join("../dataset", "SDD", "semantic_maps", name + "_mask.png")
                scene_mask = np.array(cv2.imread(mask_path))[..., 0].astype(float)
                formulated_data_of_ped = get_social_and_scene_info(copy.deepcopy(trj_of_ped), scene_mask,
                                                                    self.grids_per_line, self.grid_size,
                                                                    peds_in_frame, unit="pixels", add_empirical=self.add_empirical)

                full_data["trj_of_ped"] = trj_of_ped
                full_data["data_of_ped"] = formulated_data_of_ped
                with open(save_dir, "wb") as f:
                    pickle.dump(full_data, f, protocol=4)
                print("Saved", save_dir)


if __name__ == "__main__":
    x = eth_ucy_Processor()
    x.gen_data(incl_rev=True)
    x.gen_data(incl_rev=True, add_empirical=True)
    y = SDD_Processor()
    y.gen_data(incl_rev=True)
    y.gen_data(incl_rev=True, add_empirical=True)
