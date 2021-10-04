import numpy as np
import os


class eth_ucy_Processor:
    def __init__(self, obs_len, pred_len):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.total_len = obs_len + pred_len
        self.file_names = ["eth", "hotel", "univ-001", "univ-003", "univ-example", "zara1", "zara2", "zara3"]

    def gen_data(self):
        for name in self.file_names:
            path = os.path.join("dataset", "ethucy", "raw", name + ".csv")
            data = np.genfromtxt(path, delimiter=",").transpose()

            ped_ids = np.unique(data[:, 1])

            all_trajectories = []
            for ped_id in ped_ids:
                ped_traj = data[np.where(data[:, 1] == ped_id)]

                if len(ped_traj) < 2:
                    continue

                if len(ped_traj) >= self.total_len:
                    for i in range(len(ped_traj)+1-self.total_len):
                        all_trajectories.append(ped_traj[i:i+self.total_len])

            all_trajectories = np.array(all_trajectories)

            data_path = os.path.join("dataset", "ethucy", "processed", name + "_" + str(self.obs_len) + "_" + str(self.pred_len) + ".npy")
            np.save(data_path, all_trajectories)
            print("==> Processed & Saved", path, all_trajectories.shape)


if __name__ == "__main__":
    x = eth_ucy_Processor(8, 12)
    x.gen_data()
