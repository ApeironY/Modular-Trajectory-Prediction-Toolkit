import numpy as np
import cmath
import copy
import cv2


def rotate_social(data, angle):  # data: n * 4;
    matrix = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
    rotated_pos = np.matmul(matrix, data[:, :2].transpose()).transpose()
    rotated_vel = np.matmul(matrix, data[:, 2:].transpose()).transpose()
    res = np.concatenate((rotated_pos, rotated_vel), axis=1)
    return res  # n * 4


def get_social_and_scene_info(trj_of_ped, scene_mask, grids_per_line, grid_size, peds_in_frame,
                              unit='pixels', scene_range=None, add_empirical=False):
    """
    Parameters
    ----------
    num_labels
    trj_of_ped: dict, {ped_id: {seq: [trj_len (variable) * (frame, x, y, v_x, v_y)], start_frame: seq[0][0]}, ...}
    scene_mask
    grids_per_line
    grid_size
    peds_in_frame: dict, {frame: [id1, id2, ...]...}
    unit: str, ('meters' or 'pixels')
    scene_range: ((x_min, x_max), (y_min, y_max)) if grid_size_unit == 'meters'
    add_empirical

    Returns
    -------

    """

    assert unit in ['meters', 'pixels']
    data_of_ped = {}
    half_grids_per_line = grids_per_line // 2

    pixels_y, pixels_x = scene_mask.shape
    if unit == "pixels":
        grid_size_x = grid_size_y = int(grid_size)
    else:
        assert scene_range is not None
        grid_size_x = int(grid_size / (scene_range[0][1] - scene_range[0][0]) * pixels_x)
        grid_size_y = int(grid_size / (scene_range[1][1] - scene_range[1][0]) * pixels_y)

    markers_x = np.linspace(-half_grids_per_line * grid_size_x, half_grids_per_line * grid_size_x, grids_per_line + 1)
    markers_y = np.linspace(-half_grids_per_line * grid_size_y, half_grids_per_line * grid_size_y, grids_per_line + 1)

    max_grid_rad = int((max(grid_size_x, grid_size_y) * half_grids_per_line) * np.sqrt(2) + 10)
    pad_size = max_grid_rad + 20
    scene_mask = np.pad(scene_mask, (pad_size, pad_size), 'constant')
    if add_empirical:
        formulated_data_base = np.zeros((grids_per_line, grids_per_line, 6)).tolist()
    else:
        formulated_data_base = np.zeros((grids_per_line, grids_per_line, 5)).tolist()


    for ped_id in trj_of_ped.keys():
        trj = trj_of_ped[ped_id]["seq"]

        frames, pos_seq, vel = trj[:, 0], trj[:, 1:3], trj[:, 3:5]
        if unit == 'meters':
            pos_seq[:, 0] = (pos_seq[:, 0] - scene_range[0][0]) / (scene_range[0][1] - scene_range[0][0]) * pixels_x
            pos_seq[:, 1] = (pos_seq[:, 1] - scene_range[1][0]) / (scene_range[1][1] - scene_range[1][0]) * pixels_y
            vel[:, 0] = vel[:, 0] / (scene_range[0][1] - scene_range[0][0]) * pixels_x
            vel[:, 1] = vel[:, 1] / (scene_range[1][1] - scene_range[1][0]) * pixels_y

        pos_seq += pad_size
        # print(ped_id, pos_seq)
        res = []
        for i in range(len(trj)):
            cpx = complex(vel[i][1], vel[i][0])
            rho, theta = cmath.polar(cpx)

            occupation = np.zeros((grids_per_line, grids_per_line)).astype(bool)
            formulated_data = copy.deepcopy(formulated_data_base)
            c_X = int(pos_seq[i][0])
            c_Y = int(pos_seq[i][1])
            scene_around_ped = copy.deepcopy(
                scene_mask[c_Y - max_grid_rad:c_Y + max_grid_rad, c_X - max_grid_rad:c_X + max_grid_rad]
            )

            assert scene_around_ped.shape[0] == scene_around_ped.shape[1] == max_grid_rad * 2, \
                (scene_mask.shape, scene_around_ped.shape, c_X, c_Y, max_grid_rad)

            M = cv2.getRotationMatrix2D((max_grid_rad, max_grid_rad), -theta / np.pi * 180, 1.0)
            scene_around_ped = cv2.warpAffine(np.float32(scene_around_ped), M, (2 * max_grid_rad, 2 * max_grid_rad))
            scene_around_ped_trunc = \
                scene_around_ped[
                    max_grid_rad - half_grids_per_line * grid_size_y:max_grid_rad + half_grids_per_line * grid_size_y,
                    max_grid_rad - half_grids_per_line * grid_size_x:max_grid_rad + half_grids_per_line * grid_size_x,
                ]

            for idx1 in range(grids_per_line):
                for idx2 in range(grids_per_line):
                    grid_area = scene_around_ped_trunc[idx1 * grid_size_y:(idx1+1) * grid_size_y, idx2 * grid_size_x:(idx2+1) * grid_size_x]
                    label_idx = grid_area[grid_size_y // 2, grid_size_x // 2]
                    formulated_data[idx1][idx2][0] = np.around(label_idx)

            present_ped_list = peds_in_frame[trj[i, 0]]
            present_ped_states = []
            
            for present_ped_id in present_ped_list:
                if present_ped_id == ped_id:
                    continue
                ped_start_frame = trj_of_ped[present_ped_id]["start_frame"]
                ped_state = copy.deepcopy(trj_of_ped[present_ped_id]["seq"][int(trj[i, 0] - ped_start_frame)][1:5])
                ped_state[:2] = ped_state[:2] - trj[i][1:3]  # (rel_x, rel_y, vx, vy)
                present_ped_states.append(ped_state)

            if len(present_ped_states) != 0:
                present_ped_states = np.stack(present_ped_states, axis=0)  # num_ped * (rel_x, rel_y, vx, vy)
                present_ped_states = rotate_social(present_ped_states, theta)

                for state in present_ped_states:
                    state_cp = copy.deepcopy(state)
                    bound_x = np.where(((markers_x[:-1] <= state_cp[0]) & (state_cp[0] < markers_x[1:])) == 1)[0]
                    if len(bound_x) == 1:
                        bound_y = np.where(((markers_y[:-1] <= state_cp[1]) & (state_cp[1] < markers_y[1:])) == 1)[0]
                        if len(bound_y) == 1:
                            state_cp[0] -= (markers_x[bound_x[0]] + markers_x[bound_x[0] + 1]) / 2
                            state_cp[1] -= (markers_y[bound_y[0]] + markers_y[bound_y[0] + 1]) / 2

                            if not add_empirical:
                                if occupation[bound_y[0], bound_x[0]]:
                                    semantic_label = formulated_data[0][0][0]
                                    formulated_data[bound_y[0]][bound_x[0]] = formulated_data[bound_y[0]][bound_x[0]] + [semantic_label] + state_cp.tolist()
                                else:
                                    formulated_data[bound_y[0]][bound_x[0]][1:] = state_cp.tolist()
                                    occupation[bound_y[0], bound_x[0]] = True
                            else:
                                if occupation[bound_y[0], bound_x[0]]:
                                    semantic_label = formulated_data[0][0][0]
                                    formulated_data[bound_y[0]][bound_x[0]] = formulated_data[bound_y[0]][bound_x[0]] + [semantic_label] + state_cp.tolist() + [0]
                                else:
                                    formulated_data[bound_y[0]][bound_x[0]][1:5] = state_cp.tolist()
                                    occupation[bound_y[0], bound_x[0]] = True
                                    formulated_data[bound_y[0]][bound_x[0]][5] += 1

            if add_empirical:
                count_sum = 0
                for t_x in range(grids_per_line):
                    for t_y in range(grids_per_line):
                        count_sum += formulated_data[t_x][t_y][5]

                for t_x in range(grids_per_line):
                    for t_y in range(grids_per_line):
                        if count_sum != 0:
                            formulated_data[t_x][t_y][5] /= count_sum
                        if len(formulated_data[t_x][t_y]) > 6:
                            for k in range(len(formulated_data[t_x][t_y]) // 6 - 1):
                                formulated_data[t_x][t_y][5 + 6 * k] = formulated_data[t_x][t_y][5]

            res.append(formulated_data)
        data_of_ped[ped_id] = res

    return data_of_ped



