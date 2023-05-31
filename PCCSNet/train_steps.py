import time
import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import copy

from torch.utils.data import DataLoader
from utils.data_process import transform_trajectory, get_scene_data
from utils.utility import get_lr_scheduler, get_modality, display_performance, get_memory_data
from utils.metrics import final_displacement_error, displacement_error, identity_loss, exp_l2_loss
from utils.checkings import check_classifier_consistency
from models.auxiliary import ObsEncoderTrainer, PredEncoderTrainer
from models.components import PCCSNet


def train_encoder(net, train_loader, val_loader, perf_dict, save_dir_base, prefix, loss_fn, args):
    batch_num = len(train_loader)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr)
    scheduler = get_lr_scheduler(args.policy, optimizer, max_iter=batch_num * args.epochs)
    dict_key = prefix + "_Encoder"

    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss, val_loss = 0, 0
        val_res = np.zeros(2).astype(float)
        net.train()
        for data in train_loader:
            batch_data, batch_data_gt = data["input"].cuda(), data["future"].cuda()
            optimizer.zero_grad()
            net_output = net(batch_data)
            net_output = transform_trajectory(net_output)
            loss = loss_fn(net_output, batch_data_gt)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= batch_num

        net.eval()
        with torch.no_grad():
            for data in val_loader:
                batch_data, batch_data_gt = data["input"].cuda(), data["future"].cuda()
                net_output = net(batch_data)
                net_output = transform_trajectory(net_output)
                loss = loss_fn(net_output, batch_data_gt)
                val_loss += loss.item()
                ADE = displacement_error(net_output, batch_data_gt, mode="sum").item() / args.pred_len
                FDE = final_displacement_error(net_output[:, -1], batch_data_gt[:, -1], mode="sum").item()
                val_res += [ADE, FDE]

        val_loss /= len(val_loader)
        val_res /= len(val_loader.dataset)

        if epoch % args.print_every == 0:
            end_time = time.time()
            print("==> Epoch: %d Train Loss: %.4f Val Loss: %.4f Val ADE: %.4f Val FDE: %.4f Time: %.4f"
                  % (epoch, train_loss, val_loss, val_res[0], val_res[1], end_time - start_time))
            start_time = end_time

        if val_res[int(args.FDE_prioritize)] < perf_dict[dict_key][int(args.FDE_prioritize)]:
            perf_dict[dict_key][0], perf_dict[dict_key][1] = val_res[0], val_res[1]
            torch.save(net.encoder.state_dict(), os.path.join(save_dir_base, dict_key + ".pth"))
            with open(os.path.join(save_dir_base, "Performances.pkl"), "wb") as f:
                pickle.dump(perf_dict, f, 4)
            print("==> Saved")


def train_decoder(net, train_loader, val_loader, perf_dict, save_dir_base, loss_fn, args):
    if not (perf_dict["Obs_Encoder"] == perf_dict["Decoder"][0]
            and perf_dict["Pred_Encoder"] == perf_dict["Decoder"][1]) or args.retrain:
        print("==> Encoders and Decoder Do NOT Match, Retrain Decoder...")
        perf_dict["Decoder"][0] = copy.deepcopy(perf_dict["Obs_Encoder"])
        perf_dict["Decoder"][1] = copy.deepcopy(perf_dict["Pred_Encoder"])
        perf_dict["Decoder"][2] = [1e3, 1e3]

    batch_num = len(train_loader)
    optimizer = torch.optim.Adam(net.decoder.parameters(), lr=args.base_lr)
    scheduler = get_lr_scheduler(args.policy, optimizer, max_iter=batch_num * args.epochs)

    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss, val_loss = 0, 0
        val_res = np.zeros(2).astype(float)

        net.train()
        for data in train_loader:
            batch_obs_encoding, batch_pred_encoding, batch_trj_gt \
                = data["obs_enc"].cuda(), data["pred_enc"].cuda(), data["future"].cuda()
            optimizer.zero_grad()
            net_output = net.decoder([batch_obs_encoding, batch_pred_encoding])  # bs * 12 * 2
            net_output = transform_trajectory(net_output)
            loss = loss_fn(net_output, batch_trj_gt)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= batch_num

        net.eval()
        with torch.no_grad():
            for data in val_loader:
                batch_obs_encoding, batch_pred_encoding, batch_trj_gt \
                    = data["obs_enc"].cuda(), data["pred_enc"].cuda(), data["future"].cuda()
                net_output = net.decoder([batch_obs_encoding, batch_pred_encoding])
                net_output = transform_trajectory(net_output)
                loss = loss_fn(net_output, batch_trj_gt)
                val_loss += loss.item()
                ADE = displacement_error(net_output, batch_trj_gt, mode="sum").item() / args.pred_len
                FDE = final_displacement_error(net_output[:, -1], batch_trj_gt[:, -1], mode="sum").item()
                val_res += [ADE, FDE]

        val_loss /= len(val_loader.dataset)
        val_res /= len(val_loader.dataset)

        if epoch % args.print_every == 0:
            end_time = time.time()
            print("==> Epoch: %d Train Loss: %.4f Val Loss: %.4f ADE: %.4f FDE: %.4f Time: %.4f"
                  % (epoch, train_loss, val_loss, val_res[0], val_res[1], end_time - start_time))
            start_time = end_time

        if val_res[int(args.FDE_prioritize)] < perf_dict["Decoder"][2][int(args.FDE_prioritize)]:
            perf_dict["Decoder"][2][0], perf_dict["Decoder"][2][1] = val_res[0], val_res[1]
            torch.save(net.decoder.decoder.state_dict(), os.path.join(save_dir_base, "decoder.pth"))
            torch.save(net.decoder.fc.state_dict(), os.path.join(save_dir_base, "fc.pth"))
            with open(os.path.join(save_dir_base, "Performances.pkl"), "wb") as f:
                pickle.dump(perf_dict, f, 4)
            print("==> Saved")


def train_classifier(net, train_loader, save_dir, args):
    batch_num = len(train_loader)
    optimizer = torch.optim.Adam(net.classifier.parameters(), lr=args.base_lr)
    scheduler = get_lr_scheduler(args.policy, optimizer, max_iter=batch_num * args.epochs)

    for epoch in range(args.epochs):
        t1 = time.time()
        train_loss = 0
        net.train()
        for _, batch_data in enumerate(train_loader):
            batch_features, batch_prob = batch_data["obs_enc"].cuda(), batch_data["prob"].cuda()
            optimizer.zero_grad()
            net_output = net.classifier(batch_features)
            loss = -torch.mean(torch.sum(batch_prob * torch.log(net_output + 1e-8), dim=1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= batch_num

        if epoch % args.print_every == 0:
            t2 = time.time()
            print("==> Epoch: %d Train Loss: %.4f Time: %.4f" % (epoch, train_loss, t2 - t1))

    torch.save(net.classifier.state_dict(), os.path.join(save_dir, "classifier.pth"))


def train_synthesizer(net, train_loader, all_modalities, dataset_name, perf_dict,
                      save_dir_base, args, val_loader):
    if not perf_dict["Decoder"] == perf_dict["Synthesizer"][:3] or args.retrain:
        print("==> Synthesizer Outdated, Retrain...")
        perf_dict["Synthesizer"][:3] = copy.deepcopy(perf_dict["Decoder"])
        perf_dict["Synthesizer"][3] = [1e3, 1e3]

    batch_num = len(train_loader)
    optimizer = torch.optim.Adam(net.synthesizer.parameters(), lr=args.base_lr)
    scheduler = get_lr_scheduler(args.policy, optimizer, max_iter=batch_num * args.epochs)
    for epoch in range(args.epochs):
        train_loss = 0
        t1 = time.time()
        net.train()
        for data in train_loader:
            batch_data, batch_memory_gt, batch_labels = \
                data["obs_enc"].cuda(), data["pred_enc"].cuda(), data["cluster_idx"]
            optimizer.zero_grad()
            pred_encoding = net.synthesizer(batch_data, batch_labels, all_modalities)
            loss = identity_loss(pred_encoding, batch_memory_gt.cuda())
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= batch_num

        if epoch >= int(0.1 * args.epochs):
            net.eval()
            val_res = evaluate(net, val_loader, all_modalities, dataset_name, args)
            if val_res[int(args.FDE_prioritize)] < perf_dict["Synthesizer"][3][int(args.FDE_prioritize)]:
                perf_dict["Synthesizer"][3][0] = val_res[0]
                perf_dict["Synthesizer"][3][1] = val_res[1]
                torch.save(net.synthesizer.fc1.state_dict(), os.path.join(save_dir_base, "synthesizer_fc1.pth"))
                torch.save(net.synthesizer.fc2.state_dict(), os.path.join(save_dir_base, "synthesizer_fc2.pth"))
                with open(os.path.join(save_dir_base, "Performances.pkl"), "wb") as f:
                    pickle.dump(perf_dict, f, 4)
                print("==> Saved")

        t2 = time.time()
        if epoch % args.print_every == 0:
            print("==> Epoch: %d Train Loss: %.4f Time: %.4f" % (epoch, train_loss, t2 - t1))


def evaluate(net, val_loader, all_modalities, dataset_name, args):
    k_list = args.eval_topK
    assert len(k_list) > 0
    val_res = np.zeros(2).astype(float)
    mode_dict = {
        0: "ADE-prioritized",
        1: "FDE-prioritized",
        2: "Equal-Focus",
    }
    for k in k_list:
        start = time.time()
        val_res = np.zeros(2).astype(float)
        for data in val_loader:
            batch_obs_encoding, batch_pred_gt = data["obs_enc"].cuda(), data["pred"].cuda()
            net_output = net(batch_obs_encoding, k=k, all_modalities=all_modalities)
            net_output = transform_trajectory(net_output)
            assert len(net_output) == k
            ADE = displacement_error(net_output, batch_pred_gt) / args.pred_len
            FDE = final_displacement_error(net_output[:, -1], batch_pred_gt[:, -1])

            if args.eval_mode == 0:
                idx = torch.argmin(ADE)
                min_ADE, min_FDE = ADE[idx].item(), FDE[idx].item()
            elif args.eval_mode == 1:
                idx = torch.argmin(FDE)
                min_ADE, min_FDE = ADE[idx].item(), FDE[idx].item()
            else:
                min_ADE = torch.min(ADE).item()
                min_FDE = torch.min(FDE).item()

            val_res += [min_ADE, min_FDE]
        val_res /= len(val_loader)

        end = time.time()
        print("==> Dataset: %s TopK: %d Mode: %s ADE: %.4f FDE: %.4f Time: %.4f"
              % (dataset_name, k, mode_dict[args.eval_mode], val_res[0], val_res[1], end - start))

    return val_res


def gen_prediction(net, loader, all_modalities, dataset_name, args):
    k_list = args.eval_topK
    assert len(k_list) > 0
    with torch.no_grad():
        for k in k_list:
            start = time.time()
            full_output = []
            for data in loader:
                batch_obs_encoding, mat, last_obs_pos = \
                    data["obs_enc"].cuda(), data["rotate_mat"].cuda(), data["last_obs"].cuda()
                net_output = net(batch_obs_encoding, k=k, all_modalities=all_modalities)
                net_output = transform_trajectory(net_output)
                output = torch.bmm(mat.repeat(k, 1, 1), net_output.transpose(1, 2)).transpose(1, 2)
                output += last_obs_pos.unsqueeze(0)
                full_output.append(output.cpu().numpy())
            full_output = np.stack(full_output)
            end = time.time()
            file_name = "all_output_%s_top%d.npy" % (dataset_name, k)
            print("==> Dataset: " + dataset_name + " TopK: %d Time: %.4f FileName: %s"
                  % (k, end - start, file_name), full_output.shape)
            np.save(file_name, full_output)


def train(train_set, val_set, op_code, dataset_name, args):
    save_dir = os.path.join(args.save_dir, args.data_from, dataset_name)

    if os.path.exists(os.path.join(save_dir, "Performances.pkl")):
        with open(os.path.join(save_dir, "Performances.pkl"), "rb") as f:
            perf_dict = pickle.load(f)
        display_performance(perf_dict)
    else:
        perf_dict = {
            "Obs_Encoder": [1e3, 1e3],
            "Pred_Encoder": [1e3, 1e3],
            "Decoder": [[1e3, 1e3]] * 3,
            "Synthesizer": [[1e3, 1e3]] * 4
        }

    if op_code == 0:
        if args.retrain:
            print("Retrain Obs Encoder")
            perf_dict["Obs_Encoder"] = [1e3, 1e3]

        print("==> Training Obs_Encoder...")
        start_time = time.time()
        net = ObsEncoderTrainer(obs_len=args.obs_len, pre_len=args.pred_len,
                                hidden_size=args.obs_feat_size, num_layer=args.encoder_layer).cuda()
        train_set.set_mode(0)
        val_set.set_mode(0)
        train_loader = DataLoader(train_set, batch_size=args.bs)
        val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False)

        train_encoder(net, train_loader, val_loader, perf_dict, save_dir, "Obs", loss_fn=nn.MSELoss(), args=args)
        end_time = time.time()
        print("==> Time: %.2f" % (end_time - start_time))

    elif op_code == 1:
        print("==> Training Pred_Encoder...")
        start_time = time.time()
        net = PredEncoderTrainer(obs_len=args.obs_len, pre_len=args.pred_len,
                                 hidden_size=args.pred_feat_size, num_layer=args.encoder_layer).cuda()
        train_set.set_mode(1)
        val_set.set_mode(1)
        train_loader = DataLoader(train_set, batch_size=args.bs)
        val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False)

        train_encoder(net, train_loader, val_loader,
                      perf_dict, save_dir, "Pred", loss_fn=nn.MSELoss(), args=args)
        end_time = time.time()
        print("==> Time: %.2f" % (end_time - start_time))

    elif op_code in [2, 3, 4, 5]:
        net = PCCSNet(obs_len=args.obs_len, pre_len=args.pred_len, n_cluster=args.n_cluster,
                      obs_hidden_size=args.obs_feat_size, pred_hidden_size=args.pred_feat_size,
                      num_layer=args.encoder_layer).cuda()

        net.load_encoders(save_dir)

        train_set.set_mode(0)
        val_set.set_mode(0)
        train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False)

        train_obs_encoding, train_pred_encoding, val_obs_encoding, val_pred_encoding = [], [], [], []
        for data in train_loader:
            batch_data_obs, batch_data_pred = data["input"].cuda(), data["future"].cuda()
            obs_encoding, pred_encoding = net.gen_encoding(batch_data_obs, batch_data_pred)
            train_obs_encoding.append(obs_encoding)
            train_pred_encoding.append(pred_encoding)
        for data in val_loader:
            batch_data_obs, batch_data_pred = data["input"].cuda(), data["future"].cuda()
            obs_encoding, pred_encoding = net.gen_encoding(batch_data_obs, batch_data_pred)
            val_obs_encoding.append(obs_encoding)
            val_pred_encoding.append(pred_encoding)

        train_set.obs_enc = torch.cat(train_obs_encoding, dim=0)
        train_set.pred_enc = torch.cat(train_pred_encoding, dim=0)
        val_set.obs_enc = torch.cat(val_obs_encoding, dim=0)
        val_set.pred_enc = torch.cat(val_pred_encoding, dim=0)

        if op_code == 2:  # Train Decoder
            print("==> Training Decoder...")
            start_time = time.time()
            train_set.set_mode(2)
            val_set.set_mode(2)
            train_loader = DataLoader(train_set, batch_size=args.bs)
            val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False)

            train_decoder(net, train_loader, val_loader, perf_dict, save_dir, loss_fn=exp_l2_loss, args=args)
            end_time = time.time()
            print("==> Time: %.2f" % (end_time - start_time))

        else:
            memory_data, cluster_result, size_array = get_memory_data(train_set, save_dir, args)
            memory_data = torch.from_numpy(memory_data)
            train_set.cluster_results = torch.from_numpy(cluster_result).long()
            net.load_decoder(save_dir)

            if op_code == 3:  # Train Classifier
                print("==> Training Classifier...")
                start_time = time.time()
                data_size = len(train_set)
                if args.disable_modality_loss:
                    gt_prob = torch.zeros(data_size, args.n_cluster)
                    gt_prob[list(range(data_size)), cluster_result] = 1
                else:
                    gt_prob = get_scene_data(dataset_name, train_set, cluster_result, save_dir, args)
                    gt_prob = torch.from_numpy(gt_prob)

                train_set.classifier_gt = gt_prob
                train_set.set_mode(4)
                train_loader = DataLoader(train_set, batch_size=args.bs)
                train_classifier(net, train_loader, save_dir, args)
                end_time = time.time()
                print("==> Saved. Time: %.2f" % (end_time - start_time))

            else:
                net.load_classifier(save_dir)
                check_classifier_consistency(save_dir, args)

                val_set.set_mode(5)
                val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
                all_modalities = get_modality(memory_data, cluster_result, args.n_cluster).cuda()

                if op_code == 4:  # Train Modality Synthesizer
                    print("==> Training Modality Synthesizer...")
                    start_time = time.time()
                    train_set.set_mode(3)
                    train_loader = DataLoader(train_set, batch_size=args.bs)
                    train_synthesizer(net, train_loader, all_modalities, dataset_name, perf_dict,
                                      save_dir, args, val_loader)
                    end_time = time.time()
                    print("==> Time: %.2f" % (end_time - start_time))

                elif op_code == 5:  # Eval
                    print("==> Evaluation")
                    net.load_synthesizer(save_dir)
                    net.eval()
                    evaluate(net, val_loader, all_modalities, dataset_name, args)

    elif op_code == 6:
        net = PCCSNet(obs_len=args.obs_len, pre_len=args.pred_len, n_cluster=args.n_cluster,
                      obs_hidden_size=args.obs_feat_size, pred_hidden_size=args.pred_feat_size,
                      num_layer=args.encoder_layer).cuda()

        net.load_models(save_dir)
        check_classifier_consistency(save_dir, args)
        val_set.set_mode(0)
        loader = DataLoader(val_set, batch_size=args.bs, shuffle=False)

        obs_encodings = []
        for data in loader:
            batch_data_obs = data["input"].cuda()
            obs_encoding, _ = net.gen_encoding(batch_data_obs, None, for_training=False)
            obs_encodings.append(obs_encoding)
        val_set.obs_enc = torch.cat(obs_encodings, dim=0)

        memory_data, cluster_result, size_array = get_memory_data(None, save_dir, args)
        memory_data = torch.from_numpy(memory_data)
        all_modalities = get_modality(memory_data, cluster_result, args.n_cluster).cuda()

        val_set.set_mode(6)
        loader = DataLoader(val_set, batch_size=1, shuffle=False)
        gen_prediction(net, loader, all_modalities, dataset_name, args)

    else:
        return
