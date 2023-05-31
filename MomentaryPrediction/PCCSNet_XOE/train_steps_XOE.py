import time
import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import copy

from torch.utils.data import DataLoader
from PCCSNet_XOE.utils.data_process import transform_trajectory, get_scene_data
from PCCSNet_XOE.utils.utility import get_lr_scheduler, get_modality, display_performance, get_memory_data
from PCCSNet_XOE.utils.metrics import final_displacement_error, displacement_error, exp_l2_loss
from PCCSNet_XOE.utils.checkings import check_classifier_consistency
from PCCSNet_XOE.models.auxiliary import XOE_Trainer, PredEncoderTrainer
from PCCSNet_XOE.models.components import PCCSNet_XOE, PCCSNet
from PCCSNet_XOE.utils.data_process import collate_helper_AOE, collate_helper_MOE
from PCCSNet_XOE.train_steps import train_encoder, train_synthesizer, train_classifier, evaluate, gen_prediction


def train_XOE_encoder(net, train_loader, val_loader, perf_dict, save_dir_base, prefix, loss_fn, args):
    batch_num = len(train_loader)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr)
    scheduler = get_lr_scheduler(args.policy, optimizer, max_iter=batch_num * args.epochs)
    dict_key = prefix + "_Encoder"

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss, val_loss = 0, 0
        val_res = np.zeros(2).astype(float)
        net.train()
        for data in train_loader:
            if args.use_AOE:
                batch_trj, batch_obs, batch_sizes, batch_gt, seq_lens = \
                    data["obs_trj"].cuda(), data["obs_data"], data["data_sizes"], data["pred"].cuda(), data["seq_lens"]
            else:
                batch_trj, batch_obs, batch_sizes, batch_gt, seq_lens = \
                    data["obs_trj"].cuda(), data["obs_data"], data["data_sizes"], data["pred"].cuda(), None

            optimizer.zero_grad()
            trj_output = net(batch_trj, batch_obs, batch_sizes, seq_lens)
            trj_output = transform_trajectory(trj_output)

            loss = loss_fn(trj_output, batch_gt)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= batch_num

        net.eval()
        with torch.no_grad():
            for data in val_loader:
                if args.use_AOE:
                    batch_trj, batch_obs, batch_sizes, batch_gt, seq_lens = \
                        data["obs_trj"].cuda(), data["obs_data"], data["data_sizes"], data["pred"].cuda(), data["seq_lens"]
                else:
                    batch_trj, batch_obs, batch_sizes, batch_gt, seq_lens = \
                        data["obs_trj"].cuda(), data["obs_data"], data["data_sizes"], data["pred"].cuda(), None

                trj_output = net(batch_trj, batch_obs, batch_sizes, seq_lens)
                trj_output = transform_trajectory(trj_output)

                loss = loss_fn(trj_output, batch_gt)
                val_loss += loss.item()
                ADE = displacement_error(trj_output, batch_gt, mode="sum").item() / args.pred_len
                FDE = final_displacement_error(trj_output[:, -1], batch_gt[:, -1], mode="sum").item()
                val_res += [ADE, FDE]

        val_loss /= len(val_loader)
        val_res /= len(val_loader.dataset)

        if epoch % args.print_every == 0:
            end_time = time.time()
            print("==> Epoch: %d Train Loss: %.4f Val Loss: %.4f Val ADE: %.4f Val FDE: %.4f Time: %.4f"
                  % (epoch, train_loss, val_loss, val_res[0], val_res[1], end_time - start_time))

        if val_res[int(args.FDE_prioritize)] < perf_dict[dict_key][int(args.FDE_prioritize)]:
            perf_dict[dict_key][0], perf_dict[dict_key][1] = val_res[0], val_res[1]
            torch.save(net.xoe.state_dict(), os.path.join(save_dir_base, dict_key + "_XOE.pth"))
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

        val_loss /= len(val_loader)
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


def train_XOE(train_set, val_set, op_code, dataset_name, args):
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
        net = XOE_Trainer(args, input_dim=6 if args.use_AOE and args.add_empirical else 5).cuda()
        if args.pretrain:
            load_path = os.path.join(os.getcwd(), args.pretrain_dir, args.data_from, dataset_name, args.load_pretrain)
            print("==> Load Pretrain", load_path)
            pretrain_dict = torch.load(load_path)
            state_dict = net.state_dict()
            for key in pretrain_dict.keys():
                if "xoe." not in key:
                    continue
                if key in state_dict:
                    if pretrain_dict[key].size() == state_dict[key].size():
                        value = pretrain_dict[key]
                        if not isinstance(value, torch.Tensor):
                            value = value.data
                        state_dict[key] = value

        train_set.set_mode(0)
        val_set.set_mode(0)
        if args.use_AOE:
            train_loader = DataLoader(train_set, batch_size=args.bs, collate_fn=collate_helper_AOE)
            val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False, collate_fn=collate_helper_AOE)
        else:
            train_loader = DataLoader(train_set, batch_size=args.bs, collate_fn=collate_helper_MOE)
            val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False, collate_fn=collate_helper_MOE)

        train_XOE_encoder(net, train_loader, val_loader, perf_dict, save_dir, "Obs", loss_fn=nn.MSELoss(), args=args)
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
        net = PCCSNet_XOE(args, input_dim=6 if args.use_AOE and args.add_empirical else 5).cuda()
        net.load_encoders(save_dir)

        train_set.set_mode(0)
        val_set.set_mode(0)
        train_obs_encoding, train_pred_encoding, val_obs_encoding, val_pred_encoding = [], [], [], []
        if args.use_AOE:
            train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=False, collate_fn=collate_helper_AOE)
            val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False, collate_fn=collate_helper_AOE)
            for data in train_loader:
                batch_trj, batch_obs, batch_sizes, batch_gt, seq_lens = \
                    data["obs_trj"].cuda(), data["obs_data"], data["data_sizes"], data["pred"].cuda(), data["seq_lens"]
                obs_encoding, pred_encoding = net.gen_encoding(batch_trj, batch_obs, batch_sizes, batch_gt, seq_lens)
                train_obs_encoding.append(obs_encoding)
                train_pred_encoding.append(pred_encoding)
            for data in val_loader:
                batch_trj, batch_obs, batch_sizes, batch_gt, seq_lens = \
                    data["obs_trj"].cuda(), data["obs_data"], data["data_sizes"], data["pred"].cuda(), data["seq_lens"]
                obs_encoding, pred_encoding = net.gen_encoding(batch_trj, batch_obs, batch_sizes, batch_gt, seq_lens)
                val_obs_encoding.append(obs_encoding)
                val_pred_encoding.append(pred_encoding)
        else:
            train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=False, collate_fn=collate_helper_MOE)
            val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False, collate_fn=collate_helper_MOE)
            for data in train_loader:
                batch_trj, batch_obs, batch_sizes, batch_gt = \
                    data["obs_trj"].cuda(), data["obs_data"], data["data_sizes"], data["pred"].cuda()
                obs_encoding, pred_encoding = net.gen_encoding(batch_trj, batch_obs, batch_sizes, batch_gt)
                train_obs_encoding.append(obs_encoding)
                train_pred_encoding.append(pred_encoding)
            for data in val_loader:
                batch_trj, batch_obs, batch_sizes, batch_gt = \
                    data["obs_trj"].cuda(), data["obs_data"], data["data_sizes"], data["pred"].cuda()
                obs_encoding, pred_encoding = net.gen_encoding(batch_trj, batch_obs, batch_sizes, batch_gt)
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
