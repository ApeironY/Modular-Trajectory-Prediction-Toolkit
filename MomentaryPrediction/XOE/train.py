import torch
import torch.nn as nn
import numpy as np
import os
import time
import pickle
from XOE.utils.data_process import collate_helper_aoe, collate_helper_moe, transform_trajectory
from XOE.utils.utility import get_lr_scheduler, displacement_error, final_displacement_error
from torch.utils.data import DataLoader
from XOE.models.xoe import XOE_Trainer, BehaviorExtractorTrainer, IntentionExtractorTrainer


def train_behavior_extractor(net, train_loader, save_dir_base, loss_fn, args):
    batch_num = len(train_loader)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, eps=1e-3)
    scheduler = get_lr_scheduler(args.policy, optimizer, max_iter=batch_num * args.epochs)

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = 0
        net.train()
        for data in train_loader:
            batch_trj, batch_gt, seq_lens = data["obs_trj"].cuda(), data["pred"].cuda(), data["seq_lens"]

            optimizer.zero_grad()
            logits = net(batch_trj[..., :2], batch_gt, seq_lens)
            # print(obs_encoded)
            # print(pred_encoded)
            labels = torch.arange(batch_trj.shape[0]).cuda()

            loss = (loss_fn(logits, labels) + loss_fn(logits.T, labels)) / 2
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= batch_num

        end_time = time.time()
        print("==> Epoch: %d Train Loss: %.4f Time: %.4f" % (epoch, train_loss, end_time - start_time))

        torch.save(net.encoder_obs.state_dict(), os.path.join(save_dir_base, 'behaviorextractor.pth'))


def train_intention_extractor(net, train_loader, val_loader, perf_dict, save_dir_base, loss_fn, args):
    batch_num = len(train_loader)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr)
    scheduler = get_lr_scheduler(args.policy, optimizer, max_iter=batch_num * args.epochs)
    dict_key = "Intention"

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss, val_loss = 0, 0
        val_res = np.zeros(1).astype(float)
        net.train()
        for data in train_loader:
            batch_trj, batch_gt, seq_lens = data["obs_trj"].cuda(), data["pred"].cuda(), data["seq_lens"]

            optimizer.zero_grad()
            intention_point = net(batch_trj[..., :2], seq_lens)
            intention_point = intention_point.squeeze(0)

            loss = loss_fn(intention_point, batch_gt[:, -1])
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= batch_num

        net.eval()
        with torch.no_grad():
            for data in val_loader:
                batch_trj, batch_gt, seq_lens = data["obs_trj"].cuda(), data["pred"].cuda(), data["seq_lens"]
                intention_point = net(batch_trj[..., :2], seq_lens)
                intention_point = intention_point.squeeze(0)

                loss = loss_fn(intention_point, batch_gt[:, -1])
                val_loss += loss.item()
                FDE = final_displacement_error(intention_point, batch_gt[:, -1], mode="sum").item()
                val_res[0] += FDE

        val_loss /= len(val_loader)
        val_res /= len(val_loader.dataset)

        end_time = time.time()
        print("==> Epoch: %d Train Loss: %.4f Val Loss: %.4f FDE: %.4f Time: %.4f"
                % (epoch, train_loss, val_loss, val_res[0], end_time - start_time))

        if val_res[0] < perf_dict[dict_key][0]:
            perf_dict[dict_key][0] = val_res[0]
            torch.save(net.encoder.state_dict(), os.path.join(save_dir_base, "intentionextractor.pth"))
            with open(os.path.join(save_dir_base, "Performances.pkl"), "wb") as f:
                pickle.dump(perf_dict, f, 4)
            print("==> Saved")


def pretrain_AOE(train_dataset, val_dataset, op_code, args, dataset_name):
    save_dir_base = os.path.join(args.save_dir, args.data_from, dataset_name)

    if os.path.exists(os.path.join(save_dir_base, "Performances.pkl")):
        with open(os.path.join(save_dir_base, "Performances.pkl"), 'rb') as f:
            perf_dict = pickle.load(f)
        print("Intention FDE:", perf_dict["Intention"][0])
    else:
        perf_dict = {
            "Intention": [1e3]
        }

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, collate_fn=collate_helper_aoe)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, collate_fn=collate_helper_aoe)

    if op_code == 0:
        print("==> Training Behavior Extractor")
        if not args.add_behavior:
            print("==> No Behavior, please set add_behavior to True")
            exit()
        start_time = time.time()
        net = BehaviorExtractorTrainer(hidden_size=args.obs_feat_size, num_layer=args.num_layer).cuda()

        train_behavior_extractor(net, train_loader, save_dir_base, loss_fn=nn.CrossEntropyLoss(), args=args)
        end_time = time.time()
        print("==> Time: %.2f" % (end_time - start_time))

    elif op_code == 1:
        print("==> Training Intention Extractor")
        if not args.add_intention:
            print("==> No Intention, please set add_intention to True")
            exit()
        start_time = time.time()
        net = IntentionExtractorTrainer(hidden_size=args.obs_feat_size, num_layer=args.num_layer).cuda()

        train_intention_extractor(net, train_loader, val_loader, perf_dict, save_dir_base, loss_fn=nn.MSELoss(), args=args)
        end_time = time.time()
        print("==> Time: %.2f" % (end_time - start_time))

    elif op_code == 2:
        net = XOE_Trainer(args, num_labels=6 if args.data_from == "SDD" else 2, input_dim=6 if args.add_empirical else 5).cuda()
        net.loadencoder(save_dir_base)
        batch_num = len(train_loader)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr)
        scheduler = get_lr_scheduler(args.policy, optimizer, max_iter=batch_num * args.epochs)

        ce_loss = nn.CrossEntropyLoss()
        trj_loss = nn.MSELoss()
        mask_div = 20 if args.data_from == "SDD" else 1

        print("==> Begin Training")
        for epoch in range(0, args.epochs):
            t1 = time.time()
            train_loss_mask, train_loss_recon, val_loss_mask, val_loss_recon = 0, 0, 0, 0
            val_res = np.zeros(2).astype(float)
            net.train()
            for data in train_loader:
                batch_trj, batch_obs, batch_sizes, batch_gt, scene_recon_gt, ped_recon_gt, seq_lens = \
                    data["obs_trj"].cuda(), data["obs_data"], data["data_sizes"], data["pred"].cuda(), data["scene_recon_gt"].cuda(), data["ped_recon_gt"].cuda(), data["seq_lens"]

                optimizer.zero_grad()
                trj_output, scene_logits, ped_logits = net(batch_trj, batch_obs, batch_sizes, gt_data=batch_gt, seq_lens=seq_lens)
                trj_output = transform_trajectory(trj_output)

                loss = trj_loss(trj_output, batch_gt) / mask_div
                mask_loss = loss.item()
                for i in range(args.grid_num ** 2):
                    loss += 0.3 * (ce_loss(scene_logits[:, i], scene_recon_gt[:, i]) + ce_loss(ped_logits[:, i], ped_recon_gt[:, i]))

                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss_recon += loss.item() - mask_loss
                train_loss_mask += mask_loss

            train_loss_recon /= batch_num
            train_loss_mask /= batch_num

            net.eval()
            with torch.no_grad():
                for data in val_loader:
                    batch_trj, batch_obs, batch_sizes, batch_gt, scene_recon_gt, ped_recon_gt, seq_lens = \
                        data["obs_trj"].cuda(), data["obs_data"], data["data_sizes"], data["pred"].cuda(), data["scene_recon_gt"].cuda(), data["ped_recon_gt"].cuda(), data["seq_lens"]

                    trj_output, scene_logits, ped_logits = net(batch_trj, batch_obs, batch_sizes, gt_data=batch_gt, seq_lens=seq_lens)
                    trj_output = transform_trajectory(trj_output)

                    loss = trj_loss(trj_output, batch_gt) / mask_div
                    mask_loss = loss.item()
                    for i in range(args.grid_num ** 2):
                        loss += 0.3 * (ce_loss(scene_logits[:, i], scene_recon_gt[:, i]) + ce_loss(ped_logits[:, i], ped_recon_gt[:, i]))

                    val_loss_recon += loss.item() - mask_loss
                    val_loss_mask += mask_loss

                    ADE = displacement_error(trj_output, batch_gt, mode="sum").item() / args.pred_len
                    FDE = final_displacement_error(trj_output[:, -1], batch_gt[:, -1], mode="sum").item()
                    val_res += [ADE, FDE]

            val_loss_recon /= len(val_loader)
            val_loss_mask /= len(val_loader)
            val_res /= len(val_loader.dataset)

            if True:
                print("==> Epoch: %d Train Loss M: %.4f Train Loss R: %.4f Train Loss: %.4f; Val Loss M: %.4f Val Loss R: %.4f Val Loss: %.4f; Val ADE: %.4f Val FDE: %.4f; Time: %.4f"
                    % (epoch, train_loss_mask, train_loss_recon, train_loss_recon + train_loss_mask, val_loss_mask, val_loss_recon,
                        val_loss_recon + val_loss_mask, val_res[0], val_res[1], time.time() - t1))

            torch.save(net.state_dict(), os.path.join(save_dir_base, "ep_%d.pth" % epoch))
    else:
        return


def pretrain_MOE(train_dataset, val_dataset, args, dataset_name):
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, collate_fn=collate_helper_moe)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, collate_fn=collate_helper_moe)
    net = XOE_Trainer(args, num_labels=6 if args.data_from == "SDD" else 2).cuda()
    batch_num = len(train_loader)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr)
    scheduler = get_lr_scheduler(args.policy, optimizer, max_iter=batch_num * args.epochs)

    ce_loss = nn.CrossEntropyLoss()
    trj_loss = nn.MSELoss()
    save_dir_base = os.path.join(args.save_dir, args.data_from, dataset_name)
    mask_div = 20 if args.data_from == "SDD" else 1

    print("==> Begin Training")
    for epoch in range(0, args.epochs):
        t1 = time.time()
        train_loss_mask, train_loss_recon, val_loss_mask, val_loss_recon = 0, 0, 0, 0
        val_res = np.zeros(2).astype(float)
        net.train()
        for data in train_loader:
            batch_trj, batch_obs, batch_sizes, batch_gt, scene_recon_gt, ped_recon_gt = \
                data["obs_trj"].cuda(), data["obs_data"], data["data_sizes"], data["pred"].cuda(), data["scene_recon_gt"].cuda(), data["ped_recon_gt"].cuda()

            optimizer.zero_grad()
            trj_output, scene_logits, ped_logits = net(batch_trj, batch_obs, batch_sizes, batch_gt)
            trj_output = transform_trajectory(trj_output)

            loss = trj_loss(trj_output, batch_gt) / mask_div
            mask_loss = loss.item()
            for i in range(args.grid_num ** 2):
                loss += 0.3 * (ce_loss(scene_logits[:, i], scene_recon_gt[:, i]) + ce_loss(ped_logits[:, i], ped_recon_gt[:, i]))

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss_recon += loss.item() - mask_loss
            train_loss_mask += mask_loss

        train_loss_recon /= batch_num
        train_loss_mask /= batch_num

        net.eval()
        with torch.no_grad():
            for data in val_loader:
                batch_trj, batch_obs, batch_sizes, batch_gt, scene_recon_gt, ped_recon_gt = \
                    data["obs_trj"].cuda(), data["obs_data"], data["data_sizes"], data["pred"].cuda(), data[
                        "scene_recon_gt"].cuda(), data["ped_recon_gt"].cuda()

                trj_output, scene_logits, ped_logits = net(batch_trj, batch_obs, batch_sizes, batch_gt)
                trj_output = transform_trajectory(trj_output)

                loss = trj_loss(trj_output, batch_gt) / mask_div
                mask_loss = loss.item()
                for i in range(args.grid_num ** 2):
                    loss += 0.3 * (ce_loss(scene_logits[:, i], scene_recon_gt[:, i]) + ce_loss(ped_logits[:, i], ped_recon_gt[:, i]))

                val_loss_recon += loss.item() - mask_loss
                val_loss_mask += mask_loss

                ADE = displacement_error(trj_output, batch_gt, mode="sum").item() / args.pred_len
                FDE = final_displacement_error(trj_output[:, -1], batch_gt[:, -1], mode="sum").item()
                val_res += [ADE, FDE]

        val_loss_recon /= len(val_loader)
        val_loss_mask /= len(val_loader)
        val_res /= len(val_loader.dataset)

        if True:
            print("==> Epoch: %d Train Loss M: %.4f Train Loss R: %.4f Train Loss: %.4f; Val Loss M: %.4f Val Loss R: %.4f Val Loss: %.4f; Val ADE: %.4f Val FDE: %.4f; Time: %.4f"
                  % (epoch, train_loss_mask, train_loss_recon, train_loss_recon + train_loss_mask, val_loss_mask, val_loss_recon,
                     val_loss_recon + val_loss_mask, val_res[0], val_res[1], time.time() - t1))

        torch.save(net.state_dict(), os.path.join(save_dir_base, "ep_%d.pth" % epoch))
