import numpy as np
import torch
import torch.optim as opt
import time
import os
import argparse
from torch.utils.data import DataLoader
from datasets import Social_Stimulus_Dataset, Context_Stimulus_Dataset
from model.models import Social_Stimulus_Verifier, Context_Stimulus_Verifier
from utils import get_lr_scheduler, social_collate_helper

parser = argparse.ArgumentParser()
parser.add_argument("-st", "--stimulus_type", type=str)
parser.add_argument("-d", "--dataset", type=str)
parser.add_argument("-ep", "--num_epochs", type=int, default=100)
parser.add_argument("-ee", "--eval_every", type=int, default=5)
parser.add_argument("-bs", "--batch_size", type=int, default=128)
parser.add_argument("-nmx", "--num_mixtures", type=int, default=12)
parser.add_argument("--feature_size", type=int, default=128)
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument('--model_save_dir', type=str, default='saved_models')
parser.add_argument('--data_folder', type=str, default='ethucy')
parser.add_argument("-pf", '--postfix', type=str, default='')

args = parser.parse_args()

if args.stimulus_type == 'context':
    input_channel = 1
    train_dataset = Context_Stimulus_Dataset(args.dataset, 50, "train", data_folder=args.data_folder)
    val_dataset = Context_Stimulus_Dataset(args.dataset, 50, "val", data_folder=args.data_folder)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    net = Context_Stimulus_Verifier(input_channel=input_channel, feat_dim=args.feature_size,
                                    num_mixtures=args.num_mixtures, scale=args.scale).cuda()

elif args.stimulus_type == 'social':
    obs_len = 8
    pred_len = 12
    train_dataset = Social_Stimulus_Dataset(args.dataset, "train")
    val_dataset = Social_Stimulus_Dataset(args.dataset, "val", rotation_aug=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                              collate_fn=social_collate_helper)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                            collate_fn=social_collate_helper)
    net = Social_Stimulus_Verifier(feat_dim=args.feature_size, obs_len=obs_len, pred_len=pred_len,
                                   num_mixtures=args.num_mixtures, trj_scale=args.scale).cuda()

else:
    raise NotImplementedError('Unknown Stimulus Type.')

print('Train Dataset Size', len(train_dataset))
print('Validation Dataset Size', len(val_dataset))
optimizer = opt.Adam(net.parameters(), lr=0.0001)
scheduler = get_lr_scheduler(optimizer, max_iter=len(train_loader) * args.num_epochs)

model_save_dir = os.path.join(args.model_save_dir, args.stimulus_type, args.dataset + args.postfix)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print('Saving Models to', model_save_dir, 'Scale', args.scale)

for epoch in range(args.num_epochs):
    t1 = time.time()
    total_train_loss = 0
    val_loss = 0
    net.train()
    for batch_data in train_loader:
        log_prob = net(**batch_data)
        optimizer.zero_grad()
        loss = -log_prob.mean(0)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item() * len(log_prob)
    print('==> Epoch: %d  Avg Train Loss: %.4f  Time: %.2f' % (epoch + 1, total_train_loss / len(train_dataset), time.time() - t1))

    if epoch > 0 and epoch % args.eval_every == 0:
        all_log_prob = []
        net.eval()
        with torch.no_grad():
            for batch_data in val_loader:
                log_prob = net(**batch_data)
                all_log_prob.append(log_prob.cpu().numpy())
        all_log_prob = np.concatenate(all_log_prob)
        avg_log_prob_of_real_trj = val_dataset.evaluate_avg_prob(all_log_prob)
        print('==> Eval  Avg Log Prob of GT Trajectories: %.4f' % avg_log_prob_of_real_trj)
        save_file_name = '%s_%d_mix_%d_dim_%.2fx_ep_%d.pth' % (args.stimulus_type, args.num_mixtures,
                                                               args.feature_size, args.scale, epoch)
        torch.save(net.state_dict(), os.path.join(model_save_dir, save_file_name))
        print("==> Saved")
