import os


def check_memory_consistency(save_dir, args):
    if args.no_time_check:
        return True
    obs_encoder_time = os.path.getctime(os.path.join(save_dir, "Obs_Encoder.pth"))
    pre_encoder_time = os.path.getctime(os.path.join(save_dir, "Pred_Encoder.pth"))
    memory_time = os.path.getctime(os.path.join(save_dir, "memory.npy"))
    if not (memory_time > obs_encoder_time and memory_time > pre_encoder_time):
        return False
    return True


def check_cluster_consistency(save_dir, size_array, args):
    if args.no_time_check:
        return True
    memory_time = os.path.getctime(os.path.join(save_dir, "memory.npy"))
    cluster_time = os.path.getctime(os.path.join(save_dir, "cluster_result.npy"))
    if len(size_array) != args.n_cluster or cluster_time < memory_time:
        return False
    return True


def check_prob_consistency(save_dir, args):
    if args.no_time_check:
        return True
    cluster_time = os.path.getctime(os.path.join(save_dir, "cluster_result.npy"))
    gt_time = os.path.getctime(os.path.join(save_dir, "probabilities.pkl"))
    if cluster_time > gt_time:
        return False
    return True


def check_classifier_consistency(save_dir, args):
    if args.no_time_check:
        return
    classifier_time = os.path.getctime(os.path.join(save_dir, "classifier.pth"))
    memory_time = os.path.getctime(os.path.join(save_dir, "memory.npy"))
    cluster_time = os.path.getctime(os.path.join(save_dir, "cluster_result.npy"))
    if not (classifier_time > memory_time and classifier_time > cluster_time):
        print("==> WARNING: Classifier May Be Out of Date")
