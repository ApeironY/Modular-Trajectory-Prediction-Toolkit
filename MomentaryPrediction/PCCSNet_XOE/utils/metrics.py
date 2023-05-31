import torch


def displacement_error(pred, pred_gt, mode='raw'):
    assert pred_gt.shape[1:] == pred.shape[1:]
    loss = pred_gt - pred
    loss = loss ** 2
    loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(pred, pred_gt, mode='raw'):
    assert pred_gt.shape[1:] == pred.shape[1:]
    loss = pred_gt - pred
    loss = loss ** 2
    loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


def identity_loss(a, b):
    assert len(a.shape) == len(b.shape)
    return torch.sum(torch.abs(a - b)) / len(a)


def exp_l2_loss(a, b):
    assert len(a.shape) == len(b.shape) == 3  # bs * obs_len * 2
    loss_fn = torch.nn.MSELoss()
    gamma = a.shape[1]
    loss = 0

    for i in range(12):
        factor = torch.tensor(((i + 1) / gamma))
        loss += loss_fn(a[:, i, :], b[:, i, :]) * torch.exp(factor)

    return loss
