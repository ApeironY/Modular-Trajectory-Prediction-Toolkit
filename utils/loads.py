import torch
import os


def load_encoders(net, save_dir):
    checkpoint = torch.load(os.path.join(save_dir, "Obs_Encoder.pth"))
    net.encoder_obs.load_state_dict(checkpoint)
    checkpoint = torch.load(os.path.join(save_dir, "Pred_Encoder.pth"))
    net.encoder_pre.load_state_dict(checkpoint)


def load_decoder(net, save_dir):
    checkpoint = torch.load(os.path.join(save_dir, "decoder.pth"))
    net.decoder.load_state_dict(checkpoint)
    checkpoint = torch.load(os.path.join(save_dir, "fc.pth"))
    net.fc.load_state_dict(checkpoint)
    return net


def load_classifier(net, save_dir):
    checkpoint = torch.load(os.path.join(save_dir, "classifier.pth"))
    net.classifier.load_state_dict(checkpoint)


def load_synthesizer(net, save_dir):
    checkpoint = torch.load(os.path.join(save_dir, "synthesizer_fc1.pth"))
    net.synthesizer_fc1.load_state_dict(checkpoint)
    checkpoint = torch.load(os.path.join(save_dir, "synthesizer_fc2.pth"))
    net.synthesizer_fc2.load_state_dict(checkpoint)
    return net


