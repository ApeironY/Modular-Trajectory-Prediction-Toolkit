import os
import torch
import torch.nn as nn


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer=3):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, bidirectional=True)

    def forward(self, input_data, hidden):
        output, hidden = self.lstm(input_data, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return (torch.zeros(self.num_layer * 2, batch_size, self.hidden_size).cuda(),
                torch.zeros(self.num_layer * 2, batch_size, self.hidden_size).cuda())


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, num_layer=1):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layer)

    def forward(self, input_data, hidden):
        output, hidden = self.lstm(input_data, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return (torch.zeros(self.num_layer, batch_size, self.hidden_size).cuda(),
                torch.zeros(self.num_layer, batch_size, self.hidden_size).cuda())


class Classifier(nn.Module):
    def __init__(self, in_features, n_clusters, mid_size=128):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, mid_size),
            nn.Tanh(),
            nn.Linear(mid_size, mid_size),
            nn.Tanh(),
        )
        self.top_layer = nn.Linear(mid_size, n_clusters)

    def forward(self, data):
        data = self.classifier(data)
        data = self.top_layer(data)
        return torch.softmax(data, dim=1)


class Decoder(nn.Module):
    def __init__(self, pred_len, obs_hidden_size, pred_hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.pred_len = pred_len
        self.obs_hidden_size = obs_hidden_size
        self.pred_hidden_size = pred_hidden_size
        self.num_layer = num_layers

        self.decoder = DecoderLSTM(2 * (self.obs_hidden_size + self.pred_hidden_size), self.num_layer)
        self.fc = nn.Linear(2 * (self.obs_hidden_size + self.pred_hidden_size), 2)

    def forward(self, data):
        bs = len(data[0])
        output = []
        decoder_hidden = self.decoder.initHidden(bs)
        outs = torch.cat(data, 1).unsqueeze(0)
        for i in range(self.pred_len):
            outs, decoder_hidden = self.decoder(outs, decoder_hidden)
            output.append(outs)
        output = torch.cat(output, 0)
        output = self.fc(output)
        output = output.transpose(0, 1)

        return output


class Synthesizer(nn.Module):
    def __init__(self, obs_hidden_size, pred_hidden_size):
        super(Synthesizer, self).__init__()
        self.obs_hidden_size = obs_hidden_size
        self.pred_hidden_size = pred_hidden_size

        self.fc1 = nn.Linear(self.obs_hidden_size, self.obs_hidden_size)
        self.fc2 = nn.Linear(self.obs_hidden_size + self.pred_hidden_size, self.pred_hidden_size)

    def forward(self, obs_encoded, labels, all_modalities):
        assert labels is not None and all_modalities is not None
        p1 = all_modalities[labels][:, :self.obs_hidden_size] - obs_encoded
        p1 = torch.sigmoid(self.fc1(p1))
        p2 = all_modalities[labels][:, self.obs_hidden_size:]
        concatenated_data = torch.cat((p1, p2), dim=1)
        return self.fc2(concatenated_data)


class PCCSNet(nn.Module):
    def __init__(self, obs_len=8, pre_len=12, n_cluster=200, obs_hidden_size=48, pred_hidden_size=48, num_layer=3):

        super(PCCSNet, self).__init__()

        self.obs_len = obs_len
        self.pre_len = pre_len
        self.obs_hidden_size = obs_hidden_size
        self.pred_hidden_size = pred_hidden_size
        self.num_layer = num_layer

        self.encoder_obs = EncoderLSTM(2, self.obs_hidden_size // 2, self.num_layer)
        self.encoder_pre = EncoderLSTM(2, self.pred_hidden_size // 2, self.num_layer)
        self.synthesizer = Synthesizer(self.obs_hidden_size, self.pred_hidden_size)
        self.classifier = Classifier(self.obs_hidden_size, n_cluster)
        self.decoder = Decoder(self.pre_len, self.obs_hidden_size // 2, self.pred_hidden_size // 2, self.num_layer)

    def forward(self, obs_encoded, k=None, batch_labels=None, all_modalities=None, obs_feature_len=None):
        bs = len(obs_encoded)
        assert k is not None

        class_probs = self.classifier(obs_encoded)
        prob_list = torch.argsort(class_probs, descending=True)[0]
        if bs != 1:
            raise NotImplementedError
        assert k <= len(prob_list)

        expanded_obs_encoding = obs_encoded.repeat(k, 1)
        predicted_future_encoding = self.synthesizer(expanded_obs_encoding, labels=prob_list[:k],
                                                     all_modalities=all_modalities)
        output = self.decoder([expanded_obs_encoding, predicted_future_encoding])

        return output

    def gen_encoding(self, trj_obs, trj_pred, for_training=True):
        bs, trj_len, _ = trj_obs.shape

        with torch.no_grad():
            trj_obs = trj_obs.transpose(0, 1)
            encoder_obs_hidden = self.encoder_obs.initHidden(bs)
            obs_encoded, _ = self.encoder_obs(trj_obs, encoder_obs_hidden)
            obs_encoding = obs_encoded[-1]

            if for_training:
                trj_pred = trj_pred.transpose(0, 1)
                encoder_pre_hidden = self.encoder_pre.initHidden(bs)
                pre_encoded, _ = self.encoder_pre(trj_pred, encoder_pre_hidden)
                pre_encoding = pre_encoded[-1]
            else:
                pre_encoding = None

        return obs_encoding, pre_encoding

    def load_encoders(self, save_dir):
        checkpoint = torch.load(os.path.join(save_dir, "Obs_Encoder.pth"))
        self.encoder_obs.load_state_dict(checkpoint)
        checkpoint = torch.load(os.path.join(save_dir, "Pred_Encoder.pth"))
        self.encoder_pre.load_state_dict(checkpoint)

    def load_decoder(self, save_dir):
        checkpoint = torch.load(os.path.join(save_dir, "decoder.pth"))
        self.decoder.decoder.load_state_dict(checkpoint)
        checkpoint = torch.load(os.path.join(save_dir, "fc.pth"))
        self.decoder.fc.load_state_dict(checkpoint)

    def load_classifier(self, save_dir):
        checkpoint = torch.load(os.path.join(save_dir, "classifier.pth"))
        self.classifier.load_state_dict(checkpoint)

    def load_synthesizer(self, save_dir):
        checkpoint = torch.load(os.path.join(save_dir, "synthesizer_fc1.pth"))
        self.synthesizer.fc1.load_state_dict(checkpoint)
        checkpoint = torch.load(os.path.join(save_dir, "synthesizer_fc2.pth"))
        self.synthesizer.fc2.load_state_dict(checkpoint)

    def load_models(self, save_dir):
        self.load_encoders(save_dir)
        self.load_decoder(save_dir)
        self.load_classifier(save_dir)
        self.load_synthesizer(save_dir)
