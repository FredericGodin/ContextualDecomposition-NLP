import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class Tagger(nn.Module):
    def __init__(self, paras):
        super(Tagger, self).__init__()

        self.paras = paras
        pad_index = self.paras.pad_index
        self.char_embeddings = nn.Embedding(self.paras.char_vocab_size, self.paras.char_embedding_size,
                                            padding_idx=pad_index)
        self.char_size = 0
        if paras.char_type == "bilstm":

            self.char_lstm = nn.LSTM(self.paras.char_embedding_size, self.paras.char_rec_num_units,
                                     batch_first=True,
                                     bidirectional=True)

            self.char_size += 2 * self.paras.char_rec_num_units

        elif paras.char_type == "conv":

            self.char_convs = nn.ModuleList()
            for i, filter_size in enumerate(paras.char_filter_sizes):
                conv = nn.Sequential()
                padding = (self.paras.char_filter_sizes[i] - 1, 0)
                conv.add_module("word_conv_%s" % (i), nn.Conv2d(1, self.paras.char_number_of_filters[i], kernel_size=(
                    self.paras.char_filter_sizes[i], self.paras.char_embedding_size),padding=padding))
                if self.paras.char_conv_act == "relu":
                    conv.add_module("word_conv_%s_relu" % (i), nn.ReLU())
                else:
                    conv.add_module("word_conv_%s_tanh" % (i), nn.Tanh())

                self.char_convs.append(conv)

                self.char_size += self.paras.char_number_of_filters[i]

        elif paras.char_type == "sum":
            self.char_size += self.paras.char_embedding_size

        classification_in_size = self.char_size

        self.embed_dropout = nn.Dropout(p=paras.dropout_frac)

        self.hidden2tag = nn.ModuleList()
        for i in range(len(paras.tagset_size)):
            self.hidden2tag.append(nn.Linear(classification_in_size, paras.tagset_size[i]))

    def forward(self, sentences, lengths):

        # character input to word embeddings

        sentence_inputs_chars = Variable(torch.LongTensor(sentences).cuda())

        if self.paras.char_type == "bilstm":

            emb = self.char_embeddings(sentence_inputs_chars)
            packed_input = pack_padded_sequence(emb, lengths, batch_first=True)
            lstm_out, (char_hidden_out, char_cell_out) = self.char_lstm(packed_input)

            char_hidden_out = char_hidden_out.transpose(1, 0).contiguous()
            char_hidden_out = char_hidden_out.view(char_hidden_out.size(0), -1)

            char_emb = char_hidden_out

        elif self.paras.char_type == "conv":

            x = self.char_embeddings(sentence_inputs_chars)
            emb = x.unsqueeze(1)
            conv_outs = []
            for char_conv in self.char_convs:
                x = char_conv(emb)
                x_size = x.size()
                x = torch.max(x.view(x_size[0], x_size[1], -1), dim=2)[0]
                conv_outs.append(x)

            char_emb = torch.cat(conv_outs, dim=1)

        elif self.paras.char_type == "sum":
            x = self.char_embeddings(sentence_inputs_chars)
            char_emb = torch.sum(x, dim=1)

        # word level classification

        x = self.embed_dropout(char_emb)

        tag_space = []
        for classifier in self.hidden2tag:
            tag_space.append(classifier(x))

        return tag_space


def init_ortho(m):
    if type(m) is nn.LSTM:
        for names in m._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(m, name)
                nn.init.constant_(bias.data, 0)
            for name in filter(lambda n: "weight" in n, names):
                weight = getattr(m, name)
                nn.init.orthogonal_(weight.data)


    elif type(m) is nn.Linear:
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m, 'bias'):
            nn.init.constant_(m.bias.data, 0)

    elif type(m) is nn.Embedding and m.weight.requires_grad:
        nn.init.uniform_(m.weight.data, -0.01, 0.01)
        if hasattr(m, "padding_idx"):
            torch.nn.init.constant_(m.weight.data[m.padding_idx], 0)

