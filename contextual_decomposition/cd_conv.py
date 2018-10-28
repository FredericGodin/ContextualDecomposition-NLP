from __future__ import print_function
import numpy as np
import os
import sys

sys.path.append("../")
import morpho_tagging.networks as networks
import codecs
import torch
from torch.autograd import Variable

import morpho_tagging.data_iterator as data_iterator
import pickle


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


np.random.seed(2345)


class CDConv:

    def __init__(self, settings, tag_path="../data/"):

        self.test_paras = Bunch(settings)

        # load model settings
        file = codecs.open(os.path.join(self.test_paras.model_folder, self.test_paras.settings_name), "r")
        data = eval(file.readlines()[0])

        self.paras = Bunch(data)
        self.language = self.paras.language

        filename_model = self.test_paras.settings_name.replace("settings", "data") + "_best"
        filename_vocab = self.test_paras.settings_name.replace("settings", "vocab")

        # load vocab
        file = codecs.open(os.path.join(settings["model_folder"], filename_vocab), "rb")
        self.char_vocab = pickle.load(file)[0]
        file.close()

        # load labels
        self.all_labels = data_iterator.read_tags(os.path.join(tag_path, self.paras.language + "_tags_ud_filtered.txt"),
                                                  lower=True)

        # load model
        self.model = networks.Tagger(self.paras)
        self.model.load_state_dict(torch.load(os.path.join(self.test_paras.model_folder, filename_model)))
        self.model.cuda()
        self.model.eval()

        self.conv_weights_all = [weight[0].weight.cpu().data.numpy() for weight in self.model.char_convs]
        self.conv_biases = [weight[0].bias.cpu().data.numpy() for weight in self.model.char_convs]
        self.hidden_dims = [w.shape[0] for w in self.conv_weights_all]
        self.conv_sizes = [w.shape[2] for w in self.conv_weights_all]

    @staticmethod
    def decomp_three(a, b, c, activation):
        # method provided by Murdoch et al. 2018
        a_contrib = 0.5 * (activation(a + c) - activation(c) + activation(a + b + c) - activation(b + c))
        b_contrib = 0.5 * (activation(b + c) - activation(c) + activation(a + b + c) - activation(a + c))
        return a_contrib, b_contrib, activation(c)

    def get_scores(self, word, tag, gram_lists):

        # make input
        valid_x = np.asarray(self.char_vocab.string_to_index(word, add_eow=True, add_sow=True))[None, :]
        T = valid_x.shape[1]

        # add padding
        valid_x = np.pad(valid_x, ((0, 0), (np.max(self.conv_sizes), np.max(self.conv_sizes))), "constant")
        pad_offset = np.max(self.conv_sizes)  # how much padding before the word
        pad_additional_op = np.max(self.conv_sizes)

        valid_lengths = np.asarray([valid_x.shape[1]])
        valid_y = np.zeros((1, 1), dtype=np.int32)

        tag_index = tag[0]
        tag_value = tag[1]

        valid_it = data_iterator.DataIterator(valid_x, valid_lengths, valid_y, 1)
        for sentences, tags, lengths in valid_it:
            out = self.model.forward(sentences, lengths)

            sanity_value = out[tag_index].cpu().data.numpy()[0, tag_value]

        gram_dict_relevant = {}
        gram_dict_irrelevant = {}

        for sentences, tags, lengths in valid_it:

            sentence_inputs_chars = Variable(torch.LongTensor(sentences).cuda())
            sentence_inputs_chars = sentence_inputs_chars[0]


            word_vecs = self.model.char_embeddings(sentence_inputs_chars).cpu().data.numpy()


            # for each set of chars
            for gram_list in gram_lists:

                rel_all = []
                irrel_all = []

                # calculate convolutions
                for conv_model_i in range(len(self.conv_weights_all)):

                    hidden_dim = self.hidden_dims[conv_model_i]
                    conv_size = self.conv_sizes[conv_model_i]
                    conv_weights = self.conv_weights_all[conv_model_i]
                    conv_bias = self.conv_biases[conv_model_i]

                    relevant = np.zeros((T+pad_additional_op, hidden_dim))
                    irrelevant = np.zeros((T+pad_additional_op, hidden_dim))
                    rel_contrib = np.zeros((T+pad_additional_op, hidden_dim))
                    irrel_contrib = np.zeros((T+pad_additional_op, hidden_dim))

                    # for each char part of a conv input
                    for i in range(T+pad_additional_op):

                        # split relevant and irrelevant part conv for every input
                        for conv_i in range(conv_size):
                            k = i + conv_i
                            if k - pad_offset in gram_list:
                                relevant[i] += np.dot(conv_weights[:, :, conv_i], word_vecs[k])[:, 0]
                            else:
                                irrelevant[i] += np.dot(conv_weights[:, :, conv_i], word_vecs[k])[:, 0]

                    # decompose activation function
                    for i in range(T+pad_additional_op):
                        # do decomposition of input through activation
                        rel_contrib[i], irrel_contrib[i], bias_contrib = self.decomp_three(relevant[i], irrelevant[i],
                                                                                           conv_bias,
                                                                                           lambda x: np.maximum(
                                                                                               np.zeros((hidden_dim,)),
                                                                                               x))
                        # deem bias as neutral so no-positive influence
                        irrel_contrib[i] += bias_contrib

                    # max pooling part
                    max_indices = np.argmax(rel_contrib + irrel_contrib, axis=0)
                    rel_max = rel_contrib[max_indices, np.arange(hidden_dim)]
                    irrel_max = irrel_contrib[max_indices, np.arange(hidden_dim)]

                    rel_all.append(rel_max)
                    irrel_all.append(irrel_max)

                # classification part
                W_out = self.model.hidden2tag[tag_index].weight.data.cpu().numpy()

                scores = np.dot(W_out, np.concatenate(rel_all))
                irrel_scores = np.dot(W_out, np.concatenate(irrel_all))

                correct_tag = tag_value

                gram_dict_relevant[gram_list] = scores[correct_tag]
                gram_dict_irrelevant[gram_list] = irrel_scores[correct_tag]


        # sanity check : sum of linearization should always be the same as original model output
        for gram_list in gram_lists:

            bias_score = self.model.hidden2tag[tag_index].bias.data[tag_value]
            output_value = np.sum([gram_dict_relevant[gram_list],gram_dict_irrelevant[gram_list],bias_score],axis=0)

            sanity_check = np.abs(output_value-sanity_value) < 0.0001
            assert(sanity_check)

        return gram_dict_relevant, gram_dict_irrelevant, out[tag_index].cpu().data.numpy()