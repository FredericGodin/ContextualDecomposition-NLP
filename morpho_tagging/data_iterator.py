import codecs
import numpy as np
import math
import os

batching_seed = np.random.RandomState(1234)


class CharacterGramVocab:
    def __init__(self,gram=1):

        self.unk_string = "<unk>"
        self.unk_index = 1
        self.pad_string = "<pad>"
        self.pad_index = 0
        self.special_token_string = "<special_token>"
        self.special_token_index = 2
        self.char_to_index = {self.pad_string: self.pad_index,self.unk_string: self.unk_index,self.special_token_string:self.special_token_index}
        self.vocab = [self.pad_string,self.unk_string,self.special_token_string]
        self.index_to_count = {}

        self.eow_string = "<eow>"
        self.sow_string = "<sow>"


        self.gram = gram


    def add_string(self, string_value, add_eow=False, add_sow=False):
        result = []
        string_chars = list(string_value)
        if add_sow:
            string_chars.insert(0, self.sow_string)
        if add_eow:
            string_chars.append(self.eow_string)

        for i in range(len(string_chars)-self.gram+1):

            c = "".join(string_chars[i:i+self.gram])

            if c not in self.vocab:
                index = len(self.vocab)
                self.char_to_index[c] = index
                self.index_to_count[index] = 1
                self.vocab.append(c)
                result.append(index)
            else:
                char_index = self.char_to_index[c]
                result.append(char_index)
                self.index_to_count[char_index] += 1

        return result

    def string_to_index(self, string_value,add_eow=False,add_sow=False):
        result = []

        string_chars = list(string_value)
        if add_sow:
            string_chars.insert(0, self.sow_string)
        if add_eow:
            string_chars.append(self.eow_string)


        for i in range(len(string_chars)-self.gram+1):

            c = "".join(string_chars[i:i+self.gram])
            if c not in self.vocab:
                result.append(self.char_to_index[self.unk_string])
            else:
                result.append(self.char_to_index[c])

        return result

    def index_to_char(self, index):

        if isinstance(index, int):
            if index < len(self.vocab):
                return self.vocab[index]
            else:
                return None
        elif isinstance(index, list):
            result = []
            for i in index:
                if i >= len(self.vocab):
                    result.append(None)
                else:
                    result.append(self.vocab[i])
            return result
        else:
            return None

    def get_special_token_index(self):
        return self.char_to_index[self.special_token_string]

class Tag:
    def __init__(self, name, index, values):
        self.name = name
        self.index = index
        self.values = values
        self.counts = np.zeros((len(self.values),))

    def get_tag_index(self, value):
        return self.values.index(value)

    def add(self, name):
        self.counts[self.get_tag_index(name)] += 1


def read_tags(path, lower=False):
    tag_dict = {}
    f = codecs.open(path, 'r', 'utf-8')
    for index, tags in enumerate(f.readlines()):

        parts = tags.strip().split("\t")
        tagname = parts[0]
        tag_values = parts[1:]

        if lower:
            tagname = tagname.lower()
            tag_values = [t.lower() for t in tag_values]

        tag_dict[tagname] = Tag(tagname, index, tag_values)
    f.close()

    return tag_dict


def load_morphdata_ud(paras, tag_path="../data/", char_vocab=None):
    all_labels = read_tags(os.path.join(tag_path, paras.language + "_tags_ud_filtered.txt"), lower=True)

    train_name = os.path.join(paras.data_path_ud, paras.language + "-ud-train.conllu")
    dev_name = os.path.join(paras.data_path_ud, paras.language + "-ud-dev.conllu")
    test_name = os.path.join(paras.data_path_ud, paras.language + "-ud-test.conllu")

    if char_vocab is None:
        char_vocab = CharacterGramVocab(gram=paras.char_gram)
        fixed_vocab = False
    else:
        fixed_vocab = True
    word_to_char = {}
    unique_pairs = {}

    max_length = 100
    max_length_counter = [0]

    def parse_corpus(filename, name):

        x_data = []
        l_data = []
        y_data = []

        with codecs.open(filename) as f:
            for line in f.readlines():
                if line.startswith("#"):
                    continue

                parts = line.strip().split("\t")

                if len(parts) > 1:

                    word = parts[1].strip()
                    field_line = parts[5].strip()

                    if paras.unique_words:
                        if word in unique_pairs and unique_pairs[word] == field_line:
                            # naive field match is order of fields garantueed?
                            continue
                        else:
                            unique_pairs[word] = field_line

                    fields = field_line.split("|")

                    unique_pairs[parts[1].strip()] = parts[5].strip()

                    x = np.zeros((max_length,), dtype=np.int32)
                    y = np.zeros((len(all_labels),), dtype=np.int32)

                    if word not in word_to_char:
                        if name == "train" and not fixed_vocab:
                            res = char_vocab.add_string(word, add_eow=True, add_sow=True)
                        else:
                            res = char_vocab.string_to_index(word, add_eow=True, add_sow=True)
                        word_to_char[word] = res

                        if len(res) > max_length_counter[0]:
                            max_length_counter[0] = len(res)

                    length = len(word_to_char[word])
                    x[0:length] = np.asarray(word_to_char[word])

                    field_dict = {}
                    for field in fields:
                        if "=" in field:
                            parts = field.split("=")
                            field_dict[parts[0].lower()] = parts[1].lower()

                    for tag_name, tag_element in all_labels.items():

                        if tag_name in field_dict:
                            tag_value_index = tag_element.get_tag_index(field_dict[tag_name])

                            y[tag_element.index] = tag_value_index
                            if name == "train":
                                tag_element.add(field_dict[tag_name])

                        elif name == "train":
                            tag_element.add("_na_")

                    x_data.append(x)
                    l_data.append(length)
                    y_data.append(y)

        x = np.vstack(x_data)
        lengths = np.asarray(l_data)
        y = np.vstack(y_data)

        return x[:, :max_length_counter[0]], lengths, y

    train_x, train_lengths, train_y = parse_corpus(train_name, "train")
    dev_x, dev_lengths, dev_y = parse_corpus(dev_name, "dev")
    test_x, test_lengths, test_y = parse_corpus(test_name, "test")

    return train_x, train_lengths, train_y, dev_x, dev_lengths, dev_y, test_x, test_lengths, test_y, char_vocab, all_labels


class DataIterator:
    def __init__(self, x, lengths, y, batch_size, train=False):

        self.x = x
        self.lengths = lengths
        self.y = y

        self.batch_size = batch_size
        self.number_of_sentences = x.shape[0]
        if train:
            self.n_batches = self.number_of_sentences // self.batch_size
        else:
            self.n_batches = math.ceil(self.number_of_sentences / self.batch_size)
        self.train = train

    def __iter__(self):
        if self.train:
            indexes = batching_seed.permutation(np.arange(self.number_of_sentences))
        else:
            indexes = np.arange(self.number_of_sentences)

        for i in range(self.n_batches):
            lengths = self.lengths[indexes[i * self.batch_size:(i + 1) * self.batch_size]]
            y_batch = self.y[indexes[i * self.batch_size:(i + 1) * self.batch_size]]

            x_batch = self.x[indexes[i * self.batch_size:(i + 1) * self.batch_size],
                      :]

            perm_idx = lengths.argsort(axis=0)[::-1]
            length_sentences_ordered = lengths[perm_idx]

            x_batch_ordered = x_batch[perm_idx]

            y_batch_ordered = y_batch[perm_idx]

            yield x_batch_ordered, y_batch_ordered, \
                  [int(length_sentences_ordered[j]) for j in range(len(length_sentences_ordered))]
