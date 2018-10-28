import codecs
import sys
sys.path.append("../")
from morpho_tagging import data_iterator
from collections import defaultdict
import numpy as np
import argparse
import itertools
import operator
from contextual_decomposition.cd_conv import CDConv


class Morph():
    def __init__(self, position, morph, class_type, class_value):
        self.position = position
        self.morph = morph
        self.class_type = class_type
        self.class_value = class_value


def segment_line(line, all_labels):
    segments = line.strip().split(" ")

    all_morphs = []
    all_segments = []

    position = 0
    for segment in segments:
        parts = segment.split("/")

        if len(parts[0]) > 0 and parts[0] != ">":
            morph = parts[0]
            tags = parts[1].split(",")

            for tag in tags:
                if "=" in tag:
                    # print(tag)
                    cl, value = tag.split("=")
                    if value == "singv":
                        value = "sing"
                    elif value == "plurv":
                        value = "plur"
                    elif value == "masca":
                        value = "masc"
                    elif value == "fema":
                        value = "fem"

                    all_morphs.append(Morph(position, morph, all_labels[cl].index, all_labels[cl].get_tag_index(value)))
        all_segments.append(parts[0])
        if parts[0] != ">":
            position += len(parts[0])

    word = "".join(all_segments)
    if word[-1] == ">":
        word = word[:-1]

    return word, all_morphs


def read_gold_segments(testset_path, labels_path):
    all_labels = data_iterator.read_tags(labels_path, True)

    words = defaultdict(list)

    with codecs.open(testset_path) as f:
        for line in f.readlines():

            word, all_morphs = segment_line(line, all_labels)

            # remove duplicates and features without segments
            if word not in words and len(all_morphs) > 0:
                words[word] = all_morphs

    return words


def calculate_gram_indexes(position, gram, markers=True):

    return tuple(range(position + markers,position + markers+gram))

def gram_is_part_of(position,max_poses):
    for max_pose in max_poses:
        counter = 0
        for p in position:
            if p in max_pose[0]:
                counter+=1
        if counter == len(position):
            return True
    return False

def generate_consecutive_gram_dict(gram,word_len,markers=True):
    word_len += 2*int(markers)

    gram_list = []
    indexes = np.arange(word_len)

    for i in range(0,word_len-gram+1):
        if gram > 1:
            gram_list.append(tuple(indexes[i:i+gram]))
        else:
            v = indexes[i]
            gram_list.append((v,))

    return gram_list

def generate_all_gram_dict(gram,word_len,markers=True):
    word_len += 2*int(markers)
    indexes = range(word_len)
    gram_list = list(itertools.combinations(indexes,gram))

    return gram_list

def remove_markers(gram_list,word_len):

    new_gram_list = []

    for k in gram_list:
        if 0 not in k and word_len+1 not in k:
            new_gram_list.append(k)


    return new_gram_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Morpho Contextual Decomposition Evaluation.')

    parser.add_argument("--settings_name", type=str, required=True,
                        help="Fill in the name of the settings file of type 'tagger_LANGUAGENAME_settings_DATE'")
    parser.add_argument("--model_folder", type=str, required=True,
                        help="Location of models as defined during training")
    parser.add_argument("--eval_mode", type=str, default="consecutive",
                        help="Which type of evaluation for comparing with the ground truth annotations: 'consecutive' (default) or 'all'")
    parser.add_argument("--best_n", type=int, default=1,
                        help="If we rank all contributions of all character sets, take the best n character sets")
    paras = parser.parse_args()

    #  SETTINGS
    settings_conv = dict()
    settings_conv["settings_name"] = paras.settings_name
    settings_conv["model_folder"] = paras.model_folder

    CD_model = CDConv(settings_conv)

    words = read_gold_segments("../data/" + CD_model.language + "test.gold",
                               "../data/" + CD_model.language + "_tags_ud_filtered.txt")

    confusion_matrix = np.zeros((2, 2), dtype=np.int32)

    for word, morphs in words.items():

        for morph in morphs:

            # generate all character combinations of interest
            if paras.eval_mode == "consecutive":
                gram_list = generate_consecutive_gram_dict(len(morph.morph), len(word))

            elif paras.eval_mode == "all":
                gram_list = generate_all_gram_dict(len(morph.morph), len(word))

            else:
                raise Exception("--eval_mode unknown")

            # remove start-of-word and end-of-word markers from gram list
            gram_list = remove_markers(gram_list, len(word))

            rel_scores, irrel_scores, prediction_scores = CD_model.get_scores(word,
                                                                              (morph.class_type, morph.class_value),
                                                                              gram_list)

            prediction = np.argmax(prediction_scores)
            # what is the gram set of the ground truth
            position = calculate_gram_indexes(morph.position, len(morph.morph), True)

            if paras.best_n < 1 :
                raise Exception("--best_n should be at least one or higher")

            # select top n
            max_poses = sorted(rel_scores.items(), key=operator.itemgetter(1))[-paras.best_n:]

            attribution = gram_is_part_of(position, max_poses)

            confusion_matrix[int(prediction == morph.class_value), int(attribution)] += 1

    print()

    print("Incorrect prediction/Incorrect Attribution\t\t" + str(confusion_matrix[0, 0]) )
    print("Incorrect prediction/Correct Attribution\t\t" + str(
        confusion_matrix[0, 1]))
    print("Correct prediction/Incorrect Attribution\t\t" + str(confusion_matrix[1, 0]) )
    print("Correct prediction/Correct Attribution\t\t\t" + str(confusion_matrix[1, 1]))
