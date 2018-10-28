import sys
import numpy as np
sys.path.append("../")
from contextual_decomposition.cd_conv import CDConv
from morpho_tagging import data_iterator
from contextual_decomposition.evaluate_segmentation import segment_line

##################################
# This is an example for Spanish #
##################################

settings_conv = dict()
# use a Spanish model
language = "es" ###### fill in
settings_conv["settings_name"] = "tagger_ud_"+language+"_settings_DATE" ###### fill in
settings_conv["model_folder"] = "SAVE_DIR" ###### fill in

# segmented word following the test set format
line = "gratuit/gratuito a/gender=fema" ###### fill in


########################################
# Noting to fill in from this point on #
########################################

CD_model = CDConv(settings_conv)

labels_path ="../data/"+language+"_tags_ud_filtered.txt"
all_labels = data_iterator.read_tags(labels_path,True)

def calculate_gram_indexes(position, gram, markers=True):
    return tuple(range(position + markers,position + markers+gram))

word, morphs = segment_line(line,all_labels)

for index,morph in enumerate(morphs):
    gram_list = [calculate_gram_indexes(morph.position,len(morph.morph))]
    rel_scores, irrel_scores,prediction_scores = CD_model.get_scores(word,(morph.class_type,morph.class_value),gram_list)

    prediction = np.argmax(prediction_scores)

    if prediction == morph.class_value:
        print("Prediction is correct")
    else:
        print("Prediction is incorrect")

    print("The contribution score for this morpheme is: ")
    print(list(rel_scores.values())[0])