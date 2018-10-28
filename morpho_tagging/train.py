import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
import numpy as np
import os
import time
import codecs
import pickle
import sys
sys.path.append("../")
import morpho_tagging.networks as networks
import morpho_tagging.data_iterator as data_iterator


np.random.seed(2345)

#  SETTINGS
parser = argparse.ArgumentParser(description='Morpho tagging Pytorch version.')

# which type of network
parser.add_argument("--char_type", type=str, default="conv", help="Character 'bilstm', 'conv' or 'sum'")

# input
parser.add_argument("--char_embedding_size", type=int, default=50, help="Character embedding size")
parser.add_argument("--char_gram", type=int, default=1, help="Character gram")
# bilstm char
parser.add_argument("--char_rec_num_units", type=int, default=100, help="Word or char")
# conv char
parser.add_argument("--char_filter_sizes", type=int, nargs='+', default=[1,2,3,4,5,6], help="Width of each filter")
parser.add_argument("--char_number_of_filters", type=int, nargs='+', default=[25,50,75,100,125,150],
                    help="Total number of filters")
parser.add_argument("--char_conv_act", type=str, default="relu", help="Default is relu, tanh is the other option")

# training
parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to run")
parser.add_argument("--dropout_frac", type=float, default=0., help="Optional dropout")


# dataset
parser.add_argument("--language", type=str, default="fi", help="Finnish (fi), Swedish (sv) or Spanish (es)")
parser.add_argument("--unique_words", type=int, default=1, help="Use unique words rather than all words")
parser.add_argument("--data_path_ud", type=str, required=True,
                    help="Where can I find the datafiles of UD1.4: *-ud-train.conllu, "
                         "*-ud-dev.conllu and *-ud-test.conllu")
parser.add_argument("--save_dir", type=str, required=True,help="Directory to save models")
parser.add_argument("--save_file", type=str, default="tagger_")

paras = parser.parse_args()


# load data
train_x, train_lengths, train_y, valid_x, valid_lengths, valid_y, test_x, test_lengths, test_y, char_vocab, tag_dict \
    = data_iterator.load_morphdata_ud(paras)

paras.save_file += paras.language + "_"

paras.char_vocab_size = len(char_vocab.vocab)
paras.tagset_size = dict([(t.index,len(t.values)) for t in tag_dict.values()])
paras.pad_index = char_vocab.pad_index

# iterators for data
train_it = data_iterator.DataIterator(train_x, train_lengths, train_y, paras.batch_size, train=True)
valid_it = data_iterator.DataIterator(valid_x, valid_lengths, valid_y, paras.batch_size)
test_it = data_iterator.DataIterator(test_x, test_lengths, test_y, paras.batch_size)

# make model
model = networks.Tagger(paras)
model.apply(networks.init_ortho)
model.cuda()

# loss function
loss_functions = {}
for tag_name, tag_element in tag_dict.items():
        loss_functions[tag_element.index] = nn.CrossEntropyLoss()


# optimizer
parameters = model.parameters()
optimizer = optim.Adam(parameters, lr=paras.lr)

# print total number of parameters
parameters = model.parameters()
sum_params = sum([np.prod(p.size()) for p in parameters])
print("Number of parameters: %s " % (sum_params))



print("Store settings")
start_time_str = time.strftime("%d_%b_%Y_%H_%M_%S")
save_file_model = paras.save_file + "data_" + start_time_str
save_file_settings = paras.save_file + "settings_" + start_time_str
save_file_vocab = paras.save_file + "vocab_" + start_time_str
file = codecs.open(os.path.join(paras.save_dir, save_file_settings), "w")
file.write(str(vars(paras)) + "\n")
file.close()

file = codecs.open(os.path.join(paras.save_dir, save_file_vocab), "wb")
pickle.dump([char_vocab],file)
file.close()


best_valid = 0

print(paras)
print("Started training")
for epoch in range(paras.num_epochs):

    ##################
    # training       #
    ##################
    total_loss = 0

    model.train()
    for sentences, tags, lengths in train_it:
        # set gradients zero
        model.zero_grad()
        # run model
        tag_scores = model(sentences, lengths)
        # calculate loss and backprop
        loss = []
        for tagtype_index in range(tags.shape[1]):
            gt = Variable(torch.LongTensor(tags[:,tagtype_index]).cuda())
            loss.append(loss_functions[tagtype_index](tag_scores[tagtype_index], gt))

        total_loss+=sum([l.data.cpu().numpy() for l in loss])

        sum(loss).backward()
        optimizer.step()



    ##################
    # validation     #
    ##################
    model.eval()
    total_valid = 0
    correct_valid = [0 for _ in range(len(paras.tagset_size))]
    for sentences, tags, lengths in valid_it:
        # set gradients zero
        model.zero_grad()
        # run model
        tag_scores = model(sentences, lengths)
        # calculate loss and backprop
        for tagtype_index in range(tags.shape[1]):
            gt = tags[:, tagtype_index]
            predictions = torch.max(tag_scores[tagtype_index],dim=1)[1].cpu().data.numpy()

            correct_valid[tagtype_index]+=sum(np.equal(gt,predictions))
        total_valid+=tags.shape[0]


    print("Epoch %s: train loss %s" % (epoch + 1, total_loss / train_it.n_batches))

    result=""
    for i in range(len(correct_valid)):
        result+=str(i)+": "+str("%.4f" % (correct_valid[i]/total_valid))+"\t"
    print(result)


    if sum(correct_valid) > best_valid:
        best_valid = sum(correct_valid)
        torch.save(model.state_dict(), os.path.join(paras.save_dir, save_file_model + "_best"))
        print("New best")

print("Best accuracy is: %s" % (best_valid))
torch.save(model.state_dict(), os.path.join(paras.save_dir, save_file_model + "_last"))
