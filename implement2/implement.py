#! /usr/bin/python

from math import log, floor

HEURISTIC_1 = False
HEURISTIC_2 = False
HEURISTIC_3 = False

# 'train' will classify the training data, 'test' will classify the testing data
TEST_PREFIX = 'test'

# Get words
word_file = open('vocabulary.txt','r')
words = word_file.readlines()
words = [None] + [i.strip() for i in words]
word_file.close()

# Get class names
class_file = open('newsgrouplabels.txt','r')
class_names = class_file.readlines()
class_names = [None] + [i.strip() for i in class_names]
class_file.close()

# Get training labels
tr_label_file = open('train.label','r')
tr_labels = tr_label_file.readlines()
tr_labels = [None] + [int(i.strip()) for i in tr_labels]
tr_label_file.close()

# Get training data
print '\nReading training data'
tr_data_file = open('train.data','r')
tr_data = tr_data_file.readlines()
tr_data = [[int(j) for j in i.strip().split(' ')] for i in tr_data]
tr_data_file.close()
print 'Done.\n'

# For this heuristic, remove words from the vocabulary that occur in the
#  training data less than K times.
if HEURISTIC_1:
    K = 5
    counts = [None] + [0 for i in range(len(words))]
    for element in tr_data:
	doc_id, word_id, count = element
        counts[word_id] += count

    words = [None] + [i for i in range(1,len(counts)) if counts[i] >= K]

# For this heuristic, throw out a list of commonly occurring words that are
#  likely nuisance parameters.
if HEURISTIC_2:
    # Get stoplist
    stoplist_file = open('stoplist.txt','r')
    stoplist = stoplist_file.readlines()
    stoplist = [None] + [i.strip() for i in stoplist]
    stoplist_file.close()

    words = [None] + [i for i in words[1:len(words)] if not i in stoplist]
    
# For this heuristic, keep only the K words with the most mutual information.
K = 1000 
if HEURISTIC_3:
    pass    


# TODO
"""
if len(words) != 61189:
    pass
"""

# Process training data.
# Bernoulli model: Count the number of documents each word is present in for
#  each class. Count the number of documents per class.
# Multinomial model: Count the number of times each word appears in documents
#  from each class. Count the total number of words in each class.
word_presences = [[0 for i in range(len(words))] for j in range(len(class_names))]
docs_per_class = [0 for i in range(len(class_names))]
word_nums = [[0 for i in range(len(words))] for j in range(len(class_names))]
words_per_class = [0 for i in range(len(class_names))]

progress = 0
last_doc_id = -1
print '\nProcessing training data'

for line in tr_data:
    if progress % 100000 == 0:
        print '%d/1467346 training data elements processed' % progress
    progress += 1

    doc_id, word_id, count = line
    label_id = tr_labels[doc_id]

    # Update counts
    word_presences[label_id][word_id] += 1
    if last_doc_id != doc_id:
        docs_per_class[label_id] += 1
	last_doc_id = doc_id
    word_nums[label_id][word_id] += count
    words_per_class[label_id] += count

print 'Done.\n'



# Estimate p(word|class) for both the Bernouilli and Multinomial models.
bern_probs = [[0 for i in range(len(words))] for j in range(len(class_names))]
mult_probs = [[0 for i in range(len(words))] for j in range(len(class_names))]

print 'Estimating p(word|class) probabilities'
for label_id in range(1, len(class_names)):
    docs = docs_per_class[label_id]
    total_words = words_per_class[label_id]

    for word_id in range(1, len(words)):
        word_pres = word_presences[label_id][word_id]
        word_num = word_nums[label_id][word_id]

        # Don't forget Laplace smoothing.        
        bern_probs[label_id][word_id] = float(word_pres + 1) / float(docs + 2)
        mult_probs[label_id][word_id] = float(word_num + 1) / \
         float(total_words + (len(words)-1))
print 'Done.\n'



# Calculate the probabilities that each test document belongs to each class,
#  for each model.
test_file = open(TEST_PREFIX + '.data','r')
doc_bern_probs = [None]
doc_mult_probs = [None]
#doc_words = [None]

while True:
    # Get a line from the test data file. Break if eof, else parse line.
    line = test_file.readline()
    if line == '':
        break
    doc_id, word_id, count = [int(i) for i in line.strip().split(' ')]

    # If the read line is the start of a new document, initialize probability
    #  lists and a word list for the new document.
    if doc_id == len(doc_bern_probs):
        doc_bern_probs.append([0 for i in range(len(class_names))])
        doc_mult_probs.append([0 for i in range(len(class_names))])
        #doc_words.append([])
        print 'Classifying document %d' % doc_id

    # Add to a running total for the current doc the probability of the given
    #  word occurring the given number of times, for each class. Use log
    #  probabilities for numerical stability.
    for label_id in range(1, len(class_names)):
        doc_bern_probs[-1][label_id] += log(bern_probs[label_id][word_id])
        doc_mult_probs[-1][label_id] += log(mult_probs[label_id][word_id]) * count
        #doc_words[-1].append(word_id)
print 'Done.\n'

# Get true test labels
test_label_file = open(TEST_PREFIX + '.label','r')
test_labels = test_label_file.readlines()
test_labels = [None] + [int(i.strip()) for i in test_labels]
test_label_file.close()

# Predict labels and generate confusion matrix for both models.
conf_bern = [[0 for i in range(len(class_names))] for j in range(len(class_names))]
conf_mult = [[0 for i in range(len(class_names))] for j in range(len(class_names))]
for i in range(1,len(doc_bern_probs)):
    true_label = test_labels[i]
    pred_label_bern = sorted(zip(doc_bern_probs[i][1:],range(1,len(class_names))), key = lambda x:-x[0])[0][1]
    pred_label_mult = sorted(zip(doc_mult_probs[i][1:],range(1,len(class_names))), key = lambda x:-x[0])[0][1]
    conf_bern[true_label][pred_label_bern] += 1
    conf_mult[true_label][pred_label_mult] += 1

# Print the confusion matrices and accurary rates for each model. Rows of the
#  confusion matrix are the true labels. The columns are the predicted labels.
print '\nBernoulli confusion matrix:'
for i in conf_bern[1:len(conf_bern)]:
    print i[1:len(conf_bern[0])]
print
print 'Bernoulli accuracy rates by class:'
accuracies = [float(conf_bern[i][i])/sum(conf_bern[i]) for i in range(1,21)]
for accuracy in accuracies:
    print '%.3f' % accuracy,
rate = float(sum([conf_bern[i][i] for i in range(1,21)])) / sum([sum(i) for i \
 in conf_bern])
print '\nTotal accuracy rate: %.3f\n\n' % rate

print 'Multinomial confusion matrix:'
for i in conf_mult[1:len(conf_mult)]:
    print i[1:len(conf_bern[0])]
print
print 'Multinomial accuracy rates by class:'
accuracies = [float(conf_mult[i][i])/sum(conf_mult[i]) for i in range(1,21)]
for accuracy in accuracies:
    print '%.3f' % accuracy,
rate = float(sum([conf_mult[i][i] for i in range(1,21)])) / sum([sum(i) for i \
 in conf_mult])
print '\nTotal accuracy rate: %.3f' % (sum(accuracies) / len(accuracies))
