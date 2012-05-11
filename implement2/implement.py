#! /usr/bin/python

from math import log

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



# Get and process training data.
# Bernoulli model: Count the number of documents each word is present in for
#  each class. Count the number of documents per class.
# Multinomial model: Count the number of times each word appears in documents
#  from each class. Count the total number of words in each class.
train_file = open('train.data','r')

word_presences = [[0] * len(words)] * len(class_names)
docs_per_class = [None] + ([0] * len(class_names))
word_nums = [None] + ([[None] + ([0] * len(words))] * len(class_names))
words_per_class = [None] + ([0] * len(class_names))

progress = 0
print '\nProcessing training data'

while True:
    if progress % 100000 == 0:
        print '%d/1467346 training data elements processed' % progress
    progress += 1

    # Get a line from the training data file. Break if eof, else parse line.
    line = train_file.readline()
    if line == '':
        break
    doc_id, word_id, count = [int(i) for i in line.strip().split(' ')]
    label_id = tr_labels[doc_id]

    # Update counts
    word_presences[label_id][word_id] += 1
    docs_per_class[label_id] += 1
    word_nums[label_id][word_id] += count
    words_per_class[label_id] += count

print 'Done.\n'
train_file.close()



# Estimate p(word|class) for both the Bernouilli and Multinomial models.
bern_probs = [[0] * len(words)] * len(class_names)
mult_probs = [[0] * len(words)] * len(class_names)

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
        doc_bern_probs.append([0] * len(class_names))
        doc_mult_probs.append([0] * len(class_names))
        #doc_words.append([])
        print 'Classifying document %d' % doc_id

    # Add to a running total for the current doc the probability of the given
    #  word occurring the given number of times, for each class. Use log
    #  probabilities for numerical stability.
    for label_id in range(1, len(class_names)):
        doc_bern_probs[-1][label_id] += log(bern_probs[label_id][word_id])
        doc_mult_probs[-1][label_id] += log(bern_probs[label_id][word_id]) * count
        #doc_words[-1].append(word_id)
print 'Done.\n'

# Get true test labels
test_label_file = open(TEST_PREFIX + '.label','r')
test_labels = test_label_file.readlines()
test_labels = [None] + [int(i.strip()) for i in test_labels]
test_label_file.close()

for i in range(5):
    print doc_bern_probs[i]
exit()

# Predict labels and generate confusion matrix for both models.
conf_bern = [[0] * len(class_names) for i in range(len(class_names))]
conf_mult = [[0] * len(class_names) for i in range(len(class_names))]
for i in range(1,len(doc_bern_probs)):
    true_label = test_labels[i]
    pred_label_bern = sorted(zip(doc_bern_probs[i][1:],range(1,len(class_names))), key = lambda x:-x[0])[0][1]
    pred_label_mult = sorted(zip(doc_mult_probs[i][1:],range(1,len(class_names))), key = lambda x:-x[0])[0][1]
    conf_bern[true_label][pred_label_bern] += 1
    conf_mult[true_label][pred_label_mult] += 1

for i in conf_bern:
    print i

print

for i in conf_mult:
    print i
