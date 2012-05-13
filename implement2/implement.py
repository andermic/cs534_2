#! /usr/bin/python

from math import log, floor
from sys import argv

# Initialize default values for parameters
h1 = False
h2 = False
test_prefix = 'test'

# Set parameters based on command line arguments given by user
model = argv[1]
print 'Model: %s' % {'b':'Bernoulli', 'm':'Multinomial'}[model]
flags = argv[2:]
for flag in flags:
    if len(flag.split('=')) == 2 and flag.split('=')[0] == '-h1':
        h1 = True
        k = int(flag.split('=')[1])
        print 'Using heuristic 1, k = %d' % k
if '-h2' in flags:
    h2 = True
    print 'Using heuristic 2'
if '-train' in flags:
    test_prefix = 'train'
print 'Classifying %sing data' % test_prefix



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
#  training data less than k times.
if h1:
    counts = [0 for i in range(len(words))]
    for element in tr_data:
	doc_id, word_id, count = element
        counts[word_id] += count

    words = [(i if counts[i] >= k else None) for i in range(len(counts))]

# For this heuristic, throw out a list of commonly occurring words that are
#  likely nuisance parameters.
if h2:
    # Get stoplist
    stoplist_file = open('stoplist.txt','r')
    stoplist = stoplist_file.readlines()
    stoplist = [i.strip() for i in stoplist]
    stoplist_file.close()

    words = [(None if i in stoplist else i) for i in words]



# Process training data.
# Bernoulli model: Count the number of documents each word is present in for
#  each class. Count the number of documents per class.
# Multinomial model: Count the number of times each word appears in documents
#  from each class. Count the total number of words in each class.
if model == 'b':
    word_presences = [[0 for i in range(len(words))] for j in \
     range(len(class_names))]
    docs_per_class = [0 for i in range(len(class_names))]
elif model == 'm':
    word_nums = [[0 for i in range(len(words))] for j in \
     range(len(class_names))]
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
    if words[word_id] != None:
    	last_doc_id = doc_id
        if model == 'b':
            word_presences[label_id][word_id] += 1
            if last_doc_id != doc_id:
                docs_per_class[label_id] += 1
        if model == 'm':
            word_nums[label_id][word_id] += count
            words_per_class[label_id] += count

print 'Done.\n'



# Estimate p(word|class).
probs = [[0 for i in range(len(words))] for j in range(len(class_names))]

print 'Estimating p(word|class) probabilities'
for label_id in range(1, len(class_names)):
    if model == 'b':
        docs = docs_per_class[label_id]
    elif model == 'm':
        total_words = words_per_class[label_id]

    for word_id in range(1, len(words)):
        if words[word_id] != None:
            if model == 'b':
                word_pres = word_presences[label_id][word_id]
            elif model == 'm':
                word_num = word_nums[label_id][word_id]

            # Don't forget Laplace smoothing.        
            if model == 'b':
                probs[label_id][word_id] = float(word_pres + 1) / \
                 float(docs + 2)
            elif model == 'm':
                probs[label_id][word_id] = float(word_num + 1) / \
                 float(total_words + (len(words)-1))
print 'Done.\n'



# Calculate the probabilities that each test document belongs to each class.
test_file = open(test_prefix + '.data','r')
doc_probs = [None]
doc_words = [None]

last_doc_id = 0
while True:
    # Get a line from the test data file. Break if eof, else parse line.
    line = test_file.readline()
    if line == '':
        break
    doc_id, word_id, count = [int(i) for i in line.strip().split(' ')]

    if words[word_id] != None:
        # If the read line is the start of a new document, initialize
        #  probability lists and a word list for the new document.
        if doc_id != last_doc_id:
            last_doc_id = doc_id
            doc_probs.append([0 for i in range(len(class_names))])
            doc_words.append([])
            print 'Classifying document %d' % doc_id

        # Add to a running total for the current doc the probability of the
        #  given word occurring the given number of times, for each class. Use
        #  log probabilities for numerical stability.
        for label_id in range(1, len(class_names)):
            if model == 'b':
                doc_probs[-1][label_id] += log(probs[label_id][word_id])
            elif model == 'm':
                doc_probs[-1][label_id] += log(probs[label_id][word_id]) * \
                 count
            doc_words[-1].append(word_id)
print 'Done.\n'



# Get true test labels
test_label_file = open(test_prefix + '.label','r')
test_labels = test_label_file.readlines()
test_labels = [None] + [int(i.strip()) for i in test_labels]
test_label_file.close()

# Predict labels and generate confusion matrix.
conf = [[0 for i in range(len(class_names))] for j in range(len(class_names))]
for i in range(1,len(doc_probs)):
    true_label = test_labels[i]
    pred_label = sorted(zip(doc_probs[i][1:],range(1,len(class_names))), \
     key = lambda x:-x[0])[0][1]
    conf[true_label][pred_label] += 1

# Write out confusion matrix to file.
conf_file = open('confusion.csv','w')
for row in conf[1:]:
    conf_file.write('%s\n' % ','.join([str(i) for i in row[1:]]))
conf_file.close()

# Print the confusion matrices and accurary rates. Rows of the confusion matrix 
#  are the true labels. The columns are the predicted labels.
print '%s confusion matrix:' % {'b':'Bernoulli','m':'Multinomial'}[model]
for row in conf[1:]:
    print row[1:]
print
print '%s accuracy rates by class:' % {'b':'Bernoulli','m':'Multinomial'}[model]
accuracies = [float(conf[i][i])/sum(conf[i]) for i in range(1,21)]
for accuracy in accuracies:
    print '%.3f' % accuracy,
print '\n'
rate = float(sum([conf[i][i] for i in range(1,21)])) / sum([sum(i) for i \
 in conf])
print 'Total accuracy rate: %.3f\n' % rate

if h1 or h2:
    print 'Number of words left in vocabulary after applying heuristics:',
    print str(len([i for i in words if i != None]))
