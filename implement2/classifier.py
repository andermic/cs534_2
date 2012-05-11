#!/usr/bin/env python

"""
Written by Michael Anderson & Sean Moore
For CS331 - Introduction to Artificial Intelligence
Spring 2010

Implements a Bayesian classifier that reads in pre-formatted training
 and testing data, and uses the former to classify the latter.
"""

import os
import sys
from math import log

# If the user does not provide the correct number of arguments, output a
#  usage message.
if len(sys.argv) != 5:
    print 'USAGE: classifier.py <isBaseline> <stoplist file>',
    print '<training data file> <testing data file>'
    exit()

isBaseline = sys.argv[2]
training_file_name = sys.argv[3]
testing_file_name = sys.argv[4]

########################################################################
#                   Step 2 - Construct Bayesian Network                #
########################################################################

# Get training data from file. First get the vocab list
training_file = open(training_file_name, 'r')
vocab_list = training_file.readline()
vocab_list = vocab_list.split(',')
vocab_list = vocab_list[:-1]

# Then count the occurrences of each word within a given category. Also
#  count the total number of records in each category.
counts = {}
training_record_counts = {}
loop_num = 0
while True:
    line = training_file.readline()
    if line == '':  # Reached end of file, so break
        break

    # Make the feature vector string a list, grab its class label, and
    #  give it its own dictionary if it does not already exist.
    line = line.split(',')
    class_label = line[-1].strip()
    if class_label not in counts:
        counts[class_label] = {}
        training_record_counts[class_label] = 0

    # Keep track of the total number of records found for each category
    training_record_counts[class_label] += 1

    # Iterate through the current feature list
    for word_index in range(len(vocab_list)):
        if line[word_index] == '1':
            # Increment the word count, or create it if it does not yet
            #  exist
            if vocab_list[word_index] in counts[class_label]:
                counts[class_label][vocab_list[word_index]] += 1
            else:
                counts[class_label][vocab_list[word_index]] = 1

    loop_num += 1
    print 'Read in training vector: %d' % loop_num

training_file.close()

total_records = 0
categories = training_record_counts.keys()
for category in categories:
    total_records += training_record_counts[category]

probs = {}
cprobs = {}
# Calculate the expected probability that a random record will belong to
#  a category. Calculate the expected probability that a word will occur
#  in a record of a given category. 
for category in categories:
    probs[category] = {}
    cprobs[category] = float(training_record_counts[category]) / \
     float(total_records)

    for word in vocab_list:
        count = 0
        if word in counts[category]:
            count = counts[category][word]

        # Use Dirichlet Priors trick in calculation.
        probs[category][word] = \
         (float(count + 1) / float(training_record_counts[category] + 2))


########################################################################
#                       Step 3 - Test Network                          #
########################################################################

# Get testing data from file. First get the vocab list
testing_file = open(testing_file_name, 'r')
vocab_list = testing_file.readline()
vocab_list = vocab_list.split(',')
vocab_list = vocab_list[:-1]

correct_counts = {}
testing_record_counts = {}
record_count = 0
for category in categories:
    correct_counts[category] = 0
    testing_record_counts[category] = 0
# For each file, predict its category using the training data. Keep
#  track of the total number of correct guesses for each category.
while True:
    line = testing_file.readline()
    if line == '':  # Reached end of file, so break
        break

    line = line.split(',')
    best_category = None
    best_prob = None

    # Calculate probability sums for each category, keep track of best
    #  category so far found
    for category in categories:
        cur_prob = 0    # Initialize probability sum to 0
        
        # Add log(P(Y = v))
        cur_prob += log(cprobs[category])
        
        # Add all log(P(Xj = uj | Y = v))
        for word_index in range(len(vocab_list)):
            if vocab_list[word_index] in probs[category]:
                if line[word_index] == '1':
                    cur_prob += log(probs[category][vocab_list[word_index]])
                else:
                    cur_prob += log(1-probs[category][vocab_list[word_index]])

        # If the current category improves upon the best category so far
        #  found, update best variables
        if not best_category or cur_prob > best_prob:
            best_category = category
            best_prob = cur_prob

    # Grab the actual category of the record from the last entry in the
    #  vector.
    actual_category = line[-1].strip()
    
    # If the guess was correct, increment number of correct guesses for
    #  this record's category. Also increment the number of total
    #  records processed for this category regardless.
    if best_category == actual_category:
        correct_counts[actual_category] += 1
    testing_record_counts[actual_category] += 1

    # Output the number of records so far classified so user can see
    #  that the program is actually doing something during the wait
    record_count += 1
    print 'Classified testing record ' + str(record_count)

print
print 'RESULTS:'
# Calculate the percentage of correct guesses for each category. Use
#  100 * (correct predictions in category) / (total records in category)
for category in categories:
    print '%s - %.2f%%' % (category,
     100 * float(correct_counts[category]) / \
     float(testing_record_counts[category]))
