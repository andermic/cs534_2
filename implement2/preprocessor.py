#!/usr/bin/env python

"""
Written by Michael Anderson & Sean Moore
For CS331 - Introduction to Artificial Intelligence
Spring 2010

Formats a group of zipped files for use in a Bayesian classifier
"""

import os
import sys
from zipfile import ZipFile
from math import log

# Takes a word and a list of words that are sorted alphabetically.
#  Performs a recursive binary search and returns True if the word is in
#  the sorted list, False if it is not.
def binary_search(word, list):
    mid = len(list) / 2
    if word == list[mid]:
        return True
    if len(list) == 1:
        return False
    elif word < list[mid]:
        return binary_search(word, list[:mid])
    elif word > list[mid]:
        return binary_search(word, list[mid:])

# If the user does not provide the correct number of arguments, output a
#  usage message.
if len(sys.argv) != 5:
    print 'USAGE: preprocessor.py <zipped directory> <advanced1>',
    print '<advanced2> <advanced3>'
    exit()

zipped_dir = sys.argv[1]

# To throw out features that occur less than k times, give the word
#  "true" (without quotes) as the advanced1 command line argument.
advanced1 = sys.argv[2]

# To add the k most frequently occurring bigrams and trigrams to the
#  vocabulary list, give the word "true" (without quotes) as the
#  advanced2 command line argument.
advanced2 = sys.argv[3]

# To calculate with mutual information and use only the k most relevant
#  features, give the word "true" (without quotes) as the advanced3
#  command line argument.
advanced3 = sys.argv[4]


########################################################################
#                    Step 1 - Gather Training Data                     #
########################################################################

# Decompress the training zip file
destination = zipped_dir[:zipped_dir.rfind('.')]
training_file = ZipFile(zipped_dir)
training_file.extractall(destination)
training_file.close()

# Load the stoplist into a python list
STOPLIST_FILE_NAME = 'stoplist.txt'
stoplist_obj = open(STOPLIST_FILE_NAME, 'r')
stoplist = stoplist_obj.read().split('\n')
stoplist_obj.close()

# Code from here forward assumes that the zip file contained a single
#  directory. This directory should contain subdirectories for each 
#  category. Each subdirectory should contain files associated with that
#  category.
training_root = os.path.join(destination, os.listdir(destination)[0])

categories = os.listdir(training_root)
feature_lists = []
vocab_dict = {}
token = ''

if advanced2 == 'true':
    last_token = ''
    last_last_token = ''
    ngrams = {}

if advanced3 == 'true':
    records_per_cat = {}
    feature_count_per_cat = {}
    for category in categories:
        feature_count_per_cat[category] = {}
    total_records = 0

directory_tree = os.walk(training_root)
# Walk through the files in the training directory with the help of some
#  python voodoo
for dir_name, subdir_names, file_names in directory_tree:
    category = dir_name[dir_name.rfind('/')+1:]
    if advanced3 == 'true':
        records_per_cat[category] = len(file_names)
        total_records += len(file_names)
    for file_name in file_names:
        # Create a new feature list for this file
        feature_lists.append({})
        cur_list = feature_lists[len(feature_lists) - 1]
        
        # Get the contents of the given file. Make all letters
        #  lowercase.
        stream = open( os.path.join(dir_name, file_name), 'r' )
        file = stream.read()
        file = file.lower()

        # Tokenize the file contents
        for char in file:
            # Check if the character is alphanumeric. If so, append it
            #  to the current token.
            if ( ord(char) >= ord('a') and ord(char) <= ord('z') ) or \
               ( ord(char) >= ord('0') and ord(char) <= ord('9') ):
                token += char
            else:
                # Check that the token is valid and not in the stoplist
                if token and not binary_search(token, stoplist):
                    # Add the token to the feature list, and track how
                    #  often it occurs in the whole set in vocab_dict
                    if token not in vocab_dict:
                        vocab_dict[token] = 1
                    else:
                        vocab_dict[token] += 1
                    cur_list[token] = '1'
                    
                    # For the ngrams augmentation, add contiguous pairs
                    #  and triples of valid tokens to the feature list.
                    #  Track how often they occur in the whole set using
                    #  ngrams
                    if advanced2 == 'true':
                        if last_token:
                            bigram = '%s %s' % (last_token, token)
                            if bigram not in ngrams:
                                ngrams[bigram] = 1
                            else:
                                ngrams[bigram] += 1
                            cur_list[bigram] = '1'
                        if last_token and last_last_token:
                            trigram = '%s %s %s' % \
                             (last_last_token, last_token, token)
                            if trigram not in ngrams:
                                ngrams[trigram] = 1
                            else:
                                ngrams[trigram] += 1
                            cur_list[trigram] = '1'
                        
                        last_last_token = last_token
                        last_token = token


                    if advanced3 == 'true':
                        if token not in feature_count_per_cat[category]:
                            feature_count_per_cat[category][token] = 1
                        else:
                            feature_count_per_cat[category][token] += 1

                token = ''  # Get ready for the next token

        # Add the ClassLabel, the parent directory of this file
        cur_list['ClassLabel'] = category

        print 'Processed record: ' + str(len(feature_lists))

# If using advanced Bayes augmentation 1, then eliminate all features
#  that occur less than K_ADVANCED1 times in the set
if advanced1 == 'true':
    K_ADVANCED1 = 5
    for key in vocab_dict.keys():
        if vocab_dict[key] < K_ADVANCED1:
            del(vocab_dict[key])

# If using advanced Bayes augmentation 2, then add the K_ADVANCED2 most
#  frequently occurring ngrams to the vocabulary
if advanced2 == 'true':
    K_ADVANCED2 = 25
    items = ngrams.items()
    
    # Sort ngrams in descending order by frequency
    items.sort(key=lambda item:item[1], reverse=True)
    
    for i in range(K_ADVANCED2):
        vocab_dict[items[i][0]] = items[i][1]

# If using advanced Bayes augmentation 3, then keep only the K_ADVANCED3
#  most relevant features in the vocabulary
if advanced3 == 'true':
    K_ADVANCED3 = 50

    probs = {}
    # Calculate P(X = x), P(Y = y), and P(X = x, Y = x) and use formula
    #  from mutual information slides.
    for category in categories:
        py = float(records_per_cat[category] + 1) / \
         float(total_records + len(categories))

        for word in vocab_dict.keys():
            px = float(vocab_dict[word] + 1) / float(total_records + 2)

            count = 0
            if word in feature_count_per_cat[category]:
                count = feature_count_per_cat[category][word]

            pxy = float(count + 1) / float(total_records + 2 * len(categories))

            if word not in probs:
                probs[word] = pxy * log( pxy / (px * py) )
            else:
                probs[word] += pxy * log( pxy / (px * py) )
    
    # Sort probabilities in descending order by frequency
    items = sorted(probs.items(), key=lambda item:item[1], reverse=True)
    
    new_vocab_dict = {}
    # Make a new dictionary that only contains K_ADVANCED3 of the most
    #  relevant words
    for i in range(K_ADVANCED3):
        word = items[i][0]
        new_vocab_dict[word] = vocab_dict[word]
        #print items[i]

    vocab_dict = new_vocab_dict

output_file_name = zipped_dir[:zipped_dir.find('_')] + '.txt'
output_file = open(output_file_name, 'w')

# Write the sorted vocabulary out to a file
vocab_list = sorted(vocab_dict.keys())
for word in vocab_list:
    output_file.write(word + ',')
output_file.write('ClassLabel\n')

# Write the feature lists out to a file
for i in range(len(feature_lists)):
    for word in vocab_list:
        if word in feature_lists[i]:
            output_file.write(feature_lists[i][word] + ',')
        else:
            output_file.write('0,')
    output_file.write(feature_lists[i]['ClassLabel'] + '\n')
    
    print 'Wrote feature list %d out to file' % (i + 1)
output_file.close()
