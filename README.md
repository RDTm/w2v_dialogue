# About
This code is an example of how to use a word2vec model to replace unknown words in a dialogue system. The goal of this code is certainly not to be a good dialogue management system - the response selection mechanism is made up by myself and not valid in any statistical way. The word2vec model used in my examples is a pre-trained network (trained on 100 billion words from Google news) and can be downloadad from word2vec site (https://code.google.com/archive/p/word2vec/).

# Motivation
This project was done as part of "Intelligent Systems" course in University of Tartu. In the course we used Virtual Human Toolkit's NPCEditor as our default dialogue manager. As this system was not cabable of dealing with unknown words, I propose a way to replace unknown words with known words with similar meaning. 

# Installation
I have run this code on Ubuntu 14.04, but it should work on any machine capable of running Python 2.7, Numpy, Scipy and Gensim packages (https://radimrehurek.com/gensim/). I use Flask just to create a basic website to show off the resulting dialogue system.

# Code
The code is partially inspired by the example on gensim website https://radimrehurek.com/gensim/models/word2vec.html . The main code is in dialogue_manager.py:
1) reads in a list of example query-response pairs,
2) generates more similar examples based on them,
3) builds a matrix representing "which words participared in leading to a certain response",
4) when receiving a new query, search for unknown words,
5) if unknown word has a very similar word among known words, replace it, 
6) if not, if the unknown word is similar to a global category, replace it, 
7) otherwise remove the word, 8)pass the cleaned up sentence to an answer selection based on the matrix built in 3) 


