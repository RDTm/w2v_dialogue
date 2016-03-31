import gensim,logging
from gensim.models import Word2Vec
import scipy.spatial
import numpy as np
import string

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class dialogue:
	
	def __init__(self):
		self.model = Word2Vec.load_word2vec_format('models/GoogleNews-vectors-negative300.bin.gz', binary=True)  # C binary format
		#print self.model.most_similar(positive=['woman', 'king'], negative=['man'])
		#print self.model.most_similar(positive=['pants'])
		#print self.model.most_similar(positive=['chair'])

		raw_utterance_pairs = np.genfromtxt("shop.plist.txt", dtype=None, delimiter="\t")
		raw_client_utterances = raw_utterance_pairs[:,0]
		raw_AI_utterances = raw_utterance_pairs[:,1]

		to_add = []
		same_stuff = ["I'm looking for a", "I want to buy a", "I need a", "Where can I find a","Where could I get a",
             	 	"Where can I buy a", "Where could I buy a", "Do you have a", "Can I get a", "I'd like to have a", "I'd like to get a"]
		for pair in raw_utterance_pairs:
		    if any(expr in pair[0] for expr in same_stuff):
        		for beg in same_stuff:
            			new_ones = map(lambda x: pair[0].replace(beg,x), same_stuff)
            			for n in new_ones:
                			#conditions
                			alr = n not in raw_client_utterances #not there already
                			atm = not ("ATM" in n and "buy" in n)
                			eat = not ("eat" in n and "buy" in n)
                			if all([alr, atm,eat]):
 	                   			to_add.append([n,pair[1]])

                #print to_add
		np.savetxt("stuff.txt", np.vstack([raw_utterance_pairs,to_add]), delimiter="\t",fmt="%s")
		# we are done adding sentences now
		
		#we create and ID for each utterance of the AI
		self.utterance_pairs = np.genfromtxt("stuff.txt", dtype=None, delimiter="\t")
		self.client_utterances = self.utterance_pairs[:,0]
		self.AI_utterances = self.utterance_pairs[:,1]
		self.AI_utterances = list(set(self.AI_utterances))
		self.utterance_map = dict(zip(self.AI_utterances, range(len(self.AI_utterances))))
		#print self.utterance_map
		self.reversed_utterance_map = dict((reversed(item) for item in self.utterance_map.items()))
		#print self.reversed_utterance_map

		# We first STANDARDIZE the sentences of people and then give each word and ID
		#we will want to remove puctuation from words
		exclude = set(['.',',','?','!','-'])
		self.exclude = exclude

		all_utterances = " ".join(self.client_utterances)
		all_words = all_utterances.split()
		for i,w in enumerate(all_words):
		    w = ''.join(ch for ch in w if ch not in exclude)#exlude commas, dots, questionmarks
		    if not ("I"==w or "I'" in w):
			w = w.lower()#causes problems for the word I
		    all_words[i] = w

		self.client_vocabulary = list(set(all_words))
		self.dictionary = dict(zip(self.client_vocabulary, range(len(self.client_vocabulary))))

		print self.dictionary
		self.reversed_dictionary = dict((reversed(item) for item in self.dictionary.items()))
		

		### WE WILL NOW PROCEED TO CREATE A BASIS FOR MAPPING QUERIES WITH ANSWERS
		# i need for each AI response the list of utterances that might lead to this
		# from there I get the words that lead to the response
		# for each word I will keep track of how many times it has been used in all utterances and how many times it has lead to this response
		# (evidence provided by this word)= count(this word leading to response)/count(this word)
		# for that I have a matrix of (responses x words)
		# the words found inside the responses are completely disregarded for now

		self.occurence_matrix = np.zeros((len(self.AI_utterances),len(self.client_vocabulary)))
		for pair in self.utterance_pairs:
		    #print pair
		    query = pair[0]
		    AI = pair[1]
		    #print query, AI
		    for word in query.split():
			word = ''.join(ch for ch in word if ch not in exclude)#exlude commas, dots, questionmarks
			if not ("I"==word or "I'" in word):
			    word = word.lower()#causes problems for the word I
			word_index = self.dictionary[word]
			#print word, word_index, reversed_dictionary[word_index]
			self.occurence_matrix[self.utterance_map[AI], word_index] += 1

		#we need to normalize the count by total count
		tot_occurences = np.sum(self.occurence_matrix, axis=0)
		#print np.shape(self.occurence_matrix),tot_occurences

		self.occurence_matrix = self.occurence_matrix / tot_occurences[np.newaxis,:]

		self.idx_to_vectors = {}
		self.words_to_vectors = {}
		for idx,word in enumerate(self.client_vocabulary):
		    vector = None
		    try:
			vector = self.model[word]
		    except:
			print word, " is too common or unknown and has no vector"
			vector = None
		    self.idx_to_vectors[idx] = vector
		    self.words_to_vectors[word] = vector

		#make sure both mappings are the same
		assert(all(self.idx_to_vectors[10] == self.words_to_vectors[self.reversed_dictionary[10]]))

	def find_closest_word(self, word):
	    word_vec = self.model[word]
	    print np.shape(word_vec)
	    closest=""
	    min_dist= 10000
	    for w in self.dictionary:
		if self.words_to_vectors[w] == None:
		    #print "word vec none", w
		    pass
		else:
		    c_d = scipy.spatial.distance.cosine(word_vec, self.words_to_vectors[w])
		    if c_d < min_dist:
		        min_dist = c_d
		        closest = w
	    print min_dist, closest
	    #print scipy.spatial.distance.cosine(word,closest)
	    return min_dist, closest


	def respond(self, sentence):
	    print sentence
	    sentence = ''.join(ch for ch in sentence if ch not in self.exclude)#exlude commas, dots, questionmarks
	    sentence = sentence.lower()#causes problems for the word I
	    sentence = sentence.replace("any "," ")#any is a garbage word with no meaning
	    sentence = sentence.replace("Any "," ")# any is a grabage word with no meaning
	    sentence = sentence.replace("i ","I ")
	    sentence = sentence.replace("i'","I'")
	    sentence = sentence.replace("hI","hi")

	    for utt in self.client_utterances:
		if sentence.lower() == utt.lower():
		    print "we have this excact utterance, we COULD use our knowledge: "

	    best_match = "no_match"
	    #first we should check for unkown words
	    for word in sentence.split():
		if word not in self.client_vocabulary:
		    #print word
		    try:
		        dist, closest = self.find_closest_word(word)
		        print "closest is", closest
		        if dist < 0.5:
		            sentence = sentence.replace(word, closest)
		            print "replaced ", sentence
		        else:
		            sentence = sentence.replace(word,"")
		            print "removed ", sentence
		    except:
		        print word, " is not understood by word2vec, removing it from the senetence"
		        sentence = sentence.replace(word,"")
		        print "replaced ", sentence
		        pass
		else:
		    pass
		    #print "the word is in dic", word
	    print "cleaned up sentence:", sentence
	    #presume all unkown words removed or replaced
	    words = sentence.split()
	    #print words
	    small_matrix = []
	    for w in words:
		w = ''.join(ch for ch in w if ch not in self.exclude)#exlude commas, dots, questionmarks
		if not ("I"==w or "I'" in w):
		    w = w.lower()#causes problems for the word I
		word_index = self.dictionary[w]
		small_matrix.append(self.occurence_matrix[:,word_index])
	    print np.shape(small_matrix)
	    #print np.sum(small_matrix,axis=0)
	    best = np.argmax(np.sum(small_matrix,axis=0))
	    return self.reversed_utterance_map[best]        
            

#if __name__ == '__main__':
#    DM = dialogue()
#    print DM.respond("Hello")
