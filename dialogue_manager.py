import gensim,logging
from gensim.models import Word2Vec
import scipy.spatial
import numpy as np
import string

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class dialogue:
	
	def __init__(self):
		self.model = Word2Vec.load_word2vec_format('models/GoogleNews-vectors-negative300.bin.gz', binary=True)  # C binary format

		raw_utterance_pairs = np.genfromtxt("shop.plist.txt", dtype=None, delimiter="\t")
		raw_client_utterances = raw_utterance_pairs[:,0]
		raw_AI_utterances = raw_utterance_pairs[:,1]

		# we now generate a lot more input pairs by substituting parts of phrases with quasi-similar expressions
		to_add = []
		same_stuff = ["I'm looking for", "I want to buy", "I need", "Where can I find","Where could I get",
             	 	"Where can I buy", "Where could I buy", "Do you have", "Can I get", "I'd like to have", "I'd like to get"]
		functionally_same = [["'d","would"], ["n't", " not"], ["can", "could"] ["wife", "girlfriend"]]

		for pair in raw_utterance_pairs:
		    #replace the first list with all of the synonymous ways of posing the question
		    if any(expr in pair[0] for expr in same_stuff):
        		for beg in same_stuff:
            			new_ones = map(lambda x: pair[0].replace(beg,x), same_stuff)
            			for n in new_ones:
                			#conditions
                			alr = n not in raw_client_utterances and n not in to_add # not in the list already, would mess up the matrix
                			atm = not ("ATM" in n and "buy" in n) # exception, cannot generate phrases
                			eat = not ("eat" in n and "buy" in n) # exception, cannot generate phrases
                			if all([alr, atm,eat]):
 	                   			to_add.append([n,pair[1]])

                #print to_add
		np.savetxt("generated_pairs.txt", np.vstack([raw_utterance_pairs,to_add]), delimiter="\t",fmt="%s")
		# we are done adding sentences now
		
		#we read in all generated pairs
		self.utterance_pairs = np.genfromtxt("generated_pairs.txt", dtype=None, delimiter="\t")
		self.client_utterances = self.utterance_pairs[:,0]
		self.AI_utterances = self.utterance_pairs[:,1]
		self.AI_utterances = list(set(self.AI_utterances)) #set removed duplicates if any

		#and create an ID for each utterance of the AI
		self.utterance_map = dict(zip(self.AI_utterances, range(len(self.AI_utterances))))
		self.reversed_utterance_map = dict((reversed(item) for item in self.utterance_map.items()))
		#print "mapping AI utterance to ID:", self.utterance_map, 
		#print "mapping ID to AI utterance:", self.reversed_utterance_map

		# We first STANDARDIZE the sentences of people and then give each word and ID
		# we will want to remove puctuation from words (so "Hi!" and "Hi" would not be different words)
		self.exclude = set(['.',',','?','!','-']) # exclude these symbols (see below)

		all_utterances = " ".join(self.client_utterances)
		all_words = all_utterances.split()
		for i,w in enumerate(all_words):
		    w = ''.join(ch for ch in w if ch not in self.exclude)  #exlude commas, dots, questionmarks
		    if not ("I"==w or "I'" in w):
			w = w.lower() #causes problems for the word "I", but needs to be done
		    all_words[i] = w

		self.client_vocabulary = list(set(all_words))
		self.dictionary = dict(zip(self.client_vocabulary, range(len(self.client_vocabulary)))) #give ID-s

		print "All the words that the AI knows: \n", self.dictionary
		self.reversed_dictionary = dict((reversed(item) for item in self.dictionary.items()))
		

		### WE WILL NOW PROCEED TO CREATE A BASIS FOR MAPPING QUERIES WITH ANSWERS
		# I need for each AI response the list of utterances that might lead to this answer
		# from there I get the words that lead to the response
		# for each word I will keep track of how many times it has been used in all utterances and how many times it has lead to this response
		# (evidence provided by this word) = count(this word leading to response)/count(this word)
		# I keep that in a matrix of (responses x words)
		# the words found inside the AI responses are completely disregarded for now

		self.occurence_matrix = np.zeros((len(self.AI_utterances),len(self.client_vocabulary)))
		for pair in self.utterance_pairs:
		    query = pair[0]
		    AI = pair[1]
		    for word in query.split():
			word = ''.join(ch for ch in word if ch not in self.exclude)  # exlude commas, dots, questionmarks
			if not ("I"==word or "I'" in word):
			    word = word.lower()#causes problems for the word I
			word_index = self.dictionary[word]
			self.occurence_matrix[self.utterance_map[AI], word_index] += 1

		#we need to normalize the count by total count
		tot_occurences = np.sum(self.occurence_matrix, axis=0)
		self.occurence_matrix = self.occurence_matrix / tot_occurences[np.newaxis,:]

		# HERE WE START DEALING WITH WORD2VEC
		#create mapping from words to their word2vec vectors, Also from ID to vectors
		self.idx_to_vectors = {}
		self.words_to_vectors = {}
		for idx,word in enumerate(self.client_vocabulary):
		    vector = None
		    try:
			vector = self.model[word]
		    except:
			print word, " is too common or unknown and has no vector" # word2vec has no vector for some prepositions
			vector = None
		    self.idx_to_vectors[idx] = vector
		    self.words_to_vectors[word] = vector

		# make sure both mappings are the same
		assert(all(self.idx_to_vectors[10] == self.words_to_vectors[self.reversed_dictionary[10]]))

	#function that finds the closest word among known words
	def find_closest_word(self, word):
	    word_vec = self.model[word] #get the unknown word's vector
	    print np.shape(word_vec)
	    closest = ""
	    min_dist = 100000 #just a big big number
	    for w in self.dictionary:
		if self.words_to_vectors[w] == None:
		    pass
		else:
		    c_d = scipy.spatial.distance.cosine(word_vec, self.words_to_vectors[w])
		    if c_d < min_dist:
		        min_dist = c_d
		        closest = w
	    print min_dist, closest
	    return min_dist, closest


	#function to select responses based on the matrix and input sentence
	def respond(self, sentence):
	    print sentence
	    sentence = ''.join(ch for ch in sentence if ch not in self.exclude)  # exlude commas, dots, questionmarks
	    sentence = sentence.lower()
	    # there are some words that cause problems
	    sentence = sentence.replace("any "," ")# any is a garbage word with no meaning
	    sentence = sentence.replace("Any "," ")# any is a grabage word with no meaning
	    sentence = sentence.replace("i ","I ")
	    sentence = sentence.replace("i'","I'")

	    for utt in self.client_utterances:
		if sentence.lower() == utt.lower(): 
		    print "we have this excact utterance, we COULD use our knowledge, but we do not do this. Because want to test the stability fo the method"
		    
	    best_match = "no_match"
	    #first we should check for unkown words
	    for word in sentence.split():
		if word not in self.client_vocabulary:
		    #print word
		    try: #if the word2vec has no vector for the word we get an error (for ex: a typing error)
		        dist, closest = self.find_closest_word(word)
		        print "closest word is", closest, " at ", dist
		        if dist < 0.5:
		            sentence = sentence.replace(word, closest)
		            print "Replaced sentence is: \n", sentence
		        else: #if there is no similar word. Let's find if the word belongs to some category
			    fur= scipy.spatial.distance.cosine(self.model[word], self.model["furniture"])
			    food = scipy.spatial.distance.cosine(self.model[word], self.model["food"])
			    electronics = scipy.spatial.distance.cosine(self.model[word], self.model["electronics"])
			    office = scipy.spatial.distance.cosine(self.model[word], self.model["office"])
			    sports = scipy.spatial.distance.cosine(self.model[word], self.model["sports"])
			    perfume = scipy.spatial.distance.cosine(self.model[word], self.model["perfume"])
			    household = scipy.spatial.distance.cosine(self.model[word], self.model["household"])
			    clothing = np.min([scipy.spatial.distance.cosine(self.model[word], self.model["clothing"]),scipy.spatial.distance.cosine(self.model[word], self.model["clothes"])])
			    jewelry = scipy.spatial.distance.cosine(self.model[word], self.model["jewelry"])

			    #feel free to comment this printing out
			    print "dist to furniture:", scipy.spatial.distance.cosine(self.model[word], self.model["furniture"])
			    print "dist to food:", scipy.spatial.distance.cosine(self.model[word], self.model["food"])
			    print "dist to sports:", scipy.spatial.distance.cosine(self.model[word], self.model["sports"])
			    print "dist to office:", scipy.spatial.distance.cosine(self.model[word], self.model["office"])
			    print "dist to electronics:", electronics
			    print "dist to clothing:", clothing
			    print "dist to jewelry:", jewelry
			    print "dist to household:", household
			    print "dist to perfume:", perfume

			    #we replace with category only if it is close enough
			    if np.min([fur,food,electronics,office,sports,clothing,jewelry,household,perfume])<0.85:
				    topic = ["furniture","food","electronics","office","sports","clothes","perfume","jewelry","household"][np.argmin([fur,food,electronics,office,sports,clothing,perfume,jewelry,household])]
		        	    sentence = sentence.replace(word,topic)
				    print "Replaced with topic: \n", topic
			    else: #there is no way to replace.. we remove the word and still print out what was the closest
				    sentence = sentence.replace(word,"")
				    print "BAD WORD! Most similar was: \n", self.model.most_similar(positive=[word],topn=10)
			            print "removed a word: \n", sentence
		    except:
		        print word, " is not understood by word2vec, removing it from the senetence"
		        sentence = sentence.replace(word,"")
		        print "replaced ", sentence
		        pass
		else: #the word was in dictionary
		    pass

	    print "Cleaned up sentence is: \n", sentence

	    #now we are ready to choose the best reponse
	    #presume all unkown words are now removed or replaced
	    words = sentence.split()
	    small_matrix = []  #this just takes the columns corresponding to the words present in the sentence
	    for w in words:
		w = ''.join(ch for ch in w if ch not in self.exclude)#exlude commas, dots, questionmarks
		if not ("I"==w or "I'" in w):
		    w = w.lower()#causes problems for the word I
		word_index = self.dictionary[w]
		small_matrix.append(self.occurence_matrix[:,word_index])
	    best = np.argmax(np.sum(small_matrix,axis=0)) #best sencence is given by the row with most evidence
	    return self.reversed_utterance_map[best]
            

if __name__ == '__main__':
    DM = dialogue()
    exit = False
    while not exit:
	    inp = raw_input("Please say something to the virtual shopping assistant (say exit to exit): ")
	    if inp=="exit":
		exit = True
	    else:
		print DM.respond(inp)
