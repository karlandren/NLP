import os
import argparse
import time
import string
import numpy as np
from halo import Halo
from sklearn.neighbors import NearestNeighbors


class RandomIndexing(object):
    def __init__(self, filenames, dimension=2000, non_zero=100, non_zero_values=[-1, 1], left_window_size=3, right_window_size=3):
        self.__sources = filenames
        self.__vocab = set()
        self.__dim = dimension
        self.__non_zero = non_zero
        self.__non_zero_values = non_zero_values
        self.__lws = left_window_size
        self.__rws = right_window_size
        self.__cv = None
        self.__rv = None
        self.__nbrs = None
        

    def clean_line(self, line):
        # YOUR CODE HERE

        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        # Remove characters in line that is not whitelisted and join together
        answer = ''.join(filter(whitelist.__contains__, line))
        answer = ' '.join(answer.split())
        # Return clean a line
        return [answer]


    def text_gen(self):
        # Cleans and returns all lines 
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    yield self.clean_line(line)


    def build_vocabulary(self):
        """
        Build vocabulary of words from the provided text files
        """
        # YOUR CODE HERE

        # Iterate through cleaned lines and add word to vocab if it doesn't exist
        for i in self.text_gen():
            for k in i[0].split():
                if k not in self.__vocab:
                    self.__vocab.add(k)

        self.write_vocabulary()


    @property
    def vocabulary_size(self):
        return len(self.__vocab)


    def create_word_vectors(self):
        """
        Create word embeddings using Random Indexing
        """
        # YOUR CODE HERE

        self.__rv = dict()
        self.__cv = dict()

        # For each word in the vocabulary, init a zero context vector
        for i in self.__vocab:
            self.__cv[i] = np.zeros(self.__dim)

        # For each word in the vocabulary, init a random vector with -1 and 1's
        for i in self.__vocab:
            self.__rv[i] = np.where(np.random.rand(self.__dim) > 0.5, 1 , -1)

            # set self.__dim - self.__non_zero to 0
            n_zeros = self.__dim - self.__non_zero
            zero_idxs = np.random.choice(self.__dim, n_zeros, replace=False)
            self.__rv[i][zero_idxs] = 0

        # Generate cleaned lines
        for i in self.text_gen():
            words = i[0].split()

            # Iterate through each word
            for k in range(len(words)):
                # a is the window index
                for a in range(1,self.__lws+1):
                    # Only add the a precceding and succeding random vectors if there is a word
                    if a <= k:
                        self.__cv[words[k]] += self.__rv[words[k-a]]
                    if a + k < len(words):
                        self.__cv[words[k]] += self.__rv[words[k+a]]



    def find_nearest(self, words, k=5, metric='cosine'):
        """
        Function returning k nearest neighbors for each word in `words`
        """
        # YOUR CODE HERE

        nearest = NearestNeighbors(n_neighbors=k,metric=metric)
        # Train KNN on context vectors
        X = list(self.__cv.values())
        # Set label for each cv to the string of the word
        nearest.fit(X,list(self.__cv.keys()))

        point = np.zeros(self.__dim)
        # Save vocab as array. Context vector i for word i corresponds to index i in vocab
        vocab = np.array(list(self.__vocab))

        if len(words) == 1:
            if words[0] in self.__cv:
                point = np.array(self.__cv[words[0]]).reshape(1,-1)
                # Find k nearest neighbors to the context vector of the word
                closest  = nearest.kneighbors(point,return_distance=False)   
                # Return strings of closest neighbors
                close = vocab[closest[0]]
                return [close]
            else:
                return ['Word does not exist in vocab']

        else:
            close = []
            for i in range(len(words)):  
                if words[i] in self.__cv:
                    point = np.array(self.__cv[words[i]]).reshape(1,-1)
                    # Find k nearest neighbors to the context vector of the word
                    closest  = nearest.kneighbors(point,return_distance=False)
                    # Return strings of closest neighbors
                    close.append(vocab[closest[0]])
                else:
                    close.append("Word does not exist in vocab")

            return np.array(close)


    def get_word_vector(self, word):
        """
        Returns a trained vector for the word
        """
        # YOUR CODE HERE
        if word in self.__cv:
            return self.__cv[word]
        else:
            return None


    def vocab_exists(self):
        return os.path.exists('vocab.txt')


    def read_vocabulary(self):
        vocab_exists = self.vocab_exists()
        if vocab_exists:
            with open('vocab.txt') as f:
                for line in f:
                    self.__vocab.add(line.strip())
        self.__i2w = list(self.__vocab)
        return vocab_exists


    def write_vocabulary(self):
        with open('vocab.txt', 'w') as f:
            for w in self.__vocab:
                f.write('{}\n'.format(w))


    def train(self):
        """
        Main function call to train word embeddings
        """
        spinner = Halo(spinner='arrow3')

        if self.vocab_exists():
            spinner.start(text="Reading vocabulary...")
            start = time.time()
            print("-------")
            ri.read_vocabulary()
            spinner.succeed(text="Read vocabulary in {}s. Size: {} words".format(round(time.time() - start, 2), ri.vocabulary_size))
        else:
            spinner.start(text="Building vocabulary...")
            start = time.time()
            print("*******")
            ri.build_vocabulary()
            spinner.succeed(text="Built vocabulary in {}s. Size: {} words".format(round(time.time() - start, 2), ri.vocabulary_size))
        
        spinner.start(text="Creating vectors using random indexing...")
        start = time.time()
        ri.create_word_vectors()
        spinner.succeed("Created random indexing vectors in {}s.".format(round(time.time() - start, 2)))

        spinner.succeed(text="Execution is finished! Please enter words of interest (separated by space):")


    def train_and_persist(self):
        """
        Trains word embeddings and enters the interactive loop,
        where you can enter a word and get a list of k nearest neighours.
        """
        self.train()
        text = input('> ')
        while text != 'exit':
            text = text.split()
            neighbors = ri.find_nearest(text)

            for w, n in zip(text, neighbors):
                print("Neighbors for {}: {}".format(w, n))
            text = input('> ')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Indexing word embeddings')
    parser.add_argument('-fv', '--force-vocabulary', action='store_true', help='regenerate vocabulary')
    parser.add_argument('-c', '--cleaning', action='store_true', default=False)
    parser.add_argument('-co', '--cleaned_output', default='cleaned_example.txt', help='Output file name for the cleaned text')
    args = parser.parse_args()

    if args.force_vocabulary:
        os.remove('vocab.txt')

    if args.cleaning:
        # ri = RandomIndexing([os.path.join('data', 'example.txt')])
        ri = RandomIndexing(['example.txt'])
        with open(args.cleaned_output, 'w') as f:
            for part in ri.text_gen():
                f.write("{}\n".format(" ".join(part)))
    else:
        dir_name = "data"
        filenames = [os.path.join(dir_name, fn) for fn in os.listdir(dir_name)]

        ri = RandomIndexing(filenames)
        ri.train_and_persist()
