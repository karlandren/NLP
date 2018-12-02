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
        answer = ''.join(filter(whitelist.__contains__, line))
        answer = ' '.join(answer.split())
        return [answer]


    def text_gen(self):
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    yield self.clean_line(line)


    def build_vocabulary(self):
        """
        Build vocabulary of words from the provided text files
        """
        # YOUR CODE HERE
        # print(0)

        for i in self.text_gen():
            
            # print(1)
            for k in i[0].split():
                # print(2)
                # if k.capitalize() not in self.__vocab:
                if k not in self.__vocab:

                    # print(k)
                    # print(3)
                    # self.__vocab.add(k.capitalize())
                    self.__vocab.add(k)

        print("*")
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
        for i in self.__vocab:
            self.__cv[i] = np.zeros(self.__dim)

        for i in self.__vocab:
            self.__rv[i] = np.where(np.random.rand(self.__dim) > 0.5, 1 , -1)
            # self.__rv[i][8] = 0
            # self.__rv[i][9] = 0

        for i in self.text_gen():
            # words = [w.capitalize() for w in i[0].split()]
            words = i[0].split()

            for k in range(len(words)):
                for a in range(1,self.__lws+1):
                    # print(words[k] + "**")
                    if a <= k:
                        self.__cv[words[k]] = self.__cv[words[k]] + self.__rv[words[k-a]]
                    if a+k < len(words):
                        self.__cv[words[k]] = self.__cv[words[k]] + self.__rv[words[k+a]]

        pass


    def find_nearest(self, words, k=5, metric='cosine'):
        """
        Function returning k nearest neighbors for each word in `words`
        """
        # YOUR CODE HERE
        nearest = NearestNeighbors(n_neighbors=k,metric=metric)
        X = list(self.__cv.values())
        nearest.fit(X ,list(self.__cv.keys()))
        point = np.zeros(self.__dim)
        vocab = np.array(list(self.__vocab))

        if len(words) == 1:
            point = np.array(self.__cv[words[0]]).reshape(1,-1)
            closest  = nearest.kneighbors(point,return_distance=False)    
            close = vocab[closest[0]]
            return [close]

        else:
            close = []

            for i in range(len(words)):  
                point = np.array(self.__cv[words[i]]).reshape(1,-1)
                closest  = nearest.kneighbors(point,return_distance=False)
                close.append(vocab[closest[0]])

    
            return [np.array(close)]


    def get_word_vector(self, word):
        """
        Returns a trained vector for the word
        """
        # YOUR CODE HERE
    
        return self.__cv[word]


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
