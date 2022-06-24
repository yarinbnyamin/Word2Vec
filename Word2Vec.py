import pickle
import numpy as np
import re
from collections import Counter

import nltk
from nltk import skipgrams
from nltk.corpus import stopwords
from numpy import random


def normalize_text(fn):
    """ Loading a text file and normalizing it, returning a list of sentences.

    Args:
        fn: full path to the text file to process
    """
    sentences = []

    # try:
    #   file = open(fn, 'r')
    # except :
    file = open(fn, "r", encoding="cp1252")

    lines = file.readlines()
    file.close()

    for line in lines:
        line = line.strip()

        if line == "":
            continue

        line = re.sub(r'["|“|”|.|!|?|,]+', "", line)
        line = line.lower()

        sentences.append(line)

    return sentences


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def load_model(fn):
    """ Loads a model pickle and return it.

    Args:
        fn: the full path to the model to load.
    """

    file = open(fn, "rb")
    sg_model = pickle.load(file)
    file.close()

    return sg_model


class SkipGram:
    def __init__(
        self, sentences, d=100, neg_samples=4, context=4, word_count_threshold=5
    ):
        self.sentences = sentences
        self.d = d  # embedding dimension
        self.neg_samples = (
            neg_samples  # num of negative samples for one positive sample
        )
        self.context = (
            context  # the size of the context window (not counting the target word)
        )
        self.word_count_threshold = word_count_threshold  # ignore low frequency words (appearing under the threshold)
        self.T = []  # embedding matrix
        self.C = []  # embedding matrix

        # word:count dictionary
        counts = Counter()
        for line in sentences:
            counts.update(line.split())

        # ignore low frequency words and stopwords
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))
        counts = Counter(
            {
                k: v
                for k, v in counts.items()
                if k not in stop_words and v >= word_count_threshold and "’" not in k
            }
        )
        self.word_count = dict(counts)

        # how many unique words in our dictionary
        self.vocab_size = len(counts)

        # word-index map
        self.word_index = {}
        index = 0
        for word in dict(counts).keys():
            self.word_index[word] = index
            index += 1

    def compute_similarity(self, w1, w2):
        """ Returns the cosine similarity (in [0,1]) between the specified words.

        Args:
            w1: a word
            w2: a word
        Returns: a float in [0,1]; defaults to 0.0 if one of specified words is OOV.
        """
        sim = 0.0  # default

        if w1 not in self.word_index or w2 not in self.word_index:
            return sim  # default

        nx = self.T[:, self.word_index[w1]]
        ny = self.T[:, self.word_index[w2]]

        sim = np.dot(nx, ny) / (np.linalg.norm(nx) * np.linalg.norm(ny))

        return sim

    def get_closest_words(self, w, n=5):
        """Returns a list containing the n words that are the closest to the specified word.

        Args:
            w: the word to find close words to.
            n: the number of words to return. Defaults to 5.
        """

        if w not in self.word_index:
            return []  # default

        y = self.feed_forward(w)
        n = min(n, self.vocab_size)

        candidates = []
        for word, index in self.word_index.items():
            candidates.append((word, y[index]))

        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        candidates = [word for word, score in candidates]
        return candidates[:n]

    def feed_forward(self, w):
        """Returns a normalized output layer for a word

        Args:
            w: word to get output for
        """

        # Input layer x T = Hidden layer
        input_layer_id = self.word_index[w]
        hidden = self.T[:, input_layer_id][:, None]

        # Hidden layer x C = Output layer
        output_layer = np.dot(self.C, hidden)
        y = sigmoid(output_layer)

        return y

    def learn_embeddings(self, step_size=0.001, epochs=50, save_as="vocab.pickle"):
        """Returns a trained embedding models and saves it as specified name
        this function also backup the last model ( done by the last epoch)
        with the file name of temp.pickle

        Args:
            step_size: step size for  the gradient descent. Defaults to 0.0001
            epochs: number or training epochs. Defaults to 50
            save_as: name of the trained model
        """

        print("start preprocessing")
        vocab_size = self.vocab_size

        # in skip gram we want to predict the context words from the target words
        T = np.random.rand(self.d, vocab_size)  # embedding matrix of target words
        C = np.random.rand(vocab_size, self.d)  # embedding matrix of context words

        # create learning vectors
        learning_vector = []
        for sentence in self.sentences:
            dic = {}

            # create positive and negative lists
            pos_lst = list(skipgrams(sentence.split(), int(self.context / 2), 1))
            pos_lst += [(tup[1], tup[0]) for tup in pos_lst]
            neg_lst = []
            for _ in range(self.neg_samples):
                neg_lst += [
                    (word, random.choice(list(self.word_count.keys())))
                    for word in sentence.split()
                ]

            # merge to key value
            pos = {}
            for x, y in pos_lst:
                if x not in self.word_count or y not in self.word_count:
                    continue
                pos.setdefault(x, []).append(y)
            neg = {}
            for x, y in neg_lst:
                if x not in self.word_count or y not in self.word_count:
                    continue
                neg.setdefault(x, []).append(y)

            # create the learning context vector
            for key, val in pos.items():
                dic[key] = np.zeros(self.vocab_size, dtype=int)
                for v in val:
                    dic[key][self.word_index[v]] += 1
                for v in neg[key]:
                    dic[key][self.word_index[v]] -= 1

            learning_vector += dic.items()
        print("done preprocessing")

        print("start training")
        for i in range(epochs):
            print(f"epoch {i + 1}")

            # learning:
            for key, val in learning_vector:
                # Input layer x T = Hidden layer
                input_layer_id = self.word_index[key]
                input_layer = np.zeros(self.vocab_size, dtype=int)
                input_layer[input_layer_id] = 1
                input_layer = np.vstack(input_layer)

                hidden = T[:, input_layer_id][:, None]

                # Hidden layer x C = Output layer
                output_layer = np.dot(C, hidden)
                y = sigmoid(output_layer)

                # calculate gradient
                e = y - val.reshape(self.vocab_size, 1)
                outer_grad = np.dot(hidden, e.T).T
                inner_grad = np.dot(input_layer, np.dot(C.T, e).T).T
                C -= step_size * outer_grad
                T -= step_size * inner_grad

            # backup the last trained model (the last epoch)
            self.T = T
            self.C = C
            with open("temp.pickle", "wb") as f:
                pickle.dump(self, f)

            step_size *= 1 / (1 + step_size * i)
        print("done training")

        self.T = T
        self.C = C

        with open(save_as, "wb") as f:
            pickle.dump(self, f)
        print("saved as 'vocab.pickle' file")

        return T, C


class SemantleSolver:
    def __init__(self, sg_model, top_ranks=1000, give_up=500):
        """ Initializes the Semantle solver

        Args:
            sg_model: a SkipGram object
            top_ranks: the proximity rank over which the rank is returned by check_word
            give_up: the max number of words to guess before quitting in disgrace.
        """
        self.sg_model = sg_model
        self.top_ranks = top_ranks
        self.top_ranks_dict = None  # the word to guess top rank dict - word:(sim, rank)
        self.give_up = give_up
        self.target_word = None  # the word to guess
        self.target_word_index = None  # the word to guess index

        self.set_target_word()

    def set_target_word(self, target=None):
        """ Sets a target word for the solver to guess.
             Samples from the model vocabulary if no word is specified,
             or that the specified word is not part of the model.
        """

        word_index = self.sg_model.word_index
        if target is None or target not in word_index:
            target = random.choice(list(word_index.keys()))

        self.target_word = target
        self.target_word_index = word_index[target]

        top_ranks_list = self.sg_model.get_closest_words(target, self.top_ranks)
        self.top_ranks_dict = {}
        i = 1
        for word in top_ranks_list:
            self.top_ranks_dict[word] = (
                self.sg_model.compute_similarity(target, word),
                i,
            )
            i += 1

    def check_word(self, w):
        """ Returns a tuple sim, rank (float, int), indicating the similarity between the specified word and the target
            word and the distance rank if the specified word is within the specified rank.
            The returned rank should be -1 if the word is not among the self.top_ranks closest words.

        Args:
             w: the word to guess (match against the target word)
        """

        sim = 0.0
        rank = -1

        if w in self.top_ranks_dict:
            sim, rank = self.top_ranks_dict[w]

        return sim, rank

    def semantle_game(self):
        """Returns won, shots - won is true if the player guessed the correct word,
        shots is a list of triplets (w, sim,rank). sim and rank are returned by  check_word(w)
        len(shots) cannot exceed self.give_up
        shots[0] should hold the first guess
        shots[-1] should hold the last guess (hopefully the match)
        """
        shots = []
        won = False

        print("Guess the word")
        print('If you want to give up please write "give up" as your guess')
        for i in range(self.give_up):
            guess = input("Your next guess is :")
            if guess == "give up":
                break

            sim, rank = self.check_word(guess)
            shots.append((guess, sim, rank))

            if guess == self.target_word:
                won = True
                break

            if rank == -1:
                print(f"Your guess {guess} not close to the target word")
            else:
                print(
                    f"Your guess {guess} ranked as top {rank} with similarity of {sim}"
                )

        print(f"The correct word was {self.target_word}")

        if won:
            print("You guessed correctly")
        else:
            print("You lost, better luck next time")

        self.set_target_word()

        return won, shots
