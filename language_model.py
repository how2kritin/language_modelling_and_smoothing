# Ref: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1086/handouts/cs224n-lecture2-language-models-slides.pdf
from math import log
from typing import Literal
from collections import defaultdict
import numpy as np
from tokenizer import word_tokenizer
import argparse


class NGramModel:
    def __init__(self, N: int,
                 smoothing_type: Literal['none', 'laplace', 'good-turing', 'linear_interpolation'] = "None"):
        """
        :param N: This is the size of each 'N'-gram to be generated.
        :param smoothing_type: Type of smoothing to be performed while training the language model. 'none' by default. Choose one from {'none', 'laplace', 'good-turing', 'linear_interpolation'}.
        """
        self.n = N
        self.smoothing_type = smoothing_type
        self.ngrams = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.freq_of_ngram_freq = defaultdict(int)  # for Good-Turing smoothing (N_r counts)
        self.vocab_size = 0  # to compute V for Laplace smoothing
        self.total_ngrams = 0  # to compute the total number of N-grams for Good-Turing smoothing (N value)

    def train(self, inp_str: str) -> None:
        """
        Train the N-gram model.
        :param inp_str: Corpus of data on which the model will be trained.
        :return:
        """
        tokenized_sentences = word_tokenizer(inp_str)
        self.vocab_size = len(set(word for sentence in tokenized_sentences for word in sentence))

        for sentence in tokenized_sentences:
            # this <s> padding n-1 times at the start is so that the start of the sentence has enough context to calculate probabilities.
            tokens = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                context = tuple(tokens[i:i + self.n - 1])
                self.ngrams[ngram] += 1
                self.context_counts[context] += 1
                self.total_ngrams += 1

        # to compute frequency of ngram frequencies for Good-Turing smoothing.
        if self.smoothing_type == 'good-turing':
            for count in self.ngrams.values():
                self.freq_of_ngram_freq[count] += 1

    def _calculate_good_turing_r_star(self, r: int, small_r_threshold: int = 5) -> float:
        """
        To calculate the r* value for Good-Turing smoothing.
        :param r:
        :param small_r_threshold: 5 by default. This is the threshold below which values of 'r' are considered to be "small".
        :return:
        """
        if r == 0:
            return 0  # unseen events are handled separately.

        # for small values of r, do not perform any smoothing.
        if r < small_r_threshold:
            Nr = self.freq_of_ngram_freq.get(r, 0)
            Nr_plus_1 = self.freq_of_ngram_freq.get(r + 1, 0)
            if Nr == 0:
                return r  # fallback to original count if no frequency data
            return ((r + 1) * Nr_plus_1) / Nr

        # for large values of r, perform linear regression on the log relationship line log(Nr) = a + blog(r) to get S(r).
        # collecting points for regression here
        points = [(log(r), log(Nr)) for r, Nr in self.freq_of_ngram_freq.items() if Nr > 0]
        if len(points) < 2:
            return self.freq_of_ngram_freq.get(r, 0)  # fallback if regression fails

        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])

        # performing linear regression here on the log line
        a, b = np.polyfit(x, y, 1)
        S_r = np.exp(a + b * log(r))
        S_r_plus_1 = np.exp(a + b * log(r + 1))

        return ((r + 1) * S_r_plus_1) / S_r

    def _calculate_probability(self, ngram: tuple[str, ...], context: tuple[str, ...]) -> float:
        """
        A general probability calculation function to handle various smoothing methods.
        :param ngram: The ngram whose probability we wish to calculate.
        :param context: The context (usually, the first n-1 words of the ngram.)
        :return:
        """
        ngram_count = self.ngrams.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)

        if self.smoothing_type == 'none':
            if context_count == 0:
                return 0.0  # if the context has never occurred, then the probability is trivially 0.
            return ngram_count / context_count

        elif self.smoothing_type == 'laplace':  # also called "Add-One" smoothing.
            return (ngram_count + 1) / (context_count + self.vocab_size)

        elif self.smoothing_type == 'good-turing':
            if ngram_count == 0:
                # using N1/N for unseen n-grams
                N1 = self.freq_of_ngram_freq.get(1, 0)
                return N1 / self.total_ngrams

            # calculate r* using Good-Turing formula, and use that here.
            r_star = self._calculate_good_turing_r_star(ngram_count)
            return r_star / self.total_ngrams

        return 0.0  # will reach here only for an unimplemented smoothing_type passed during initialization of an object of this class.

    def predict_next_word(self, sentence: str, n_next_words: int) -> dict:
        """
        Given a sentence, predict n_next_words possible candidates for the next word, along with their probabilities.
        :param sentence: Sentence for which next word is supposed to be predicted.
        :param n_next_words: Number of possible candidates for the next word, to predict.
        :return:
        """
        tokens = ['<s>'] * (self.n - 1) + word_tokenizer(sentence)[0]
        context = tuple(tokens[-(self.n - 1):])

        predictions = {}
        for ngram in self.ngrams:
            if ngram[:-1] == context:
                probability = self._calculate_probability(ngram, context)
                predictions[ngram[-1]] = probability

        # return the top n_next_words after sorting them by decreasing order of probabilities.
        sorted_predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True)[:n_next_words])
        return sorted_predictions

    def calculate_probability_of_sentence(self, sentence: str) -> float:
        """
        Given a sentence, calculate the probability of that sentence occurring.
        :param sentence:
        :return:
        """
        tokens = ['<s>'] * (self.n - 1) + word_tokenizer(sentence)[0] + ['</s>']
        probability = 1.0

        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            context = tuple(tokens[i:i + self.n - 1])
            probability *= self._calculate_probability(ngram, context)

        return probability


def main(N: int, lm_type: str, corpus_path: str) -> None:
    match lm_type:
        case 'l':
            smoothing_type = 'laplace'
        case 'g':
            smoothing_type = 'good-turing'
        case 'i':
            smoothing_type = 'linear_interpolation'
        case _:
            smoothing_type = 'none'
    ngm = NGramModel(N=N, smoothing_type=smoothing_type)

    try:
        with open(corpus_path, "r") as file:
            text = file.read()
        ngm.train(text)
    except FileNotFoundError:
        raise FileNotFoundError("Unable to find a file at that path to use as the corpus!")

    input_sentence = str(input('input sentence: '))
    print('score: ', ngm.calculate_probability_of_sentence(sentence=input_sentence))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int)
    parser.add_argument('lm_type', type=str)
    parser.add_argument('corpus_path', type=str)
    args = parser.parse_args()
    main(args.N, args.lm_type, args.corpus_path)
