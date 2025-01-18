# Ref: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1086/handouts/cs224n-lecture2-language-models-slides.pdf
from math import log, exp
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

    def _calculate_smoothed_Nr_counts(self, small_r_threshold: int = 5) -> dict[int, float]:
        """
        To calculate smoothed Nr counts using the Church and Gale method (ref: Page 5 of https://www.d.umn.edu/~tpederse/Courses/CS8761-FALL02/Code/sgt-gale.pdf)
        1. Calculate Zr values.
        2. Fit linear regression line to log(Zr) vs log(r).
        3. Use linear regression for large r, original Nr for small r.
        :param small_r_threshold: 5 by default. This is the threshold below which values of 'r' are considered to be "small".
        """
        nonzero_counts = [(r, Nr) for r, Nr in sorted(self.freq_of_ngram_freq.items()) if Nr > 0]
        Zr_values = {}

        for i in range(len(nonzero_counts)):
            r = nonzero_counts[i][0]
            Nr = nonzero_counts[i][1]

            if i == 0:  # for the first non-zero count
                q = 0
                t = nonzero_counts[i + 1][0]
            elif i == len(nonzero_counts) - 1:  # for the last non-zero count (r_last)
                q = nonzero_counts[i - 1][0]
                Zr_values[r] = Nr / (r - q)
                continue
            else:  # for middle cases (usual situation)
                q = nonzero_counts[i - 1][0]
                t = nonzero_counts[i + 1][0]

            Zr_values[r] = Nr / (0.5 * (t - q))

        # fitting linear regression line to log-log plot
        log_r = np.array([log(r) for r in Zr_values.keys()])
        log_Zr = np.array([log(Zr) for Zr in Zr_values.values()])
        a, b = np.polyfit(log_r, log_Zr, 1)

        # finally, calculating the smoothed values.
        smoothed_counts = {}
        max_r = max(self.freq_of_ngram_freq.keys())

        for r in range(1, max_r + 1):
            if r <= small_r_threshold:
                smoothed_counts[r] = self.freq_of_ngram_freq.get(r, 0)
            else:
                log_SNr = a + b * log(r)
                smoothed_counts[r] = exp(log_SNr)

        return defaultdict(float, smoothed_counts)

    def _calculate_good_turing_r_star(self, r: int) -> float:
        """
        To calculate the r* value for Good-Turing smoothing.
        :param r:
        :return:
        """
        if r == 0:
            return 0

        if not hasattr(self, '_smoothed_Nr'):
            self._smoothed_Nr = self._calculate_smoothed_Nr_counts()

        S_Nr = self._smoothed_Nr[r]
        S_Nr_plus_1 = self._smoothed_Nr[r + 1]

        if S_Nr == 0:  # fallback, but should never trigger this.
            return r

        return ((r + 1) * S_Nr_plus_1) / S_Nr

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
                return self.freq_of_ngram_freq.get(1, 0) / self.total_ngrams


            # as we are using Turing estimate for small r values and Good-Turing estimate for large r values, we need to renormalize them (ref: Page 8 and 9 of https://www.d.umn.edu/~tpederse/Courses/CS8761-FALL02/Code/sgt-gale.pdf)
            # get unnormalized probability using r*
            p_unnorm = self._calculate_good_turing_r_star(ngram_count) / self.total_ngrams

            # calculate sum of unnormalized probabilities for all possible continuations
            sum_p_unnorm = 0
            for possible_ngram in self.ngrams:
                if possible_ngram[:-1] == context:  # same context
                    r = self.ngrams[possible_ngram]
                    if r >= 1:
                        sum_p_unnorm += self._calculate_good_turing_r_star(r) / self.total_ngrams

            # apply renormalization formula
            N1 = self.freq_of_ngram_freq.get(1, 0)
            return (1 - N1/self.total_ngrams) * (p_unnorm / sum_p_unnorm)

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
