import sys
from math import log, exp
from typing import Literal
from _collections import defaultdict
import numpy as np
import math

class NGramModel:
    def __init__(self, N: int,
                 smoothing_type: Literal['none', 'laplace', 'good-turing'] = "None"):
        """
        :param N: This is the size of each 'N'-gram to be generated.
        :param smoothing_type: Type of smoothing to be performed while training the language model. 'none' by default. Choose one from {'none', 'laplace', 'good-turing'}.
        """
        self.n = N
        self.smoothing_type = smoothing_type
        self.ngrams = defaultdict(int)
        self.total_tokens = 0  # Total number of tokens, needed for unigram case.
        self.context_counts = defaultdict(int)
        self.freq_of_ngram_freq = defaultdict(int)  # for Good-Turing smoothing (N_r counts)
        self.vocab_size = 0  # to compute V for Laplace smoothing
        self.total_ngrams = 0  # to compute the total number of N-grams for Good-Turing smoothing (N value for Good-Turing)
        self.context_unnorm_sums = defaultdict(float)  # Store pre-calculated sums for each context

    def train(self, tokenized_sentences: list[list[str]]) -> None:
        """
        Train the N-gram model.
        :param tokenized_sentences: List of sentences, each of which is a list of tokens in that sentence.
        :return:
        """
        self.vocab_size = len(set(word for sentence in tokenized_sentences for word in sentence))
        self.total_tokens = sum(len(sentence) for sentence in tokenized_sentences)  # N for linear interpolation

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
            
            # Calculate context sums after we have all ngram counts
            for ngram, count in self.ngrams.items():
                context = ngram[:-1]
                if count >= 1:
                    r_star = self._calculate_good_turing_r_star(count)
                    self.context_unnorm_sums[context] += r_star / self.total_ngrams

    def _calculate_smoothed_Nr_counts(self, small_r_threshold: int = 5) -> dict[int, float]:
        """
        To calculate smoothed Nr counts using the Church and Gale method (ref: Page 5 of https://www.d.umn.edu/~tpederse/Courses/CS8761-FALL02/Code/sgt-gale.pdf)
        1. Calculate Zr values.
        2. Fit linear regression line to log(Zr) vs log(r).
        3. Use linear regression for large r, original Nr for small r.
        :param small_r_threshold: 5 by default. This is the threshold below which values of 'r' are considered to be "small".
        :return:
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

    def calculate_probability(self, ngram: tuple[str, ...], context: tuple[str, ...]) -> float:
        """
        A general probability calculation function to handle various smoothing methods.
        :param ngram: The ngram whose probability we wish to calculate.
        :param context: The context (usually, the first n-1 words of the ngram.)
        :return:
        """
        ngram_count = self.ngrams.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)

        if self.smoothing_type == 'none':
            if len(context) == 0:  # unigram case
                return ngram_count / self.total_tokens
            if context_count == 0:  # context hasn't been seen previously, so, probability of occurrence is trivially 0.
                return 0.0
            return ngram_count / context_count

        elif self.smoothing_type == 'laplace':
            if len(context) == 0:  # unigram case
                return (ngram_count + 1) / (self.total_tokens + self.vocab_size)
            return (ngram_count + 1) / (context_count + self.vocab_size)

        elif self.smoothing_type == 'good-turing':
            if ngram_count == 0:
                return self.freq_of_ngram_freq.get(1, 0) / self.total_ngrams

            # Use pre-calculated sum for normalization
            p_unnorm = self._calculate_good_turing_r_star(ngram_count) / self.total_ngrams
            sum_p_unnorm = self.context_unnorm_sums[context]

            # apply renormalization formula
            N1 = self.freq_of_ngram_freq.get(1, 0)
            return (1 - N1 / self.total_ngrams) * (p_unnorm / sum_p_unnorm)

    def predict_next_word(self, tokenized_sentence: list[str], n_candidates_for_next_word: int) -> dict:
        """
        Given a sentence, predict n_next_words possible candidates for the next word, along with their probabilities.
        :param tokenized_sentence: Tokenized sentence (list of strings) for which next word is supposed to be predicted.
        :param n_candidates_for_next_word: Number of possible candidates for the next word, to predict.
        :return:
        """
        tokens = ['<s>'] * (self.n - 1) + tokenized_sentence
        context = tuple(tokens[-(self.n - 1):])

        predictions = {}
        for ngram in self.ngrams:
            if ngram[:-1] == context:
                probability = self.calculate_probability(ngram, context)
                predictions[ngram[-1]] = probability

        # return the top n_next_words after sorting them by decreasing order of probabilities.
        sorted_predictions = dict(
            sorted(predictions.items(), key=lambda item: item[1], reverse=True)[:n_candidates_for_next_word])
        return sorted_predictions

    def calculate_probability_of_sentence(self, tokenized_sentence: list[str]) -> float:
        """
        Given a sentence, calculate the probability of that sentence occurring.
        :param tokenized_sentence: Tokenized input sentence (list of str) to calculate probability for.
        :return:
        """
        tokens = ['<s>'] * (self.n - 1) + tokenized_sentence + ['</s>']
        probability = 1.0

        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            context = tuple(tokens[i:i + self.n - 1])
            probability *= self.calculate_probability(ngram, context)

        return probability

    def generate_sentence_next_n_words(self, tokenized_sentence: list[str], n: int) -> list[str]:
        """
        Given a sentence, generate the next n most likely words of that sentence (at each stage, pick the most likely word to occur next.)
        :param tokenized_sentence: Tokenized sentence (list of str) for which we are trying to generate next n words.
        :param n: Number of next words to predict.
        :return:
        """
        for _ in range(n):
            next_words_list = list(self.predict_next_word(tokenized_sentence, 1).keys())
            if not next_words_list:  # if there is no word that can come next as this context hasn't been seen before
                break
            next_word = next_words_list[0]

            # stop if we hit the end of sentence token
            if next_word == '</s>':
                break

            tokenized_sentence.append(next_word)

        return tokenized_sentence

    def perplexity(self, tokenized_sentence: list[str]) -> float:
        """
        Calculate the perplexity of the language model for a given sentence.
        Perplexity = exp(-1/N * sum(log P(w_i|w_{i-n+1}...w_{i-1})))
        where N is the total number of ngrams in the sentence.

        :param tokenized_sentence: Tokenized sentence.
        :return: Perplexity value
        """
        log_prob_sum = 0.0
        total_sentence_ngrams = 0

        # add start and end tokens
        tokens = ['<s>'] * (self.n - 1) + tokenized_sentence + ['</s>']
        total_sentence_ngrams += len(tokenized_sentence) + 1  # +1 to count the </s> token

        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            context = tuple(tokens[i:i + self.n - 1])

            # get probability of the current word, given context
            prob = self.calculate_probability(ngram, context)

            # handle zero probability case (smoothing should prevent this if getting perplexity on a smoothed model, but just in case)
            if prob <= 0:
                prob = sys.float_info.epsilon

            log_prob_sum += math.log2(prob)

        # calculate and return perplexity
        return math.pow(2, -1 * log_prob_sum / total_sentence_ngrams)