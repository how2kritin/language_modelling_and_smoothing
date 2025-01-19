# Refs: 
# 1. https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1086/handouts/cs224n-lecture2-language-models-slides.pdf
# 2. https://www.d.umn.edu/~tpederse/Courses/CS8761-FALL02/Code/sgt-gale.pdf
# 3. https://aclanthology.org/A00-1031.pdf
import string
from math import log, exp
from typing import Literal
from collections import defaultdict
import numpy as np
from tokenizer import word_tokenizer
import argparse


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
            return (1 - N1 / self.total_ngrams) * (p_unnorm / sum_p_unnorm)

    def predict_next_word(self, sentence: list[str], n_candidates_for_next_word: int) -> dict:
        """
        Given a sentence, predict n_next_words possible candidates for the next word, along with their probabilities.
        :param sentence: Tokenized sentence (list of strings) for which next word is supposed to be predicted.
        :param n_candidates_for_next_word: Number of possible candidates for the next word, to predict.
        :return:
        """
        tokens = ['<s>'] * (self.n - 1) + sentence
        context = tuple(tokens[-(self.n - 1):])

        predictions = {}
        for ngram in self.ngrams:
            if ngram[:-1] == context:
                probability = self._calculate_probability(ngram, context)
                predictions[ngram[-1]] = probability

        # return the top n_next_words after sorting them by decreasing order of probabilities.
        sorted_predictions = dict(
            sorted(predictions.items(), key=lambda item: item[1], reverse=True)[:n_candidates_for_next_word])
        return sorted_predictions

    def calculate_probability_of_sentence(self, sentence: list[str]) -> float:
        """
        Given a sentence, calculate the probability of that sentence occurring.
        :param sentence: Tokenized input sentence (list of str) to calculate probability for.
        :return:
        """
        tokens = ['<s>'] * (self.n - 1) + sentence + ['</s>']
        probability = 1.0

        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            context = tuple(tokens[i:i + self.n - 1])
            probability *= self._calculate_probability(ngram, context)

        return probability

    def generate_sentence_next_n_words(self, sentence: list[str], n: int) -> str:
        """
        Given a sentence, generate the next n most likely words of that sentence (at each stage, pick the most likely word to occur next.)
        :param sentence: Tokenized sentence (list of str) for which we are trying to generate next n words.
        :param n: Number of next words to predict.
        :return:
        """
        for _ in range(n):
            next_words_list = list(self.predict_next_word(sentence, 1).keys())
            if not next_words_list: # if there is no word that can come next as this context hasn't been seen before
                break
            next_word = next_words_list[0]

            # stop if we hit the end of sentence token
            if next_word == '</s>':
                break

            if next_word not in string.punctuation and next_word not in {'<s>', '</s>'}:
                sentence += " " + next_word
            else:
                sentence += next_word

        return sentence


class LinearInterpolationOfNGramModels:
    def __init__(self, N: int) -> None:
        """
        :param N: This is the size of the largest 'N'-gram model used in this linear interpolation of models.
        """
        self.n = N
        self.lambdas = None  # for linear interpolation weights
        self.ngram_models = [NGramModel(N=n, smoothing_type='none') for n in range(1, self.n + 1)]

    def train(self, tokenized_sentences: list[list[str]]) -> None:
        """
        Train each i-gram model where 1 <= i <= N, and calculate the linear interpolation weights (lambdas).
        :param tokenized_sentences: List of sentences, each of which is a list of tokens in that sentence.
        :return:
        """
        for model in self.ngram_models:
            model.train(tokenized_sentences)

        # calculate linear interpolation weights.
        self._calculate_linear_interpolation_weights()

    def _calculate_linear_interpolation_weights(self) -> None:
        """
        Calculate lambda weights for linear interpolation following the algorithm:
        For each n-gram:
            Compare the ratios:
                (f(w_{i-n+1}...w_i) - 1) / (f(w_{i-n+1}...w_{i-1}) - 1)
                (f(w_{i-n+2}...w_i) - 1) / (f(w_{i-n+2}...w_{i-1}) - 1)
                ...
                (f(w_i) - 1) / (N - 1)
            Increment corresponding lambda based on which value is maximum
        Normalize the lambdas to sum to 1.
        Ref: Page 3 of https://aclanthology.org/A00-1031.pdf
        """
        self.lambdas = [0.0] * self.n  # Initialize all lambdas to 0
        N = sum(self.ngram_models[0].ngrams.values())  # total number of tokens in corpus (total number of unigrams)

        # for each observed N-gram
        for ngram, count in self.ngram_models[-1].ngrams.items():
            if len(ngram) != self.n or count == 0:
                continue

            # calculate values to compare for each order k-gram
            values = []

            # for each k from 1 to n
            for k in range(1, self.n + 1):
                sub_ngram = ngram[-k:]  # take the last k tokens
                sub_context = sub_ngram[:-1]  # take all but the last token

                if k == 1:  # unigram case
                    if N > 1:
                        values.append((count - 1) / (N - 1))
                    else:
                        values.append(0.0)
                else:
                    sub_count = self.ngram_models[k - 1].ngrams.get(sub_ngram, 0)
                    context_count = self.ngram_models[k - 1].context_counts.get(sub_context, 0)
                    if context_count > 1:  # need at least 2 occurrences for valid ratio
                        values.append((sub_count - 1) / (context_count - 1))
                    else:
                        values.append(0.0)

            # increment lambda corresponding to maximum value
            max_index = values.index(max(values))
            self.lambdas[max_index] += count

        # normalize lambdas to sum to 1
        total = sum(self.lambdas)
        if total > 0:
            self.lambdas = [l / total for l in self.lambdas]
        else:
            # just fallback to equal weights if no data (i.e., total of all lambdas came out to be 0 because each of them remained 0).
            self.lambdas = [1.0 / self.n] * self.n

    def _calculate_probability(self, ngram: tuple[str, ...]) -> float:
        if self.lambdas is None:  # this should be done during training. however, we are doing it here as well, for the sake of completeness.
            self._calculate_linear_interpolation_weights()

        probability = 0.0
        N = sum(self.ngram_models[0].ngrams.values())  # total number of tokens in corpus (total number of unigrams)

        # for each k from 1 to n
        for k in range(1, self.n + 1):
            sub_ngram = ngram[-k:]  # take the last k tokens
            sub_context = sub_ngram[:-1]  # take all but the last token

            if k == 1:  # unigram case
                if N > 0:
                    prob = self.ngram_models[0].ngrams.get(sub_ngram, 0) / N
                else:
                    prob = 0.0
            else:
                sub_count = self.ngram_models[k - 1].ngrams.get(sub_ngram, 0)
                context_count = self.ngram_models[k - 1].context_counts.get(sub_context, 0)
                if context_count > 0:
                    prob = sub_count / context_count
                else:
                    prob = 0.0

            probability += self.lambdas[k - 1] * prob

        return probability

    def predict_next_word(self, sentence: list[str], n_candidates_for_next_word: int) -> dict:
        """
        Given a sentence, predict n_next_words possible candidates for the next word, along with their probabilities.
        :param sentence: Tokenized sentence (list of strings) for which next word is supposed to be predicted.
        :param n_candidates_for_next_word: Number of possible candidates for the next word, to predict.
        :return:
        """
        tokens = ['<s>'] * (self.n - 1) + sentence
        context = tuple(tokens[-(self.n - 1):])

        # collect all possible next words from all n-gram models
        predictions = defaultdict(float)
        for k in range(1, self.n + 1):
            sub_context = context[-(k - 1):] if k > 1 else tuple()

            # look through all n-grams of order k
            for ngram in self.ngram_models[k - 1].ngrams:
                if len(ngram) == k and ngram[:-1] == sub_context:
                    next_word = ngram[-1]

                    # calculate probability using linear interpolation
                    full_ngram = context + (next_word,)
                    predictions[next_word] = self._calculate_probability(full_ngram)

        # return the top n_next_words after sorting them by decreasing order of probabilities.
        sorted_predictions = dict(
            sorted(predictions.items(), key=lambda item: item[1], reverse=True)[:n_candidates_for_next_word])
        return sorted_predictions

    def calculate_probability_of_sentence(self, sentence: list[str]) -> float:
        """
        Given a sentence, calculate the probability of that sentence occurring.
        :param sentence: Tokenized input sentence (list of str) to calculate probability for.
        :return: Probability of the sentence.
        """
        tokens = ['<s>'] * (self.n - 1) + sentence + ['</s>']
        probability = 1.0

        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            prob = self._calculate_probability(ngram)
            probability *= prob

        return probability

    def generate_sentence_next_n_words(self, sentence: list[str], n: int) -> str:
        """
        Given a sentence, generate the next n most likely words of that sentence (at each stage, pick the most likely word to occur next.)
        :param sentence: Tokenized sentence (list of str) for which we are trying to generate next n words.
        :param n: Number of next words to predict.
        :return:
        """
        for _ in range(n):
            next_words_list = list(self.predict_next_word(sentence, 1).keys())
            if not next_words_list: # if there is no word that can come next as this context hasn't been seen before
                break
            next_word = next_words_list[0]

            # stop if we hit the end of sentence token
            if next_word == '</s>':
                break

            if next_word not in string.punctuation and next_word not in {'<s>', '</s>'}:
                sentence += " " + next_word
            else:
                sentence += next_word

        return sentence


def main(N: int, lm_type: str, corpus_path: str) -> None:
    match lm_type:
        case 'l':
            smoothing_type = 'laplace'
        case 'g':
            smoothing_type = 'good-turing'
        case 'i':
            smoothing_type = 'linear_interpolation'
        case 'n':
            smoothing_type = 'none'

    try:
        with open(corpus_path, "r") as file:
            text = file.read()
        tokenized_sentences = word_tokenizer(text)
    except FileNotFoundError:
        raise FileNotFoundError("Unable to find a file at that path to use as the corpus!")

    if smoothing_type == 'linear_interpolation':
        ngm = LinearInterpolationOfNGramModels(N)
    else:
        ngm = NGramModel(N=N, smoothing_type=smoothing_type)

    ngm.train(tokenized_sentences)
    input_sentence = str(input('input sentence: '))
    print('score: ', ngm.calculate_probability_of_sentence(sentence=word_tokenizer(input_sentence)[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int)
    parser.add_argument('lm_type', type=str, choices=['n', 'l', 'g', 'i'])
    parser.add_argument('corpus_path', type=str)
    args = parser.parse_args()
    main(args.N, args.lm_type, args.corpus_path)
