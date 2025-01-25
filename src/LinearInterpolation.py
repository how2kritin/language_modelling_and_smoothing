from NGramModel import NGramModel
import sys
import math
from _collections import defaultdict

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
                    unigram_count = self.ngram_models[0].ngrams.get(sub_ngram, 0)
                    if N > 1:
                        values.append((unigram_count - 1) / (N - 1))
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

    def calculate_probability(self, ngram: tuple[str, ...]) -> float:
        if self.lambdas is None:  # this should be done during training. however, we are doing it here as well, for the sake of completeness.
            self._calculate_linear_interpolation_weights()

        probability = 0.0

        # for each k from 1 to n
        for k in range(1, self.n + 1):
            sub_ngram = ngram[-k:]  # take the last k tokens
            sub_context = sub_ngram[:-1]  # take all but the last token

            prob = self.ngram_models[k - 1].calculate_probability(sub_ngram, sub_context)
            probability += self.lambdas[k - 1] * prob

        return probability

    def predict_next_word(self, tokenized_sentence: list[str], n_candidates_for_next_word: int) -> dict:
        """
        Given a sentence, predict n_next_words possible candidates for the next word, along with their probabilities.
        :param tokenized_sentence: Tokenized sentence (list of strings) for which next word is supposed to be predicted.
        :param n_candidates_for_next_word: Number of possible candidates for the next word, to predict.
        :return:
        """
        tokens = ['<s>'] * (self.n - 1) + tokenized_sentence
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
                    predictions[next_word] = self.calculate_probability(full_ngram)

        # return the top n_next_words after sorting them by decreasing order of probabilities.
        sorted_predictions = dict(
            sorted(predictions.items(), key=lambda item: item[1], reverse=True)[:n_candidates_for_next_word])
        return sorted_predictions

    def calculate_probability_of_sentence(self, tokenized_sentence: list[str]) -> float:
        """
        Given a sentence, calculate the probability of that sentence occurring.
        :param tokenized_sentence: Tokenized input sentence (list of str) to calculate probability for.
        :return: Probability of the sentence.
        """
        tokens = ['<s>'] * (self.n - 1) + tokenized_sentence + ['</s>']
        probability = 1.0

        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            prob = self.calculate_probability(ngram)
            probability *= prob

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

    def perplexity(self, tokenized_sentences: list[list[str]]) -> float:
        """
        Calculate the perplexity of the linearly interpolated language model on a given corpus.
        Perplexity = exp(-1/N * sum(log P(w_i|w_{i-n+1}...w_{i-1})))
        where N is the total number of words in corpus data and P is the interpolated probability.

        :param tokenized_sentences: List of tokenized sentences to calculate perplexity on
        :return:
        """
        log_prob_sum = 0.0
        total_words = 0

        for sentence in tokenized_sentences:
            # adding start and end tokens
            tokens = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            total_words += len(sentence) + 1  # +1 to count </s> token

            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                prob = self.calculate_probability(ngram)

                # handle zero probability scenario (as log(0) is not defined)
                if prob <= 0:
                    prob = sys.float_info.epsilon

                log_prob_sum += math.log2(prob)

        # calculate and return perplexity
        return math.pow(2, -1 * log_prob_sum / total_words)