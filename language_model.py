# Ref: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1086/handouts/cs224n-lecture2-language-models-slides.pdf

from typing import Literal
from collections import defaultdict, Counter
from tokenizer import word_tokenizer

class NGramModel:
    def __init__(self, N: int, smoothing_type: Literal['none', 'laplace', 'good-turing', 'linear_interpolation'] = "None"):
        """
        :param N: This is the size of each 'N'-gram to be generated.
        :param smoothing_type: Type of smoothing to be performed while training the language model. 'none' by default. Choose one from {'none', 'laplace', 'good-turing', 'linear_interpolation'}.
        """
        self.n = N
        self.smoothing_type = smoothing_type
        self.ngrams = defaultdict(int)
        self.context_counts = defaultdict(int)

    def train(self, inp_str: str) -> None:
        """
        Train the N-gram model.
        :param inp_str: Corpus of data on which the model will be trained.
        :return:
        """
        tokenized_sentences = word_tokenizer(inp_str)
        for sentence in tokenized_sentences:
            # This <s> padding n-1 times at the start is so that the start of the sentence has enough context to calculate probabilities.
            tokens = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                context = tuple(tokens[i:i + self.n - 1])
                self.ngrams[ngram] += 1
                self.context_counts[context] += 1

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
        for ngram, count in self.ngrams.items():
            if ngram[:-1] == context:
                predictions[ngram[-1]] = count / self.context_counts[context]

        # Return top n_next_words after sorting them by decreasing order of probabilities.
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
            print(ngram)
            ngram_count = self.ngrams.get(ngram, 0)
            context_count = self.context_counts.get(context, 0)

            if context_count == 0:
                return 0.0  # If the context has never appeared, then the probability is trivially 0.

            probability *= ngram_count / context_count

        return probability



def main():
    ngm = NGramModel(N=2)
    text = """The quick brown fox jumps over the lazy dog. 
              The fox is quick and brown. 
              The dog is lazy but happy."""
    ngm.train(text)
    print(ngm.predict_next_word(sentence="The", n_next_words=3))
    print(ngm.calculate_probability_of_sentence(sentence="The dog is lazy dog."))


if __name__ == "__main__":
    main()