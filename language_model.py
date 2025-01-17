from tokenizer import word_tokenizer

class NGramModel:
    def __init__(self, N: int):
        """
        :param N: This is the size of each 'N'-gram to be generated.
        """
        self.n = N
        