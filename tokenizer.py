from spacy.lang.en import English


def word_tokenizer(inp: str) -> list[list[str]]:
    """
    Takes a string as input and returns the word tokens of the string in a list of lists, where each list contains the tokenized words of that sentence.
    Includes punctuations.
    :param inp:
    :return:
    """
    nlp = English()
    nlp.add_pipe("sentencizer")
    tokenized_text = nlp(inp)

    return [[token.text for token in sent] for sent in tokenized_text.sents]


def main():
    inp_sentence = str(input("your text: "))
    token_list = word_tokenizer(inp_sentence)
    print("tokenized text: ", token_list)


if __name__ == "__main__":
    main()
