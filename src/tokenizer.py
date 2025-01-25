# import re
# from nltk.tokenize import sent_tokenize
# import nltk
import spacy

def word_tokenizer(text: str) -> list[list[str]]:
    nlp = spacy.load("en_core_web_sm")
    text = text.strip()
    text = " ".join(text.split())
    doc = nlp(text)

    tokenized_sentences = []
    for sent in doc.sents:
        tokenized_sentences.append([token.text for token in sent])

    return tokenized_sentences


def main() -> None:
    inp_sentence = str(input("your text: "))
    token_list = word_tokenizer(inp_sentence)
    print("tokenized text: ", token_list)


if __name__ == "__main__":
    main()
