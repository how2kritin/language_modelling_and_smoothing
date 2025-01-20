from tokenizer import word_tokenizer
from language_model import NGramModel, LinearInterpolationOfNGramModels
import argparse
import re

# function to detokenize a list of tokens
def detokenize(tokens):
    sentence = " ".join(tokens)
    # fixing spaces before punctuation
    sentence = re.sub(r"\s+([.,!?;:\"\')])", r"\1", sentence)
    # fixing spaces after opening quotes/brackets
    sentence = re.sub(r"([\"'(\[\{])\s+", r"\1", sentence)
    return sentence



def main(N: int, lm_type: str, corpus_path: str, k: int, gen_type: str) -> None:
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

    if gen_type == 's':
        generated_list_of_words = ngm.generate_sentence_next_n_words(sentence=word_tokenizer(input_sentence)[0], n=k)
        # sentence = TreebankWordDetokenizer().detokenize(generated_list_of_words)
        print(detokenize(generated_list_of_words))
    elif gen_type == 'w':
        predicted_words_dict = ngm.predict_next_word(sentence=word_tokenizer(input_sentence)[0], n_candidates_for_next_word=k)
        if len(predicted_words_dict) == 0:
            print("Could not predict any possible candidates for the next word.")
            return

        print("output:")
        for key, val in predicted_words_dict.items():
            print(key, val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int)
    parser.add_argument('lm_type', type=str, choices=['n', 'l', 'g', 'i'])
    parser.add_argument('corpus_path', type=str)
    parser.add_argument('k', type=int)
    parser.add_argument('gen_type', type=str, choices=['w', 's'])
    args = parser.parse_args()
    main(args.N, args.lm_type, args.corpus_path, args.k, args.gen_type)
