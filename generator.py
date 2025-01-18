from language_model import NGramModel
import argparse

def main(N: int, lm_type: str, corpus_path: str, k: int) -> None:
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

    predicted_words_dict = ngm.predict_next_word(sentence=input_sentence, n_next_words=k)
    if len(predicted_words_dict) == 0:
        print("Could not predict any possible candidates for the next word due to a small corpus.")
        return

    print("output:")
    for key, val in predicted_words_dict.items():
        print(key, val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int)
    parser.add_argument('lm_type', type=str)
    parser.add_argument('corpus_path', type=str)
    parser.add_argument('k', type=int)
    args = parser.parse_args()
    main(args.N, args.lm_type, args.corpus_path, args.k)