# Refs: 
# 1. https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1086/handouts/cs224n-lecture2-language-models-slides.pdf
# 2. https://www.d.umn.edu/~tpederse/Courses/CS8761-FALL02/Code/sgt-gale.pdf
# 3. https://aclanthology.org/A00-1031.pdf
import os
import random
from tokenizer import word_tokenizer
import argparse
import re
from NGramModel import NGramModel
from LinearInterpolation import LinearInterpolationOfNGramModels

random_state = 42  # for train-test split reproducibility while obtaining perplexity
random.seed(random_state)


# function to detokenize a list of tokens
def detokenize(tokens):
    sentence = " ".join(tokens)
    # fixing spaces before punctuation
    sentence = re.sub(r"\s+([.,!?;:\"\')])", r"\1", sentence)
    # fixing spaces after opening quotes/brackets
    sentence = re.sub(r"([\"'(\[{])\s+", r"\1", sentence)
    return sentence


def calculate_and_save_perplexities(sentences: list[list[str]], ngm: NGramModel | LinearInterpolationOfNGramModels,
                                    output_file: str) -> None:
    total_perplexity = 0
    num_sentences = len(sentences)

    results = []
    for sentence in sentences:
        sentence_perplexity = ngm.perplexity(sentence)
        total_perplexity += sentence_perplexity
        results.append((detokenize(sentence), sentence_perplexity))

    avg_perplexity = total_perplexity / num_sentences

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{avg_perplexity}\n")
        for sentence_text, perplexity in results:
            f.write(f"{sentence_text}\t{perplexity}\n")


def main(N: int, lm_type: str, corpus_path: str, task: str) -> None:
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

    if task == 'pr':
        # take a sentence as input, and return probability of that sentence occurring
        ngm.train(tokenized_sentences)
        input_sentence = str(input('input sentence: '))
        print('score: ', ngm.calculate_probability_of_sentence(tokenized_sentence=word_tokenizer(input_sentence)[0]))
    if task == 'pe':
        base_name = os.path.basename(corpus_path)
        output_dir = "./perplexity_scores"
        output_base = f"2022101071_{N}_{smoothing_type}_{base_name}"
        train_file = os.path.join(output_dir, output_base + "_train.txt")
        test_file = os.path.join(output_dir, output_base + "_test.txt")

        # creating test set of 1000 randomly sampled sentences
        test_size = 1000
        all_indices = list(range(len(tokenized_sentences)))
        test_indices = set(random.sample(all_indices, test_size))

        # split into train and test sets
        train_sentences = [sent for idx, sent in enumerate(tokenized_sentences)
                           if idx not in test_indices]
        test_sentences = [sent for idx, sent in enumerate(tokenized_sentences)
                          if idx in test_indices]

        ngm.train(train_sentences)

        calculate_and_save_perplexities(train_sentences, ngm, train_file)
        calculate_and_save_perplexities(test_sentences, ngm, test_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int)
    parser.add_argument('lm_type', type=str, choices=['n', 'l', 'g', 'i'])
    parser.add_argument('corpus_path', type=str)
    parser.add_argument('task', type=str, choices=['pr', 'pe'])
    args = parser.parse_args()
    main(args.N, args.lm_type, args.corpus_path, args.task)
