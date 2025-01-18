# Language Modelling and Smoothing

## Pre-requisites

1. python
2. A python package manager such as `pip` or `conda`.
3. (OPTIONAL) virtualenv to create a virtual environment.
4. All the python libraries mentioned in `requirements.txt`.

## Tokenization

Using `spaCy` to first tokenize the given string into sentences, and then to obtain each word of that sentence as a
token. This is done in order to conform to the given output format (as mentioned in the Submission Format). Hence,
punctuations are included in the tokenized text.

---

## Smoothing and Interpolation

### Instructions to run

```python3 language_model.py <N> <lm_type> <corpus_path>```  
Here, `<lm_type>` must be one of `l` for Laplace Smoothing, `g` for Good-Turing Smoothing, `i` for Linear Interpolation. If none of these are chosen and something else entirely is given, then it defaults to no smoothing.  
`<N>` is the N-gram size, and `<corpus_path>` is the path to the corpus.

---

## Generation

### Instructions to run

```python3 generator.py <N> <lm_type> <corpus_path> <k>```  
Here, `<lm_type>` must be one of `l` for Laplace Smoothing, `g` for Good-Turing Smoothing, `i` for Linear Interpolation. If none of these are chosen and something else entirely is given, then it defaults to no smoothing.  
`<N>` is the N-gram size, `<corpus_path>` is the path to the corpus, and `<k>` is the number of candidates for the next word that are supposed to be printed.

---

