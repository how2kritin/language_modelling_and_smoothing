# Language Modelling and Smoothing

## Resources

1. [Stanford NLP - Language Models lecture](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1086/handouts/cs224n-lecture2-language-models-slides.pdf)

## Pre-requisites

1. python
2. A python package manager such as `pip` or `conda`.
3. (OPTIONAL) virtualenv to create a virtual environment.
4. All the python libraries mentioned in `requirements.txt`.

## Tokenization

Using `spaCy` to first tokenize the given string into sentences, and then to obtain each word of that sentence as a
token. This is done in order to conform to the given output format (as mentioned in the Submission Format). Hence,
punctuations are included in the tokenized text.

I have increased the max `spaCy` tokenization length to 2000000 characters. Feel free to change this if required, but be
warned that it uses more RAM if you do so. In order to reduce the amount of RAM being used, I have disabled the `ner` (
Named Entity Recognition) and `parser` modules of `spaCy`. Feel free to enable this if you need them.

---

## Smoothing and Interpolation

### Instructions to run

```python3 language_model.py <N> <lm_type> <corpus_path>```  
Here, `<lm_type>` must be one of `l` for Laplace Smoothing, `g` for Good-Turing Smoothing, `i` for Linear Interpolation.
If none of these are chosen and something else entirely is given, then it defaults to no smoothing.  
`<N>` is the N-gram size, and `<corpus_path>` is the path to the corpus.

### Laplace Smoothing

$$P(w|h) = \frac{c(w,h) + 1}{c(h) + V}$$ where V is the total vocabulary size (assumed known).

Essentially, we pretend that we saw every word once more than we actually did.

### Good-Turing Smoothing

$$P_{GT}(w_1...w_n) = \frac{r^*}{N}$$
where:
$$r^* = \frac{(r + 1)S(N_{r + 1})}{S(N_r)}$$
Here, $S(\cdot)$ is the smoothed function. For small values of $r$, $S(N_r) = N_r$ is a reasonable assumption (no
smoothing is performed). However, for larger values of $r$, values of $S(N_r)$ are read off the regression line given by
the logarithmic relationship $$log(N_r) = a + blog(r)$$ where $N_r$ is the number of times $n$-grams of frequency $r$ have
occurred.

For unseen events:
$$P_{GT}(w_1...w_n) = \frac{N_1}{N}$$

### Linear Interpolation


---

## Generation

### Instructions to run

```python3 generator.py <N> <lm_type> <corpus_path> <k>```  
Here, `<lm_type>` must be one of `l` for Laplace Smoothing, `g` for Good-Turing Smoothing, `i` for Linear Interpolation.
If none of these are chosen and something else entirely is given, then it defaults to no smoothing.  
`<N>` is the N-gram size, `<corpus_path>` is the path to the corpus, and `<k>` is the number of candidates for the next
word that are supposed to be printed.

---

