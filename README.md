# Language Modelling and Smoothing

# Resources

1. [Stanford NLP - Language Models lecture](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1086/handouts/cs224n-lecture2-language-models-slides.pdf)
2. [Simple Good-Turing Smoothing](https://www.d.umn.edu/~tpederse/Courses/CS8761-FALL02/Code/sgt-gale.pdf)
3. [Linear Interpolation - TnT -- A Statistical Part-of-Speech Tagger](https://aclanthology.org/A00-1031.pdf)

# Pre-requisites

1. python
2. A python package manager such as `pip` or `conda`.
3. (OPTIONAL) virtualenv to create a virtual environment.
4. All the python libraries mentioned in `requirements.txt`.

# Tokenization

Using `spaCy` to first tokenize the given string into sentences, and then to obtain each word of that sentence as a
token. This is done in order to conform to the given output format (as mentioned in the Submission Format). Hence,
punctuations are included in the tokenized text.

I have increased the max `spaCy` tokenization length to 2000000 characters. Feel free to change this if required, but be
warned that it uses more RAM if you do so. In order to reduce the amount of RAM being used, I have disabled the `ner` (
Named Entity Recognition) and `parser` modules of `spaCy`. Feel free to enable this if you need them.

---

# Smoothing and Interpolation

## Instructions to run

```python3 language_model.py <N> <lm_type> <corpus_path>```  
Here, `<lm_type>` must be one of `l` for Laplace Smoothing, `g` for Good-Turing Smoothing, `i` for Linear Interpolation.
If none of these are chosen and something else entirely is given, then it defaults to no smoothing.  
`<N>` is the N-gram size, and `<corpus_path>` is the path to the corpus.

## Different types of smoothing used, and their logic

### Laplace Smoothing

$$P(w|h) = \frac{c(w,h) + 1}{c(h) + V}$$

where $V$ is the total vocabulary size (assumed known).

Essentially, we pretend that we saw every word once more than we actually did. Hence, it is also called "Add-One"
smoothing.

### Good-Turing Smoothing

$$P_{GT}(w_1...w_n) = \frac{r^*}{N}$$

where:

$$r^* = \frac{(r + 1)S(N_{r + 1})}{S(N_r)}$$

Here, $S(\cdot)$ is the smoothed function. For small values of $r$, $S(N_r) = N_r$ is a reasonable assumption (no
smoothing is performed). However, for larger values of $r$, values of $S(N_r)$ are read off the regression line given by
the logarithmic relationship

$$log(N_r) = a + blog(r)$$

where $N_r$ is the number of times $n$-grams of frequency $r$
have occurred.

However, this plot of $log(N_r)$ versus $log(r)$ is problematic because for large $r$, many $N_r$ will be zero. Instead,
we plot a revised quantity, $log(Z_r)$ versus $log(r)$, where $Z_r$ is defined as

$$Z_r = \frac{N_r}{\frac{1}{2}(t - q)}$$

and where $q$, $r$ and $t$ are three consecutive subscripts with non-zero counts $N_q$, $N_r$, $N_t$. For the special
case where $r$ is 1, we take $q = 0$. In the opposite special case, when $r = r_{last}$ is the index of the _last_
non-zero count, replace the divisor $\frac{1}{2}(t-q)$ with $r_{last}-q$.

**For unseen events:**

$$P_{GT}(w_1...w_n) = \frac{N_1}{N}$$

Here, we are using the _Turing_ estimate for small $r$ values, and the _Good-Turing_ estimate for large $r$ values.
Since we are combining two different estimates of probabilities, we do not expect
them to add to one. In this condition, our estimates are called _unnormalized_. We make sure that the
probability estimates add to one by dividing by the total of the _unnormalized_ estimates. This is called
_renormalization_.

$$P_{SGT} = (1 - \frac{N_1}{N}) \frac{P^{unnorm}_r}{\sum P^{unnorm}_r} \hspace{5mm}r \geq 1$$

This renormalized estimate is the _Simple Good-Turing_ (SGT) smoothing estimate. This is what we will be using here.

### Linear Interpolation

When performing this for a trigram, we estimate the trigram probabilities as follows:

$$P(t_3|t_1, t_2) = \lambda_1\hat{P}(t_3) + \lambda_2\hat{P}(t_3|t_2) + \lambda_3\hat{P}(t_3|t_1, t_2)$$

where $\hat{P}$ are the maximum likelihood estimates of the probabilities and $\lambda_1 + \lambda_2 + \lambda_3 = 1$
so $P$ again represent probability distributions.

The following is the algorithm to calculate the weights for context-independent linear interpolation λ₁, λ₂, λ₃ when
the n-gram frequencies are known. N is the size of the corpus. If the denominator in one of the expressions
is 0, we define the result of that expression to be 0. _**[Page 3 of ref [3](https://aclanthology.org/A00-1031.pdf)]**_

```
Set λ₁ = λ₂ = λ₃ = 0

For each trigram t₁, t₂, t₃ with f(t₁, t₂, t₃) > 0:
    Depending on the maximum of the following three values:
    
    Case f(t₁, t₂, t₃) - 1 / f(t₁, t₂) - 1:
        Increment λ₃ by f(t₁, t₂, t₃)
    
    Case f(t₂, t₃) - 1 / f(t₂) - 1:
        Increment λ₂ by f(t₁, t₂, t₃)
    
    Case f(t₃) - 1 / N - 1:
        Increment λ₁ by f(t₁, t₂, t₃)
        
Normalize λ₁, λ₂, λ₃
```

This idea can be easily extrapolated to any $n$-gram, by simply considering the n-gram probability distribution as being
dependent on all the previous $i$-gram's Maximum Likelihood Estimates (MLEs) (where $1 \leq i \leq n$).

---

# Generation

## Instructions to run

```python3 generator.py <N> <lm_type> <corpus_path> <k>```  
Here, `<lm_type>` must be one of `l` for Laplace Smoothing, `g` for Good-Turing Smoothing, `i` for Linear Interpolation.
If none of these are chosen and something else entirely is given, then it defaults to no smoothing.  
`<N>` is the N-gram size, `<corpus_path>` is the path to the corpus, and `<k>` is the number of candidates for the next
word that are supposed to be printed.

---

