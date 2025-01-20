import re
from nltk.tokenize import sent_tokenize
import nltk

def word_tokenizer(text: str) -> list[list[str]]:
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    # Regex patterns for various placeholders
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
    hashtag_pattern = re.compile(r'#[A-Za-z0-9_]+')
    time_pattern = re.compile(r'\b\d{1,2}:\d{2}\s?(?:[aApP]\.?[mM]\.?)?\b')
    # Capture a numeric value (with optional decimal) followed by '%'
    percent_pattern = re.compile(r'\b\d+(?:\.\d+)?%\b')
    # Age patterns like "25 year-old" or "25y" or "25yr"
    age_pattern = re.compile(r'\b\d+\s?(?:y|yr|yrs|-year-old|years-old|-years-old|year-old|years old|year old)\b')
    # Generic numeric pattern for integers or floats
    number_pattern = re.compile(r'\b\d+(?:\.\d+)?\b')

    # Substitutions with placeholders
    text = url_pattern.sub('<URL>', text)
    text = mention_pattern.sub('<MENTION>', text)
    text = hashtag_pattern.sub('<HASHTAG>', text)
    text = time_pattern.sub('<TIME>', text)
    # Important to replace the percent pattern before generic numbers
    text = percent_pattern.sub('<PERCENT>', text)
    text = age_pattern.sub('<AGE>', text)
    text = number_pattern.sub('<NUMBER>', text)

    # Split into sentences via NLTK
    sentences = sent_tokenize(text)

    # This pattern handles:
    #   - Placeholders like <URL> or <MENTION> or <AGE>, etc.
    #   - Words with optional contractions (like "I'm", "we've").
    #   - Single punctuation characters (e.g. ",", "!", "?").
    token_pattern = re.compile(
        r'<URL>|<MENTION>|<HASHTAG>|<TIME>|<PERCENT>|<AGE>|<NUMBER>'
        r'|[A-Za-z]+(?:\'[A-Za-z]+)?|[^\w\s]'
    )

    tokenized_sentences = []
    for sentence in sentences:
        tokens = token_pattern.findall(sentence)

        # filter out empty strings or whitespace-only tokens
        tokens = [token for token in tokens if token.strip()]
        tokenized_sentences.append(tokens)

    return tokenized_sentences


def main() -> None:
    inp_sentence = str(input("your text: "))
    token_list = word_tokenizer(inp_sentence)
    print("tokenized text: ", token_list)


if __name__ == "__main__":
    main()
