# Essentials of Large Language Models

## Definition

* LM can be defined as probabilistic model that assigns probabilities to sequence of words or token in a given language.
* Goal is to capture the patterns and structure of the language and predict the likelihood of a particular sequence of words.
* Let's assume we have a vocabulary ```V``` that contains a sequence of words or tokens denoted as ```w1, w2,...wn```, where ```n``` is the length of the sequence.
* The LM assign ```(p)``` to every possible sequence or order of word belonging to vocabulary```(V)```.
* The probability of the entire dsequence can be expressed as follows:
    * ```p(w1, w2,...wn)```

## Example

* Assume we have ```V= {chase, the, cat, the, mouse}```, and the following probabilities ```(p)``` are assigned:
    * ```p{chase, the, cat, the, mouse}``` = 0.0001
    * ```p{the, chase, cat, the, mouse}``` = 0.003
    * ```p{chase, the, mouse, the, cat}``` = 0.0021
    * ```p{the, cat, chase, the, mouse}``` = 0.02
    * ```p{the, mouse, chase, the, cat}``` = 0.01

> **_NOTE:_**  LMs must have external knowledge for them to be able to assign meanigful probabilities; therefore, they are trained. During this training process, the model learns to assign higher probabilities to words more likely to follow a given context. After training, the LM can generate text by sampling words based on these learned probabilities.

## Prediction

* We can also predict a word in a given sequence.
* An LM estimates this probability by considering the conditional probabilities of each word given the previous words in the sequence.
* Using the chain rule of probability, the joint probability of the sequence can be decomposed as:
* ```p(w1, w2,...,wn) = p(w1) * p(w2|w1) * p(w3|w1,w2)...p(wn|w1,w2,...,wn-1)```
* For example:
  * ```p(the, cat, chase, the, mouse) = p(the). p(cat|the). p(chase|the, cat). p(the|the, cat, chase) .p(mouse|the, cat, chase, the)```

## N-gram language model

* N-hram models are a type of probabilistic LM used in NLP and computational linguistics.
* These models are based on the idea that the probability of a word depends on the previous ```n - 1``` words in the sequence. The term ```n-grams``` refers to a consecutive sequence of n items.
* For Example: consider the following sentence: ```I live language models```
  * Unigram (1-gram): ``` "I", "love", "language", "models"```
  * Bigram (2-gram): ``` "I love", "love language", "language models"```
  * Trigram (3-gram): ``` "I love language", "love language models"```
  * 4-gram: ``` "I love language models"```
 
> **_NOTE:_** More advanced LMs, sunh as recurrent NN, have been replaced by LLMs.

## Algorithm

* **Tokenization**: Split the input text into individual words or tokens.
* **N-gram generation**: Create n-grams by forming sequence of ```n``` consecutive words from the tokenized text.
* **Frequency counting**: Count the occurences of each n-gram in the training corpus.
* **Probability estimation**: Calculate the conditional probability of each wor given to its previous ```n - 1``` words using the frequency count.
* **Smoothing**: To handle unseen n-grams and avoid zero probabilities.
* **Text generation**: Start with an initial seed of ```n - 1``` words, predict the next wor based on probabilities, and iteratively generate the next words to form a sequence.
* **Repeat generation**: Continue generating words until the desired length or a stopping condition is reached

## Example:
```python
import random

class NGramLanguageModel:
    def __init__(self, n):
        self.n = n
        self.ngrams = {}
        self.start_tokens = ['<start>'] * (n - 1)

    def train(self, corpus):
        for sentence in corpus:
            tokens = self.start_tokens + sentence.split() + ['<end>']
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                if ngram in self.ngrams:
                    self.ngrams[ngram] += 1
                else:
                    self.ngrams[ngram] = 1

    def generate_text(self, seed_text, length=10):
        seed_tokens = seed_text.split()
        padded_seed_text = self.start_tokens[-(self.n - 1 - len(seed_tokens)):] + seed_tokens
        generated_text = list(padded_seed_text)
        current_ngram = tuple(generated_text[-self.n + 1:])

        for _ in range(length):
            next_words = [ngram[-1] for ngram in self.ngrams.keys() if ngram[:-1] == current_ngram]
            if next_words:
                next_word = random.choice(next_words)
                generated_text.append(next_word)
                current_ngram = tuple(generated_text[-self.n + 1:])
            else:
                break

        return ' '.join(generated_text[len(self.start_tokens):])

# Toy corpus
toy_corpus = [
    "This is a simple example.",
    "The example demonstrates an n-gram language model.",
    "N-grams are used in natural language processing.",
    "This is a toy corpus for language modeling."
]

n = 3 # Change n-gram order here

# Example usage with seed text
model = NGramLanguageModel(n)  
model.train(toy_corpus)

seed_text = "This"  # Change seed text here
generated_text = model.generate_text(seed_text, length=3)
print("Seed text:", seed_text)
print("Generated text:", generated_text)

```

## Large Language Models

* Refers to advance NLP models trained on massive amounts of textual data. These models are designed to understand and generate human-like text based on the input they receive.


## LLMs vs LMs

| Aspect | LLMs | LMs |
| ------ | ---- | --- |
| Scale and Parameters | tens to hundreds of billions of parameters | Millions of parameters |
| Training Data | Trained on vast and diverse datasets from the internet | Can Can be trained on smaller, domain-specific datasets |
| Versatility | Higly versatile, excelling across various NLP tasks | Task specific, might require more fine-tuning |
| Computational Resources | Demands significant computational power and specialized hardware | More computational efficient, accessible on standard hardware |
| Use Cases | Complex language understanding, translation, summarization, creative writing | Specific tasks like sentiment analysis and named entity recognition |

































  
