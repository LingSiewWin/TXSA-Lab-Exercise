import random
from collections import defaultdict

# Sample corpus (like training data)
corpus = [
    "the cat sat on the mat",
    "the dog ate the food",
    "a cat likes to sleep",
    "a dog likes to play",
    "humans and cats live together",
    "humans love their pets"
]

# Build bi-gram model
bigram_model = defaultdict(list)

for sentence in corpus:
    words = sentence.split()
    for w1, w2 in zip(words[:-1], words[1:]):
        bigram_model[w1].append(w2)

# Generate sentence
start_word = input("Please enter a phrase: ").strip().lower()
length = int(input("Enter the sentence length: "))

current_word = start_word
result = [current_word]

for _ in range(length - 1):
    next_words = bigram_model.get(current_word, None)
    if not next_words:
        break
    current_word = random.choice(next_words)
    result.append(current_word)

generated_sentence = ' '.join(result)
print("\nGenerated sentence:")
print(generated_sentence)