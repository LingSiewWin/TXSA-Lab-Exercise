import re
from nltk.stem import PorterStemmer, LancasterStemmer

# Sample words
words = ["running", "jumps", "easily", "happily", "beautiful"]

# Regular Expression Stemmer
def regex_stem(word):
    return re.sub(r'(ing|ed|es|s)$', '', word)

regex_stems = [regex_stem(word) for word in words]

# Porter Stemmer
porter = PorterStemmer()
porter_stems = [porter.stem(word) for word in words]

# Lancaster Stemmer
lancaster = LancasterStemmer()
lancaster_stems = [lancaster.stem(word) for word in words]

# Output Results
print("Original Words:", words)
print("Regex Stems:", regex_stems)
print("Porter Stems:", porter_stems)
print("Lancaster Stems:", lancaster_stems)