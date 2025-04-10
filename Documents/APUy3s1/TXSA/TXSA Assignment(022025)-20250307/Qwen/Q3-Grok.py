import nltk
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from nltk.tag import RegexpTagger

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Read sentence from file
with open('Q3/Data_2.txt', 'r') as file:
    sentence = file.read().strip()

# Tokenize once for efficiency
tokens = word_tokenize(sentence)

# 1. NLTK POS Tagger
nltk_tags = nltk.pos_tag(tokens)

# 2. TextBlob POS Tagger
blob = TextBlob(sentence)
textblob_tags = blob.tags

# 3. Regular Expression Tagger with improved patterns
patterns = [
    (r'^(The|the|a|A|an|An)$', 'DT'),  # Determiners
    (r'^(big|black|white)$', 'JJ'),    # Common adjectives
    (r'.*ed$', 'VBD'),                  # Past tense verbs
    (r'^(and|or|but)$', 'CC'),         # Conjunctions
    (r'^(at|in|on|to)$', 'IN'),        # Prepositions
    (r'^(away)$', 'RB'),               # Specific adverb
    (r'.*', 'NN')                      # Default to noun
]
regexp_tagger = RegexpTagger(patterns)
regexp_tags = regexp_tagger.tag(tokens)

# Report outputs
print("NLTK POS Tags:")
print(nltk_tags)
print("\nTextBlob POS Tags:")
print(textblob_tags)
print("\nRegular Expression Tagger POS Tags:")
print(regexp_tags)