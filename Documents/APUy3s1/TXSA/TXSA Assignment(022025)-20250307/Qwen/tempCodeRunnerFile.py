import nltk
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from nltk.tag import RegexpTagger
from tabulate import tabulate

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

with open('Q3/Data_2.txt', 'r') as file:
    sentence = file.read().strip()

# Tokenize once for consistency
tokens = word_tokenize(sentence)  # This includes the period if present in the file

# 1. NLTK POS Tagger
nltk_tags = nltk.pos_tag(tokens)

# 2. TextBlob POS Tagger (force tokenization to match NLTK)
blob = TextBlob(sentence)
textblob_tags = blob.tags

# 3. Regular Expression Tagger with tailored patterns
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

# Prepare data for tabular output
table_data = []
for i, token in enumerate(tokens):
    nltk_tag = nltk_tags[i][1]
    textblob_tag = textblob_tags[i][1] if i < len(textblob_tags) else 'N/A'  # Handle length mismatch
    regexp_tag = regexp_tags[i][1]
    table_data.append([token, nltk_tag, textblob_tag, regexp_tag])

# Display in a clean table
headers = ["Word", "NLTK Tag", "TextBlob Tag", "Regexp Tag"]
print("POS Tagging Results:")
print(tabulate(table_data, headers=headers, tablefmt="grid"))



