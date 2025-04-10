import nltk
from nltk import CFG
from nltk.parse import ChartParser
from nltk.tokenize import word_tokenize

with open('Q3/Data_2.txt', 'r') as file:
    sentence = file.read().strip()

# Tokenize with NLTK (separates punctuation)
tokens = word_tokenize(sentence)
print(f"Tokens: {tokens}")  # Debug to see tokenized input

# Define grammar with comments on separate lines
grammar = CFG.fromstring("""
    # Sentence with optional coordination and punctuation
    S -> NP VP CC VP PUNCT | NP VP PUNCT
    # Noun phrase
    NP -> DT JJ JJ NN | DT JJ NN | DT NN
    # Verb phrase
    VP -> VBD PP | VBD RB
    # Prepositional phrase
    PP -> IN NP
    # Determiners
    DT -> 'The' | 'the'
    # Adjectives
    JJ -> 'big' | 'black' | 'white'
    # Nouns
    NN -> 'dog' | 'cat'
    # Past tense verbs
    VBD -> 'barked' | 'chased'
    # Prepositions
    IN -> 'at'
    # Conjunctions
    CC -> 'and'
    # Adverbs
    RB -> 'away'
    # Punctuation
    PUNCT -> '.'
""")

# Parse and display trees
try:
    parser = ChartParser(grammar)
    print("Possible Parse Trees:")
    for i, tree in enumerate(parser.parse(tokens), 1):
        print(f"\nTree {i}:")
        print(tree)
        tree.pretty_print()
except ValueError as e:
    print(f"Error: {e}")
    print("The grammar doesn’t cover all tokens. Check the file content and grammar rules.")
    
    
    