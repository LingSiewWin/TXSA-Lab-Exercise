import nltk
from nltk import CFG
from nltk.parse import ChartParser

# Read sentence from file
with open('Q3/Data_2.txt', 'r') as file:
    sentence = file.read().strip()

# Define grammar
grammar = CFG.fromstring("""
    S -> NP VP CC VP | NP VP
    NP -> DT JJ JJ NN | DT JJ NN | DT NN
    VP -> VBD PP | VBD RB
    PP -> IN NP
    DT -> 'The' | 'the'
    JJ -> 'big' | 'black' | 'white'
    NN -> 'dog' | 'cat'
    VBD -> 'barked' | 'chased'
    IN -> 'at'
    CC -> 'and'
    RB -> 'away'
""")

# Tokenize for parsing (split since grammar uses exact words)
tokens = sentence.split()

# Parse and display trees
parser = ChartParser(grammar)
print("Possible Parse Trees:")
for i, tree in enumerate(parser.parse(tokens), 1):
    print(f"\nTree {i}:")
    print(tree)
    tree.pretty_print()