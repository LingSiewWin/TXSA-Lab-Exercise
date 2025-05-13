import nltk
import collections
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews


def word_feats(words):
    return dict([(word, True) for word in words])


negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

negcutoff = int(len(negfeats) * 3 / 4)
poscutoff = int(len(posfeats) * 3 / 4)

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(testfeats):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)

print("\nThe accuracy is: ", nltk.classify.accuracy(classifier, testfeats) * 100, "%")

print ('pos precision:', nltk.scores.precision(refsets['pos'], testsets['pos']))
print ('pos recall:', nltk.scores.recall(refsets['pos'], testsets['pos']))
print ('pos F-measure:', nltk.scores.f_measure(refsets['pos'], testsets['pos']))
print ('neg precision:', nltk.scores.precision(refsets['neg'], testsets['neg']))
print ('neg recall:', nltk.scores.recall(refsets['neg'], testsets['neg']))
print ('neg F-measure:', nltk.scores.f_measure(refsets['neg'], testsets['neg']))
