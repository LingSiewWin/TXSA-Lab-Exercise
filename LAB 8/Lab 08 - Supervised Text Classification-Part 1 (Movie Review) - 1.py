import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

len(documents)

for row in range(0, len(documents) - 1):
    print(documents[row][1])

random.shuffle(documents)
print(documents[1])

all_words = []
for w in movie_reviews.words():
    wl = w.lower()
    all_words.append(wl)

all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))

print(all_words["bad"])
print(all_words["good"])
print(all_words["excellent"])

word_features = list(all_words)[:2000]
print(word_features)

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featureset = [(find_features(rev), category) for (rev, category) in documents]
len(featureset)
featureset[2]

train_set = featureset[100:]
len(train_set)

test_set = featureset[:100]
len(test_set)

# Naive Bayes with NLTK
classifier_NB = nltk.NaiveBayesClassifier.train(train_set)
print("\nThe accuracy of NB is equal to: ", nltk.classify.accuracy(classifier_NB, test_set))
classifier_NB.show_most_informative_features(30)
sorted(classifier_NB.labels())

print("cunning feature is for", classifier_NB.classify(find_features('shower')))
print("unimaginative feature is for", classifier_NB.classify(find_features('unimaginative')))


# Decision Tree with NLTK
classifier_DT = nltk.classify.DecisionTreeClassifier.train(train_set, entropy_cutoff=0, support_cutoff=0)
print("\nThe accuracy of DT is equal to: ", nltk.classify.accuracy(classifier_DT, test_set))
print(classifier_DT)


# Sklearn Classifier with NLTK
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
classifier_SKLC = SklearnClassifier(BernoulliNB()).train(train_set)
print("\nThe accuracy of SKLC is equal to: ", nltk.classify.accuracy(classifier_SKLC, test_set))


# SVM with NLTK
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
classifier_SVC = SklearnClassifier(SVC(), sparse=False).train(train_set)
print("\nThe accuracy of SVC is equal to: ", nltk.classify.accuracy(classifier_SVC, test_set))


# MAXENT with NLTK
from nltk.classify import maxent
encoding = maxent.TypedMaxentFeatureEncoding.train(train_set, count_cutoff=3, alwayson_features=True)
classifier_MAX = maxent.MaxentClassifier.train(train_set, bernoulli=False, encoding=encoding, trace=0)
print("\nThe accuracy of MAXENT is equal to: ", nltk.classify.accuracy(classifier_MAX, test_set))
classifier_MAX.show_most_informative_features(20)


# Overall Accuracies - Using Movie Review dataset and NLTK package
print("~~~~~~~~~~~~~~~~~~ ACCURACY VALUES ~~~~~~~~~~~~~~~~~~~~")
print("\nThe accuracy of NB  is equal to: ", nltk.classify.accuracy(classifier_NB, test_set))
print("\nThe accuracy of DT is equal to: ", nltk.classify.accuracy(classifier_DT, test_set))
print("\nThe accuracy of SKLC is equal to: ", nltk.classify.accuracy(classifier_SKLC, test_set))
print("\nThe accuracy of SVC is equal to: ", nltk.classify.accuracy(classifier_SVC, test_set))
print("\nThe accuracy of MAXENT is equal to: ", nltk.classify.accuracy(classifier_MAX, test_set))

