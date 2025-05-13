# ~~~~~~~~~~~~~~~~~~~~~~~~~ Gender Identification ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def gender_features(word):
    return {'last_letter': word[-1]}
gender_features('Shrek')

from nltk.corpus import names
labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
[(name, 'female') for name in names.words('female.txt')])

len(labeled_names)

import random
random.shuffle(labeled_names)
len(labeled_names)


import nltk
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
len(train_set)
len(test_set)

classifier_NB_1 = nltk.NaiveBayesClassifier.train(train_set)
print("Neo is a", classifier_NB_1.classify(gender_features('Neo')))
print("Annie is a", classifier_NB_1.classify(gender_features('Annie')))
print("\nThe accuracy of NB_1 is equal to: ", nltk.classify.accuracy(classifier_NB_1, test_set))
classifier_NB_1.show_most_informative_features(20)
sorted(classifier_NB_1.labels())


from nltk.classify import apply_features
train_set_2 = apply_features(gender_features, labeled_names[500:])
test_set_2 = apply_features(gender_features, labeled_names[:500])
classifier_NB_2 = nltk.NaiveBayesClassifier.train(train_set_2)
print("Neo is a", classifier_NB_2.classify(gender_features('Neo')))
print("Annie is a", classifier_NB_2.classify(gender_features('Annie')))
print("\nThe accuracy of NB_2 is equal to: ", nltk.classify.accuracy(classifier_NB_2, test_set_2))
classifier_NB_2.show_most_informative_features(20)


# Using Sklearn package to split the dataset ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.model_selection import train_test_split
train, test = train_test_split(featuresets, test_size=0.2, random_state=123)
len(train)
len(test)

# Naive Bayes with NLTK
import nltk
classifier_NB = nltk.NaiveBayesClassifier.train(train)
print("Neo is a", classifier_NB.classify(gender_features('Neo')))
print("Annie is a", classifier_NB.classify(gender_features('Annie')))
print("\nThe accuracy of NB is equal to: ", nltk.classify.accuracy(classifier_NB, test))
classifier_NB.show_most_informative_features(20)

# Decision Tree with NLTK
classifier_DT = nltk.classify.DecisionTreeClassifier.train(train, entropy_cutoff=0, support_cutoff=0)
print("Neo is a", classifier_DT.classify(gender_features('Neo')))
print("Annie is a", classifier_DT.classify(gender_features('Annie')))
print("\nThe accuracy of DT is equal to: ", nltk.classify.accuracy(classifier_DT, test))
print(classifier_DT)

# SklearnClassifier with NLTK
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
classifier_SKLC = SklearnClassifier(BernoulliNB()).train(train)
print("Neo is a", classifier_SKLC.classify(gender_features('Neo')))
print("Annie is a", classifier_SKLC.classify(gender_features('Annie')))
print("\nThe accuracy of SKLC is equal to: ", nltk.classify.accuracy(classifier_SKLC, test))


# SVM with NLTK
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
classifier_SVC = SklearnClassifier(SVC(), sparse=False).train(train)
print("Neo is a", classifier_SVC.classify(gender_features('Neo')))
print("Annie is a", classifier_SVC.classify(gender_features('Annie')))
print("\nThe accuracy of SVC is equal to: ", nltk.classify.accuracy(classifier_SVC, test))


# MAXENT with NLTK
from nltk.classify import maxent
encoding = maxent.TypedMaxentFeatureEncoding.train(train, count_cutoff=3, alwayson_features=True)
classifier_MAX = maxent.MaxentClassifier.train(train, bernoulli=False, encoding=encoding, trace=0)
print("Neo is a", classifier_MAX.classify(gender_features('Neo')))
print("Annie is a", classifier_MAX.classify(gender_features('Annie')))
print("\nThe accuracy of MAXENT is equal to: ", nltk.classify.accuracy(classifier_MAX, test))
classifier_MAX.show_most_informative_features(20)


# Naive Bayes from textblob
from textblob.classifiers import NaiveBayesClassifier
classifier_TB = NaiveBayesClassifier(train)
print("Neo is a", classifier_TB.classify(gender_features('Neo')))
print("Annie is a", classifier_TB.classify(gender_features('Annie')))
print("\nThe accuracy of TB_NB is equal to: ", classifier_TB.accuracy(test))


# Overall Accuracies - Using Sklearn package to split the dataset
print("~~~~~~~~~~~~~~~~~~ ACCURACY VALUES ~~~~~~~~~~~~~~~~~~~~")
print("\nThe accuracy of NB  is equal to: ", nltk.classify.accuracy(classifier_NB, test))
print("\nThe accuracy of DT is equal to: ", nltk.classify.accuracy(classifier_DT, test))
print("\nThe accuracy of SKLC is equal to: ", nltk.classify.accuracy(classifier_SKLC, test))
print("\nThe accuracy of SVC is equal to: ", nltk.classify.accuracy(classifier_SVC, test))
print("\nThe accuracy of MAXENT is equal to: ", nltk.classify.accuracy(classifier_MAX, test))
print("\nThe accuracy of TB_NB is equal to: ", classifier_TB.accuracy(test))
