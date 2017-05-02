import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt


HAM = 0
SPAM = 1
# Validate the number of spams and hams (175 each)
dfValid = pd.read_csv('./data/Youtube01-Psy.csv', delimiter=',')

print 'Number of spam messages:', dfValid[dfValid['CLASS'] == SPAM]['CLASS'].count()
print 'Number of ham messages:', dfValid[dfValid['CLASS'] == HAM]['CLASS'].count()

df = pd.read_csv('./data/Youtube01-Psy.csv', delimiter=',')

X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['CONTENT'], df['CLASS'])
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)


classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)


# for i, prediction in enumerate(predictions):
#    print 'Prediction: %s. Message: %s' % ('ham' if prediction == HAM else 'spam', X_test_raw[i]) 

# Accuracy measures fraction of the classifier's prediction that are correct
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print 'Accuracy', np.mean(scores), scores

# Precision
precisions = cross_val_score(classifier, X_train, y_train, cv=5, scoring='precision')
print 'Precision', np.mean(precisions), precisions

recalls = cross_val_score(classifier, X_train, y_train, cv=5, scoring='recall')
print 'Recall', np.mean(recalls), recalls


f1s = cross_val_score(classifier, X_train, y_train, cv=5, scoring='f1')
print 'F1', np.mean(f1s), f1s


predictions = classifier.predict_proba(X_test)
false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:,1])
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %.2f' % roc_auc)
plt.legend(loc='lower_right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()


X_real = [
    'I like this video',
    'Huh, anyway check out this you[tube] channel',
    'This music good',
    'Buy drugs', 
    'You should check my channel for Funny VIDEOS',
    'F***',
    'I love PSY',
    'His music is damn good',
    'Subscribe to me',
    'Admit it you just came here to check the number',
    'Science is good for science',
    'Go to this channel'
]
X_predict = vectorizer.transform(X_real)
real_predictions = classifier.predict(X_predict)
print '\n'
for i, prediction in enumerate(real_predictions):
    result = 'ham' if prediction == HAM else 'spam'
    print 'Result is %s for: %s' % (result, X_real[i])

