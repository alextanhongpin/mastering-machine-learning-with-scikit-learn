import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

df = pd.read_csv('data/winequality-red.csv', sep=';')
print df.describe()


# plt.scatter(df['alcohol'], df['quality'])
# plt.xlabel('Alcohol')
# plt.ylabel('Quality')
# plt.title('Alcohol against Quality')
# plt.show()

X = df[list(df.columns)[:-1]]
y = df['quality']
# We use cross validation instead
# X_train, X_test, y_train, y_test = train_test_split(X, y)

regressor = LinearRegression()
# regressor.fit(X_train, y_train)
# y_predictions = regressor.predict(X_test)
# print 'R-squared: ', regressor.score(X_test, y_test)

scores = cross_val_score(regressor, X, y, cv=5)
print scores.mean(), scores

