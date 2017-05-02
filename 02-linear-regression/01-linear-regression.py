from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib

from sklearn.linear_model import LinearRegression
# Diameters in inches
X = [[6], [8], [10], [14], [18]]
# Price in dollars
y = [7, 9, 13, 17.5, 18]



# Create and fit the model
model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'linear.pkl')

# modeltest = joblib.load('linear.pkl')
# print 'Predicted', modeltest.predict(12)

print 'A 12" pizza should cost: $%.2f' % model.predict(12)

plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X, y, 'k.')

extendedX = X + [[0], [25]]
plt.plot(extendedX, model.predict(extendedX), color='blue', linewidth=1)
plt.axis([0, 25, 0, 25])
plt.grid(True)



# Calculate the model's fitness using Residual Sum of Squares
print 'Residual sum of squares: %.2f' % np.mean((model.predict(X) - y) ** 2)

# Variance is a measure of how far a set of values is spread out
variance = np.var(X, ddof=1)
print 'Variance is: %.2f' % variance

# Covariance is a measure of how two variables change together
flatX = [x[0] for x in X]
covariance = np.cov(flatX, y)[0][1]
print 'Covariance is: %.2f' % covariance


# Beta is the cov(x,y)/var(x)
beta = covariance / variance
print 'Beta is: %.2f' % beta

# equation y_mean = a + b * x_mean
x_mean = np.mean(X)
y_mean = np.mean(y)

# Solve for a = y_mean - b * x_mean
a = y_mean - beta * x_mean
print 'A is: %.2f' % a

# R-squared measures how well the observed values of the response variables
# are predicted by the model
X_test = [[8], [9], [11], [16], [12]]
y_test = [[11], [8.5], [15], [18], [11]]

print 'R-squared: %.4f' % model.score(X_test, y_test)

plt.show()
