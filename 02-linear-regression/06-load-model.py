from sklearn.externals import joblib



modeltest = joblib.load('linear.pkl')
print 'Predicted', modeltest.predict(12)
