import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    df = pd.read_csv('./data/ad.data', header=None)
    explanatory_variable_columns = set(df.columns.values)
    response_variable_column = df[len(df.columns.values) - 1]

    # The last column describes the target
    explanatory_variable_columns.remove(len(df.columns.values) - 1)

    y = [1 if e == 'ad.' else 0 for e in response_variable_column]
    X = df[list(explanatory_variable_columns)]

    X.replace(to_replace=' *\?', value=-1, regex=True, inplace=True)
    # Split the traning and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipeline = Pipeline([
        # ('clf', DecisionTreeClassifier(criterion='entropy'))
        ('clf', RandomForestClassifier(criterion='entropy'))
    ])

    parameters = {
        'clf__n_estimators': (5, 10, 20, 50),
        'clf__max_depth': (150, 155, 160),
        'clf__min_samples_split': (2, 3),
        'clf__min_samples_leaf': (2, 3)
    }

    # Set GridSearchCV to maximize the model's F1 score
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='f1')
    grid_search.fit(X_train, y_train)
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameter set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.key()):
        print '\t%s:%r' % (param_name, best_parameters[param_name])
    predictions = grid_search.predict(X_test)

    print classification_report(y_test, predictions)