
from sklearn.tree import DecisionTreeClassifier

X = [
    # plays fetch | is grumpy | favorite food
    ['yes', 'no', 'bacon'],
    ['no', 'yes', 'dog food'],
    ['no', 'yes', 'cat food'],
    ['no', 'yes', 'bacon'],
    ['no', 'no', 'cat food'],
    ['no', 'yes', 'bacon'],
    ['no', 'yes', 'cat food'],
    ['no', 'no', 'dog food'],
    ['no', 'yes', 'cat food'],
    ['yes', 'no', 'dog food'],
    ['yes', 'no', 'bacon'],
    ['no', 'no', 'cat food'],
    ['yes', 'yes', 'cat food'],
    ['yes', 'yes', 'bacon']
]

y = [
    'dog', 'dog', 'cat', 'cat', 'cat', 'cat', 'cat',
    'dog', 'cat', 'dog', 'dog', 'cat', 'cat', 'dog'
]

X_train = []
for x in X:
    plays_fetch = 1 if x[0] == 'yes' else 0
    is_grumpy = 1 if x[1] == 'yes' else 0
    food = 0
    if x[2] == 'bacon':
        food = 0
    elif x[2] == 'dog food':
        food = 1
    else:
        food = 2
    X_train.append([plays_fetch, is_grumpy, food])

y_train = [1 if y_small == 'dog' else 0 for y_small in y]

model =  DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)

def pet(b):
    return 'dog' if b[0] == 1 else 'cat'

# play fetch, is grumpy, 0=bacon,1=dogfood,2=catfood
print pet(model.predict([[0, 1, 2]]))
print pet(model.predict([[0, 0, 2]]))
print pet(model.predict([[0, 1, 0]]))
print pet(model.predict([[0, 1, 1]]))
print pet(model.predict([10, 0, 0]))
