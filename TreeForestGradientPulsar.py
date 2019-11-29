import numpy
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt
import matplotlib.lines as lines
import csv

###QUESTION 1###

#splits according to a given size of first array
def split(list, size):
    return list[:size], list[size:]

#read the file
reader = csv.reader(open("pulsar_stars.csv", "r"), delimiter=",")

#discard first line
next(reader)

# convert to floats from strings
x = list(reader)
for i in range(len(x)):
    for j in range(len(x[i])):
        x[i][j] = float(x[i][j])

#confirmation of 17898 lines
print("Length:",len(x))

#split off 20% for test
test, x = split(x,3580)

#split remaining into validation and training groups
val, train = split(x, 3580)

#confirmation of sizes
print("Test:", len(test), "Validation:", len(val), "Training:", len(train), "\n")

#create y values for train
trainY = []
for x in train:
    trainY.append(x[-1])
    x.pop(-1)

#create y values for validation
valY = []
for x in val:
    valY.append(x[-1])
    x.pop(-1)

#create y values for testing
testY = []
for x in test:
    testY.append(x[-1])
    x.pop(-1)


#created Decision Tree Classifier
dtc = DecisionTreeClassifier(criterion='gini', max_depth=8)

#fit on training data
dtc.fit(train, trainY)

#predictions on validation data
valPred = dtc.predict_proba(val)

#cross-entropy on validation data
valCE = sk.metrics.log_loss(valY, valPred)
print("Validation GINI Cross-Entropy:",valCE)

###Information Gain###

#created Decision Tree Classifier
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=8)

#fit on training data
dtc.fit(train, trainY)

#predictions on validation data
valPred = dtc.predict_proba(val)

#cross-entropy on validation data
valCE = sk.metrics.log_loss(valY, valPred)
print("Validation Information Gain Cross-Entropy:",valCE, '\n')

###TESTING USING IG###

#created Decision Tree Classifier
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=8)

#fit on training data
dtc.fit(train, trainY)
dtc.fit(val, valY)

#predictions on validation data (used later)
testTreePred = dtc.predict_proba(test)

###QUESTION 3###
sizes = [5, 10, 50, 100, 200]

for i in sizes:
    #create and fit the random forest
    rfc = RandomForestClassifier(criterion='gini', n_estimators=i,max_depth=8,max_features=4)
    rfc.fit(train, trainY)

    #predictions on validation data
    valPred = rfc.predict_proba(val)

    #cross-entropy on validation data
    valCE = sk.metrics.log_loss(valY, valPred)
    print(i, "Tree Cross-Entropy:",valCE)
print("")

#since 100 trees performed best, train with 100 trees
rfc = RandomForestClassifier(criterion='gini', n_estimators=100,max_depth=8,max_features=4)

rfc.fit(train,trainY)
rfc.fit(val,valY)

testForestPred = rfc.predict_proba(test)

###QUESTION 4###
sampleSizes = [.1, .25, .5, .75, 1]

for i in sampleSizes:
    gbc = GradientBoostingClassifier(max_depth=8,subsample=i)
    gbc.fit(train, trainY)

    #predictions on validation data
    valPred = gbc.predict_proba(val)

    #cross-entropy on validation data
    valCE = sk.metrics.log_loss(valY, valPred)
    print(i, "Subsample Cross-Entropy:",valCE)
print("")

#100% performs best so we train using that model
gbc = GradientBoostingClassifier(max_depth=8, subsample=1)

gbc.fit(train,trainY)
gbc.fit(val,valY)

testGradientPred = gbc.predict_proba(test)

###QUESTION 5###
p = 0
counter = 0
a = 0
c = 0
b = 0
d = 0
results = numpy.empty((3580))
pod = numpy.empty((3,1001))
pofd = numpy.empty((3,1001))
while p < 1.001:
    #tree prediction
    for i in range(len(testTreePred)):
        if testTreePred[i][1] < p:
            results[i] = 0
        else:
            results[i] = 1

        #calculate a and c values
        if testY[i] == 1:
            if results[i] == 1:
                a += 1
            else:
                c += 1
        else:
            #calculate b and d values
            if results[i] == 0:
                d += 1
            else:
                b += 1
    
    pod[0][counter] = float(a / (a+c))
    pofd[0][counter] = float(b / (b + d))

    a = 0
    b = 0
    c = 0
    d = 0

    #forest prediction
    for i in range(len(testForestPred)):
        if testForestPred[i][1] < p:
            results[i] = 0
        else:
            results[i] = 1

        #calculate a and c values
        if testY[i] == 1:
            if results[i] == 1:
                a += 1
            else:
                c += 1
        else:
            #calculate b and d values
            if results[i] == 0:
                d += 1
            else:
                b += 1
    
    pod[1][counter] = float(a / (a+c))
    pofd[1][counter] = float(b / (b + d))

    a = 0
    b = 0
    c = 0
    d = 0

    #gradient prediction
    for i in range(len(testGradientPred)):
        if testGradientPred[i][1] < p:
            results[i] = 0
        else:
            results[i] = 1

        #calculate a and c values
        if testY[i] == 1:
            if results[i] == 1:
                a += 1
            else:
                c += 1
        else:
            #calculate b and d values
            if results[i] == 0:
                d += 1
            else:
                b += 1
    
    pod[2][counter] = float(a / (a+c))
    pofd[2][counter] = float(b / (b + d))

    a = 0
    b = 0
    c = 0
    d = 0
    p += .001
    counter += 1

plt.plot(pofd[0],pod[0])
plt.plot([0,1],[0,1],'--',color='gray')
plt.title("ROC Curve (Tree)")
plt.ylabel('Probability of detection (POD)')
plt.xlabel('Probability of false detection (POFD)')
plt.show()

plt.plot(pofd[1],pod[1])
plt.plot([0,1],[0,1],'--',color='gray')
plt.title("ROC Curve (Forest)")
plt.ylabel('Probability of detection (POD)')
plt.xlabel('Probability of false detection (POFD)')
plt.show()

plt.plot(pofd[2],pod[2])
plt.plot([0,1],[0,1],'--',color='gray')
plt.title("ROC Curve (Gradient)")
plt.ylabel('Probability of detection (POD)')
plt.xlabel('Probability of false detection (POFD)')
plt.show()

#finding Peirce score:
best = 0
threshold = 0

for i in range(len(pod[0])):
    if pod[0][i] - pofd[0][i] > best:
        best = pod[0][i] - pofd[0][i]
        threshold = i * .001

print("Best Peirce Score for Tree:", best, "\nThreshold:", threshold)

best = 0
threshold = 0

for i in range(len(pod[1])):
    if pod[1][i] - pofd[1][i] > best:
        best = pod[1][i] - pofd[1][i]
        threshold = i * .001

print("\nBest Peirce Score for Forest:", best, "\nThreshold:", threshold)

best = 0
threshold = 0

for i in range(len(pod[2])):
    if pod[2][i] - pofd[2][i] > best:
        best = pod[2][i] - pofd[2][i]
        threshold = i * .001

print("\nBest Peirce Score for Gradient:", best, "\nThreshold:", threshold)
print('\n')

auc = 0.0
for i in range(len(pod[0]) - 1):
    auc -= (pod[0][i] * (pofd[0][i + 1] - pofd[0][i]))

print("Tree AUC:",auc)

auc = 0.0
for i in range(len(pod[1]) - 1):
    auc -= (pod[1][i] * (pofd[1][i + 1] - pofd[1][i]))

print("Forest AUC:",auc)

auc = 0.0
for i in range(len(pod[2]) - 1):
    auc -= (pod[2][i] * (pofd[1][i + 1] - pofd[1][i]))

print("Gradient AUC:",auc)
