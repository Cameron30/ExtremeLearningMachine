import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import csv

# For comparison to other methods
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

#my ELM code
from ELM import ELMNetwork

#create new dictionary for later
test_maes_dictionary = dict()

#set plot style
plt.style.use('ggplot')
sns.set_context("talk")

##SPLIT ARRAY##
def split(list, size):
    return list[:size], list[size:]

##PREPARE DATA##
def prepareData():
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
    X_test, x = split(x,3580)

    #split remaining into validation and training groups
    X_val, X_train = split(x, 3580)

    #confirmation of sizes
    print("Test:", len(X_test), "Validation:", len(X_val), "Training:", len(X_train), "\n")

    #create y values for train
    y_train = []
    for x in X_train:
        y_train.append(x[-1])
        x.pop(-1)

    #create y values for validation
    y_val = []
    for x in X_val:
        y_val.append(x[-1])
        x.pop(-1)

    #create y values for testing
    y_test = []
    for x in X_test:
        y_test.append(x[-1])
        x.pop(-1)

    #convert all to numpy arrays for ease of use later
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    #finds largest value and divides so that everyting is guaranteed to be < 1 (granted that training has largest number)
    max_y_train = max(abs(y_train))
    y_train = y_train / max_y_train
    y_val = y_val / max_y_train
    y_test = y_test / max_y_train

    return (X_train, y_train, X_val, y_val, X_test, y_test)

def addVariance(X_train, X_val, X_test):
    #add variance to the data
    for x in X_train:
        x = np.append(x, x[1] * x[1])

    #add variance to the data
    for x in X_val:
        x = np.append(x, x[1] * x[1])

    #add variance to the data
    for x in X_test:
        x = np.append(x, x[1] * x[1])

    return (X_train, X_val, X_test)

##ROC CURVE##
def ROCCurve(neurons, myColor, legend, X_train, y_train, X_val, y_val, X_test, y_test):
    print ("Training with %s neurons..."%neurons)

    #fit to training data then predict
    ELM = ELMNetwork(neurons)
    ELM.fit(X_train, y_train)
    prediction = ELM.predict(X_train)
        
    #fit to validation data then predict
    ELM.fit(X_val, y_val)
    prediction = ELM.predict(X_val)

    #predict test data
    prediction = ELM.predict(X_test)

    #initialize values for ROC calculations
    p = 0
    counter = 0
    a = 0
    c = 0
    b = 0
    d = 0
    results = np.empty((3580))
    pod = np.empty((1001))
    pofd = np.empty((1001))

    #go through the intervals
    while p < 1.001:
        #tree prediction
        for i in range(len(prediction)):
            if prediction[i] > .999:
                prediction[i] = .999
            if prediction[i] < p:
                results[i] = 0
            else:
                results[i] = 1

            #calculate a and c values
            if y_test[i] == 1:
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
    
        #calculate pod, pofd
        pod[counter] = float(a / (a+c))
        pofd[counter] = float(b / (b + d))

        #increase interval
        p += .001
        counter += 1

        #reset for next interval
        a=0
        b=0
        c=0
        d=0
        
    plt.plot(pofd, pod,color=myColor)
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.title("ROC Curve of ELM")
    plt.legend(legend)
    plt.ylabel('Probability of detection (POD)')
    plt.xlabel('Probability of false detection (POFD)')
    #plt.show()

def compareToOthers():
    ## ELM TRAINING
    MAE_TRAIN_MINS = []
    MAE_VAL_MINS = []
    MAE_TEST_MINS = []

    for M in range(1, 150, 1):
        MAES_TRAIN = []
        MAES_VAL = []
        MAES_TEST = []
        print ("Training with %s neurons..."%M)

        #try each one 10 times for best trial
        for i in range(10):
            ELM = ELMNetwork(M)
            ELM.fit(X_train, y_train)
            prediction = ELM.predict(X_train)
            MAES_TRAIN.append(mean_absolute_error(y_train, prediction))
            
            ELM.fit(X_val, y_val)
            prediction = ELM.predict(X_val)
            MAES_VAL.append(mean_absolute_error(y_val, prediction))

            prediction = ELM.predict(X_test)
            MAES_TEST.append(mean_absolute_error(y_test, prediction))
        MAE_TEST_MINS.append(min(MAES_TEST))
        MAE_VAL_MINS.append(min(MAES_VAL))
        MAE_TRAIN_MINS.append(MAES_TRAIN[np.argmin(MAES_TEST)])

    print("Minimum MAE ELM =", min(MAE_TEST_MINS))
    test_maes_dictionary["ELM"] = min(MAE_TEST_MINS)
    
    ## LINEAR REGRESSION TRAINING
    mae = []
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr.fit(X_val, y_val)
    prediction = lr.predict(X_test)
    mae.append(mean_absolute_error(y_test, prediction))
    print("Minimum MAE LR =", min(mae))
    test_maes_dictionary["LinReg"] = min(mae)

    ## K-NEAREST NEIGHBORS TRAINING
    mae = []
    kn = KNeighborsRegressor()
    kn.fit(X_train, y_train)
    kn.fit(X_val, y_val)
    prediction = kn.predict(X_test)
    mae.append(mean_absolute_error(y_test, prediction))
    print("Minimum MAE KNN =", min(mae))
    test_maes_dictionary["KNN"] = min(mae)
    fpr, tpr, _ = roc_curve(y_test, prediction)
    plt.plot(fpr, tpr, label = 'K-Nearest neighbors')

    ## DECISION TREES TRAINING
    mae = []
    max_depth = 5
    min_samples_split = 50
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
    tree.fit(X_train, y_train)
    tree.fit(X_val, y_val)
    prediction = tree.predict(X_test)
    mae.append(mean_absolute_error(y_test, prediction))
    print("Minimum MAE TREE = ", min(mae))
    test_maes_dictionary["Dec. Tree"] = min(mae)

    ## SUPPORT VECTORS MACHINE TRAINING
    mae = []
    kernel = "sigmoid"
    svr = SVR(kernel=kernel)
    svr.fit(X_train, y_train)
    svr.fit(X_val, y_val)
    prediction = svr.predict(X_test)
    mae.append(mean_absolute_error(y_test, prediction))
    print("Minimum MAE SVR = ", min(mae))
    test_maes_dictionary["SVM"] = min(mae)


    ## RANDOM FOREST TRAINING
    mae = []
    n_estimators = 100

    rf = RandomForestRegressor(n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    rf.fit(X_val, y_val)
    prediction = rf.predict(X_test)
    mae.append(mean_absolute_error(y_test, prediction))
    print("Minimum MAE R.Forest = ", min(mae))
    test_maes_dictionary["R. Forest"] = min(mae)

    #plotting the ROC curve
    fpr, tpr, _ = roc_curve(y_test, prediction)
    plt.plot(fpr, tpr, label = 'Random Forest')
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.title("ROC Curve of RF, KNN")
    plt.ylabel('Probability of detection (POD)')
    plt.xlabel('Probability of false detection (POFD)')
    plt.legend(loc='best')
    plt.show()
    

    df = pd.DataFrame()
    df["test"] = MAE_TEST_MINS
    df["val"] = MAE_VAL_MINS
    df["train"] = MAE_TRAIN_MINS

    ax = df.plot(figsize=(16, 7))
    ax.set_xlabel("# of Neurons in the hidden layer")
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_title("ELM mean absolute error with varied neurons in the hidden layer")
    plt.show()
    
    plt.figure(figsize=(16, 7))
    D = test_maes_dictionary
    plt.bar(range(len(D)), D.values(), align='center')
    plt.xticks(range(len(D)), D.keys())
    plt.ylabel("Mean Absolute Error")
    plt.title("Error Comparison between Classic Regression Models and ELM")
    plt.show()
    

#prepare the data
X_train, y_train, X_val, y_val, X_test, y_test = prepareData()

#comparison to other types
compareToOthers()

#get ROC curve of ELM
neurons = 100
ROCCurve(neurons, 'blue', '', X_train, y_train, X_val, y_val, X_test, y_test)

#add variance to the data as extra column
X_train, X_val, X_test = addVariance(X_train, X_val, X_test)
ROCCurve(neurons, 'red', ['No Variance', 'Random actions', 'Variance'], X_train, y_train, X_val, y_val, X_test, y_test)
plt.show()