
from string import punctuation

import numpy as np
import pandas as pd

# !!! MAKE SURE TO USE SVC.decision_function(X), NOT SVC.predict(X) !!!
# (this makes ``continuous-valued'' predictions)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
# self import
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from tabulate import tabulate

import warnings
#warnings.filterwarnings("ignore")



######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.
    
    Parameters
    --------------------
        fname  -- string, filename
        
    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


def write_label_answer(vec, outfile):
    """
    Writes your label vector to the given file.
    
    Parameters
    --------------------
        vec     -- numpy array of shape (n,) or (n,1), predicted scores
        outfile -- string, output filename
    """
    
    # for this project, you should predict 70 labels
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return
    
    np.savetxt(outfile, vec)    


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    
    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        infile    -- string, filename
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list
        lines = fid.readlines()
        index = 0
        for line in lines:   
            words = extract_words(line)
            for i in words:
                if (i in word_list.keys()) == False:
                    word_list[i] = index
                    index = index + 1

        ### ========== TODO : END ========== ###

    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = sum(1 for line in open(infile,'rU'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix

        # j = 0
        # for l in fid:
        #     words = extract_words(l)
        #     i = 0
        #     for k in words:
        #         if k in word_list:
        #             feature_matrix[i, j] = 1
        #         i = i + 1
        #     j = j + 1

        for index, line in enumerate(fid):

            words_extracted = extract_words(line)

            for i in words_extracted:
                index_i = word_list[i]
                feature_matrix[index,index_i] = 1



        pass
        ### ========== TODO : END ========== ###
        
    return feature_matrix


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1-score', 'auroc', 'precision',
                           'sensitivity', 'specificity'        
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1
    
    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance
    # score = 0
    if metric == "accuracy":
        score = metrics.accuracy_score(y_true, y_label)
    elif metric == "f1_score":
        score = metrics.f1_score(y_true, y_label)
    elif metric == "auroc":
        score = metrics.roc_auc_score(y_true, y_label)
    elif metric == "precision":
        score = metrics.precision_score(y_true, y_label)

    elif metric == "sensitivity":
        conf_matrix = metrics.confusion_matrix(y_true, y_label)
        score = conf_matrix[1,1]/float((conf_matrix[1,1]+conf_matrix[1, 0]))
        #score = metrics.confusion_matrix(y_true, y_label)

    elif metric == "specificity":
        #score = metrics.confusion_matrix(y_true, y_label)
        conf_matrix = metrics.confusion_matrix(y_true, y_label)
        score = conf_matrix[0,0]/float((conf_matrix[0,0]+conf_matrix[0,1]))
        
    return score
    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    ### ========== TODO : START ========== ###
    # part 2b: compute average cross-validation performance 
    score2 = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        #print("after clf.fit")
        y_pred_cv = clf.decision_function(X_test)
        score2.append(performance(y_true = y_test, y_pred = y_pred_cv, metric=metric))

    score2 = np.mean(score2)

    return score2
    ### ========== TODO : END ========== ###


def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """
    
    print ('Linear SVM Hyperparameter Selection based on ' + str(metric) + ':')
    C_range = 10.0 ** np.arange(-3, 3)
    
    ### ========== TODO : START ========== ###
    # part 2c: select optimal hyperparameter using cross-validation
    c = [10** -3, 10** -2, 10**-1, 10**0, 10**1, 10**2]
    score1 = []
    for i in c:
        model = SVC(kernel="linear", C=i)
        #print("after svc")
        score1.append(cv_performance(X=X, y=y, kf=kf, clf=model, metric=metric))

    best_index = score1.index(max(score1))

    #print("score1", score1)
    
    return score1[best_index], c[best_index]
    ### ========== TODO : END ========== ###


def select_param_rbf(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric  -- string, option used to select performance measure
    
    Returns
    --------------------
        gamma, C -- tuple of floats, optimal parameter values for an RBF-kernel SVM
    """
    
    print ('RBF SVM Hyperparameter Selection based on ' + str(metric) + ':')
    
    ### ========== TODO : START ========== ###
    # part 3b: create grid, then select optimal hyperparameters using cross-validation
    c = [10** -3, 10** -2, 10**-1, 10**0, 10**1, 10**2]
    r = [10** -3, 10** -2, 10**-1, 10**0, 10**1, 10**2]

    #c = [10** -3, 10** -2 ]
    #r = [10** -3, 10** -2]
    score2 = []
    best = 0
    for i in c:
        for j in r:
            model = SVC(kernel="rbf", C=i, gamma=j)
            score2.append(tuple((cv_performance(X=X, y=y, kf=kf, clf=model, metric=metric), i, j)))

    best = max(score2)
    #print("score2", score2)
    #print("Max", best)
    # return gamma, c
    return best[0], best[1], best[2]
    ### ========== TODO : END ========== ###


def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
    """

    ### ========== TODO : START ========== ###
    # part 4b: return performance on test data by first computing predictions and then calling performance
    #cv_performance()

    y_pred = clf.decision_function(X)
    score = performance(y_true=y, y_pred=y_pred, metric= metric)

    return score
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################
 
def main() :
    np.random.seed(1234)
    
    # read the tweets and its labels   
    dictionary = extract_dictionary('../data/tweets.txt')
    X = extract_feature_vectors('../data/tweets.txt', dictionary)

    y = read_vector_file('../data/labels.txt')
    
    metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    
    ### ========== TODO : START ========== ###
    # part 1c: split data into training (training + cross-validation) and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/9)


    # part 2b: create stratified folds (5-fold CV)
    
    skf = StratifiedKFold(n_splits = 5)

    # skf.split(X_train, y_train)
   
    
    # part 2d: for each metric, select optimal hyperparameter for linear-kernel SVM using CV
    best = []
    best.append("Best Score")
    best_c = []
    best_c.append("Best C")
    for m in metric_list:
        #result, c = select_param_linear(X_train, y_train, kf = skf.split(X_train, y_train), metric=m)
        result, c = select_param_linear(X_train, y_train, kf = skf, metric=m)
        best.append(result)
        best_c.append(c)

    table = [ metric_list, best, best_c]
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    print("From the result we can see that C=1 is the best for the model.")
    print("")

    # part 3a: How does gamma affect generalization error?
    print("part 3a: The gamma parameter defines how far the influence of a single training example reaches, ")
    print("with low values meaning ‘far’ and high values meaning ‘close’.")
    print("")

    # part 3b: what kind of grid you used and why
    print("part 3b: I check every possible combination of C and gamma.")
    print("")
    
    # part 3c: for each metric, select optimal hyperparameter for RBF-SVM using CV
    print("part 3c: for each metric, select optimal hyperparameter for RBF-SVM using CV")
    c_list = []
    c_list.append("Best c")

    g_list = []
    g_list.append("Gamma")

    s_list = []
    s_list.append("Score")
    for m in metric_list:
        #best_c, best_gamma, best_score = select_param_rbf(X_train, y_train, kf = skf.split(X_train, y_train), metric=m)
        best_score, best_c, best_gamma = select_param_rbf(X_train, y_train, kf = skf, metric=m)
        c_list.append(round(best_c,2))
        g_list.append(round(best_gamma,2))
        s_list.append(round(best_score,2))
    
    table = [ metric_list, s_list, c_list, g_list]
   
    print("c_list", c_list)
    print("g_list", g_list)
    print("s_list", s_list)
 

    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    print("Compare with Linear-Kernel SVM, we can see that in RBF-kernel SVM method c = 0.001 are most likely")
    print("to be the best, but in Linear-Kernel SVM c = 1 are the best.")
    print("")
    
    # part 4a: train linear- and RBF-kernel SVMs with selected hyperparameters

    print("part 4a: train linear- and RBF-kernel SVMs with selected hyperparameters")
    print("Based on the previous testing result, I choose c=1 for Linear Kernel SVM and c = 0.01, gamma = 0.01")
    print("for the RBF-kernel method.")
    print("")

    model3 = SVC(kernel="linear", C=1)
    model3.fit(X_train, y_train)
    result3 = performance_test(model3, X_test, y_test, metric="f1_score")
    

    model4 = SVC(kernel="rbf", C=0.01, gamma=0.01)
    model4.fit(X_train, y_train)
    result4 = performance_test(model4, X_test, y_test, metric="f1_score")
    




    #model2 = SVC(kernel="rbf", C=i, gamma=j)

    # part 4c: report performance on test data
    print("part 4c: report performance on test data")
    print("The performance of Linear-Kernel SVM is", result3, "which is good enough for this model.")
    print("The performance of RBF model", result4, "which is also good enough for this model. ")
    print("Base on the result, Linear-Kernel SVM is better than RBF model. ")
    ### ========== TODO : END ========== ###
    
    
if __name__ == "__main__" :
    main()
