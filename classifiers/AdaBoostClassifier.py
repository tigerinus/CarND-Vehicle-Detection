import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle
import time

from scipy.ndimage.measurements import label

from skimage.feature import hog
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn import discriminant_analysis
from sklearn import ensemble
from sklearn import gaussian_process 
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm
from sklearn import tree
from sklearn import neural_network

from IPython.display import HTML
from moviepy.editor import VideoFileClip
from tqdm import tqdm


orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = 32
hist_bins = 32

conv= cv2.COLOR_RGB2YCrCb
invconv = cv2.COLOR_YCR_CB2RGB

hist_bin_range = (0, 255)

# max 5068
training_size = 5068

figure_size = (16, 16)

X_train_npy = 'X_train.npy'
X_test_npy = 'X_test.npy'
y_train_npy = 'y_train.npy'
y_test_npy = 'y_test.npy'

#parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 'C':[1, 3, 6, 9]}
#svc = svm.SVC()
#clf = GridSearchCV(svc, parameters, n_jobs=4, verbose=3)
#clf.fit(X_train, y_train)

#accuracy = clf.score(X_test, y_test)
#params = clf.best_params_
#print('Accuracy = {}, Best Parameters = {}'.format(accuracy, params))

# Best Parameters = {'C': 9, 'kernel': 'rbf'}

#clf = svm.SVC(C=9, kernel='rbf')
#clf = svm.LinearSVC()
#clf = tree.DecisionTreeClassifier()
#clf = naive_bayes.GaussianNB()


if os.path.isfile(X_train_npy):
    X_train = np.load(X_train_npy)

if os.path.isfile(y_train_npy):
    y_train = np.load(y_train_npy)

if os.path.isfile(X_test_npy):
    X_test = np.load(X_test_npy)

if os.path.isfile(y_test_npy):
    y_test = np.load(y_test_npy)


clf_list = {
    #'KNeighborsClassifier': neighbors.KNeighborsClassifier(2),
    #'SVC_linear': svm.SVC(kernel="linear", C=0.025),
    #'LinearSVC': svm.LinearSVC(C=0.025),
    #'SVC': svm.SVC(gamma=2, C=1),
    #'GaussianProcessClassifier': gaussian_process.GaussianProcessClassifier(1.0 * gaussian_process.kernels.RBF(1.0)),
    #'DecisionTreeClassifier': tree.DecisionTreeClassifier(max_depth=5),
    #'RandomForestClassifier': ensemble.RandomForestClassifier(max_depth=5),
    #'MLPClassifier': neural_network.MLPClassifier(),
    'AdaBoostClassifier': ensemble.AdaBoostClassifier(),
    #'GaussianNB': naive_bayes.GaussianNB(),
    #'QuadraticDiscriminantAnalysis': discriminant_analysis.QuadraticDiscriminantAnalysis(),
    #'SGDClassifier': linear_model.SGDClassifier()
}


for clf_name in clf_list:
    print('Training {}...'.format(clf_name))

    clf_file = '{}.pickle'.format(clf_name)
    if os.path.isfile(clf_file):
        print('{} found. Loading clf from it...'.format(clf_file))
        clf_list[clf_name] = pickle.load(open(clf_file, 'rb'))
    else:
        clf = clf_list[clf_name]
    
        t1=time.time()
        clf.fit(X_train, y_train)
        t2 = time.time()
        print('Traning took {} seconds.'.format(round(t2 - t1, 2)))

        pickle.dump(clf, open(clf_file, 'wb'))

for clf_name in clf_list:
    clf = clf_list[clf_name]
    
    print('Testing {}...'.format(clf_name))
    t1=time.time()
    accuracy = clf.score(X_test, y_test)
    t2 = time.time()
    print('Testing took {} seconds. Accuracy = {}.'.format(round(t2 - t1, 2), accuracy))