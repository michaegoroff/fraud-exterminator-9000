import argparse
import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from argparser.argparser import ArgParser
from image_scanner.scanner import Scanner
from models.LogisticRegression import LogisticRegression
from models.SupportVectorMachine import SVM
from utils import display_results, save_custom, load_custom

#argument parser
parser = ArgParser()

#training stage
if parser.get_args().stage == 'train':

    #read csv
    df = pd.read_csv('data/'+parser.get_args().file)
    #split to X and Y
    X = df.drop('class', axis=1)
    Y = df['class']
    #split to train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=69, shuffle=True)
    #perform scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    #LogReg model (custom)
    if(parser.get_args().model == 'logreg_custom'):
        regressor = LogisticRegression(learning_rate=parser.get_args().learning_rate, iters=parser.get_args().iterations)
        #train
        regressor.fit(X_train, Y_train)
        #make predictions
        predictions = regressor.predict(X_test)
        predictions_proba = regressor.predict_proba(X_test)
        display_results(Y_test, predictions, predictions_proba)
        #save model
        save_custom(regressor, parser.get_args().model)

    #support vector machine (custom)
    elif(parser.get_args().model == 'svm_custom'):
        svm = SVM(learning_rate=parser.get_args().learning_rate, iters=parser.get_args().iterations)
        #train
        svm.fit(X_train, Y_train)
        #make predictions
        predictions = svm.predict(X_test)
        predictions_proba = svm.predict_proba(X_test)
        display_results(Y_test, predictions, predictions_proba)
        #save model
        save_custom(svm, parser.get_args().model)

    #random forest (sklearn-built-in)
    elif(parser.get_args().model == 'r_forest'):
        forest = RandomForestClassifier(n_estimators=1000, random_state=69)
        #train
        forest.fit(X_train, Y_train)
        #make predictions
        predictions = forest.predict(X_test)
        predictions_proba = forest.predict_proba(X_test)
        display_results(Y_test, predictions, predictions_proba)
        #save model
        joblib.dump(forest, 'saved_params/' + parser.get_args().model + '_' + '.pkl')

    #linear regression (sklearn-built-in)
    elif(parser.get_args().model == 'logreg'):
        regressor = sklearn.linear_model.LogisticRegression()
        #train
        regressor.fit(X_train, Y_train)
        #make predictions
        predictions = regressor.predict(X_test)
        predictions_proba = regressor.predict_proba(X_test)
        display_results(Y_test, predictions, predictions_proba)
        #save model
        joblib.dump(regressor, 'saved_params/' + parser.get_args().model + '_' + '.pkl')

    else:
        raise argparse.ArgumentTypeError('Invalid model argument')

#prediction stage
elif parser.get_args().stage == 'predict':
    #scan provided image and get metrics
    scanner = Scanner(parser.get_args().image_folder)
    test_samples, class_labels = scanner.process_batch()

    #predict if it's a legitimate bill or a counterfeit one
    if parser.get_args().model == 'logreg_custom':
        #load custom model and pass the weights & biases
        weights, bias = load_custom(parser.get_args().model)
        regressor = LogisticRegression(learning_rate=parser.get_args().learning_rate, iters=parser.get_args().iterations, weights=weights, bias=bias)
        #predict
        predictions = regressor.predict(test_samples)
        predictions_proba = regressor.predict_proba(test_samples)
        display_results(class_labels, predictions, predictions_proba)
        #print_prediction_result(predictions, test_samples)

    elif parser.get_args().model == 'svm_custom':
        #load custom model and pass the weights & biases
        weights, bias = load_custom(parser.get_args().model)
        svm = SVM(learning_rate=parser.get_args().learning_rate, iters=parser.get_args().iterations, weights=weights, bias=bias)
        predictions = svm.predict(test_samples)
        predictions_proba = svm.predict_proba(test_samples)
        display_results(class_labels, predictions, predictions_proba)
        #print_prediction_result(predictions, test_samples)

    elif parser.get_args().model == 'r_forest':
        forest = joblib.load('saved_params/' + parser.get_args().model + '_' + '.pkl')
        #predict
        #reshape test sample to 2d: [test_sample]
        predictions = forest.predict(test_samples)
        predictions_proba = forest.predict_proba(test_samples)
        display_results(class_labels, predictions, predictions_proba)
        #print_prediction_result(predictions, test_samples)

    elif parser.get_args().model == 'logreg':
        regressor = joblib.load('saved_params/' + parser.get_args().model + '_' + '.pkl')
        #predict
        #reshape test sample to 2d: [test_sample]
        predictions = regressor.predict(test_samples)
        predictions_proba = regressor.predict_proba(test_samples)
        display_results(class_labels, predictions, predictions_proba)
        #print_prediction_result(predictions, test_samples)
    else:
        raise argparse.ArgumentTypeError('Invalid model argument')

else:
    raise argparse.ArgumentTypeError('Invalid stage argument')
