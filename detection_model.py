import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from metric_learn import ITML, LMNN
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.base import clone
from imblearn.ensemble import RUSBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

pd.set_option('expand_frame_repr', False)


def rolling_window_forecast(X, y, start_year, end_year, window_size):
    classifiers = {
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'Logit': LogisticRegression(solver='liblinear'),
        'Probit': LogisticRegression(solver='sag'),
        'NaiveBayes': GaussianNB(),
        'Tree-ID3': DecisionTreeClassifier(criterion='entropy'),
        'Tree-C4.5': DecisionTreeClassifier(criterion='entropy'),
        'Tree-CART': DecisionTreeClassifier(criterion='gini'),
        'FNN1': MLPClassifier(hidden_layer_sizes=(32,), solver='sgd'),
        'FNN2': MLPClassifier(hidden_layer_sizes=(32, 32), solver='sgd'),
        'FNN3': MLPClassifier(hidden_layer_sizes=(32, 32, 32), solver='sgd'),
        'SVM-Lin': SVC(kernel='linear'),
        'SVM-Poly': SVC(kernel='poly'),
        'SVM-RBF': SVC(kernel='rbf'),
        'KNN': KNeighborsClassifier(),
        # 'NCA': NeighborhoodComponentsAnalysis(),
        # 'ITML': ITML(),
        # 'LMNN': LMNN()
        'RUSBoost': RUSBoostClassifier(n_estimators=3000, random_state=42)
    }


    for year in range(start_year, end_year):
        train_start = year - window_size
        train_end = year - 1

        X_train = X.loc[(slice(None), slice(train_start, train_end)), :]
        y_train = y.loc[(slice(None), slice(train_start, train_end)), :]
        X_test = X.loc[(slice(None), year), :]
        y_test = y.loc[(slice(None), year), :]
        test_index = X_test.index


        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.fit_transform(y_test)

        M = len(X_train)
        N = len(X_train)+len(X_test)
        d = int((N-M)/10)
        J = window_size
        predictions = SG_AL(X_train, y_train, X_test, y_test, classifiers, M, N, d, J)

        predictions_df = pd.DataFrame({year: predictions})
        predictions_df.index = test_index
        predictions_df.to_csv("prediction_"+str(year)+".csv")
        print("The prection for year " + str(year)+" has completed!")


def SG_AL(X_train, y_train, X_test, y_test, classifiers, M, N, d, J):

    while d <= (N - M):
        # Step 1: Detection results using multiple classifiers
        Z_train = np.zeros((M, len(classifiers)))
        Z_test = np.zeros((N - M, len(classifiers)))
        for k, clf in enumerate(classifiers):
            estimator = clone(classifiers[clf])
            predictions = cross_val_predict(estimator, X_train, y_train, cv=J)
            Z_train[:, k] = predictions
            estimator.fit(X_train, y_train)
            Z_test[:, k] = estimator.predict(X_test)
            print("Stage 1:"+ clf + " has been fitted!")

        # Step 2: Aggregation of an ensemble of the K detection results
        P_test = np.zeros_like(Z_test)
        for k, clf in enumerate(classifiers):
            estimator = clone(classifiers[clf])
            predictions = cross_val_predict(estimator, Z_train, y_train, cv=J)
            estimator.fit(Z_train, y_train)
            P_test[:, k] = estimator.predict(Z_test)
            print("Stage 2:"+ clf + " has been fitted!")

        p_mean = P_test.mean(axis=1)
        p_std = P_test.std(axis=1)
        np.save('p_mean.npy',p_mean)
        np.save('p_std.npy', p_std)

        print("The part of SG has completed! Still interating...")

        # Step 3: Feedback to update the training set
        # p_mean = np.load('p_mean.npy', allow_pickle=True)
        # p_std = np.load('p_std.npy', allow_pickle=True)
        most_consensus_indices = np.argsort(p_std)[:d]
        feedback_X = X_test[most_consensus_indices]
        feedback_y = [min(set(y['Label']), key=lambda y: abs(p_mean[i] - y)) for i in most_consensus_indices]

        # Update training set
        X_train = np.vstack([X_train, feedback_X])
        y_train = np.hstack([y_train, feedback_y])

        # Remove feedback samples from test set
        X_test = np.delete(X_test, most_consensus_indices, axis=0)
        y_test = np.delete(y_test, most_consensus_indices, axis=0)

        d += len(most_consensus_indices)
        print("The part of AL has completed! Await interation...")
        break

    print("Interation completes! ")
    predictions = p_mean
    # predictions = [min(set(y['Label']), key=lambda y: abs(p - y)) for p in p_mean]

    return predictions

def FarmPredict(factors):
    X = factors.to_numpy()
    n, m = X.shape

    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    C = np.dot(Z.T, Z) / (Z.shape[0] - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(C)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    threshold = 1 + np.sqrt(m / (n - 1))
    k = max(np.where(eigenvalues > threshold)[0]) + 1

    V_k = eigenvectors[:, :k]
    F = np.dot(Z, V_k)
    B = np.linalg.inv(F.T @ F) @ F.T @ Z
    U = Z - np.dot(F, B)

    FplusU = np.hstack((F, U))
    factors_new = pd.DataFrame(data=FplusU, index=factors.index, columns=[f'FarmFactors{i + 1}' for i in range(k+m)])

    return factors_new


if __name__ == '__main__':
    data = pd.read_csv('..\model_input.csv')
    data.set_index(['Symbol','ReportYear'],inplace=True)

    X = data.iloc[:, :-1]
    X = FarmPredict(X)
    y = pd.DataFrame(data.iloc[:, -1])

    rolling_window_forecast(X, y, 2006, 2022, 5)

