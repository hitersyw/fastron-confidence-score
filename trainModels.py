from scipy.spatial.distance import cdist
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from os import path

import json
import numpy as np
import sys

def rbf(x1, x2, g=1):
  return np.exp(-g * (cdist(x1, x2, 'euclidean') ** 2))

def load_data(dataset_path):

  training_file = path.join(dataset_path, "training.csv")
  holdout_file = path.join(dataset_path, "validation.csv")
  test_file = path.join(dataset_path, "test.csv")

  training_data = np.loadtxt(training_file, skiprows=0, delimiter=',')
  validation_data = np.loadtxt(holdout_file, skiprows=0, delimiter=',')
  test_data = np.loadtxt(test_file, skiprows=0, delimiter=',')

  training_X = training_data[:, 0:2]
  training_y = training_data[:, 2]
  training_y[training_y == -1] = 0  # convert y to 0;

  validation_X = validation_data[:, 0:2]
  validation_y = validation_data[:, 2]
  validation_y[validation_y == -1] = 0

  test_X = test_data[:, 0:2]
  test_y = test_data[:, 2]
  test_y[test_y == -1] = 0  # convert y to 0;

  return training_X, training_y, validation_X, validation_y, test_X, test_y


def trainMLP(data, dataset_path):
  print("Starting training for MLP...")
  training_X, training_y, validation_X, validation_y, test_X, test_y = data

  # TODO: Do we need to transform the output from {-1, 1} to {0, 1}?
  mlp_clf = MLPClassifier(solver='adam', alpha=1e-4, activation='relu', hidden_layer_sizes=(256, 256),
                          random_state=1, max_iter=1000, verbose=True).fit(training_X, training_y)

  y_mlp_validation = mlp_clf.predict(validation_X)
  p_mlp_validation = mlp_clf.predict_proba(validation_X).T

  y_mlp_test = mlp_clf.predict(test_X)
  p_mlp_test = mlp_clf.predict_proba(test_X).T

  header_mlp = ['y_pred'] + mlp_clf.classes_.tolist()
  # print(header_mlp)
  output_mlp_validation = np.vstack((y_mlp_validation, p_mlp_validation)).T
  output_mlp_test = np.vstack((y_mlp_test, p_mlp_test)).T

  coefs = []
  intercepts = []
  for coef in mlp_clf.coefs_:
    coefs.append(coef.tolist())

  for intercept in mlp_clf.intercepts_:
    intercepts.append(intercept.tolist())

  output_mlp = {
    'coef': coefs,
    'intercept': intercepts,
    'output_header': header_mlp,
    'validation_output': output_mlp_validation.tolist(),
    'test_output': output_mlp_test.tolist()
  }

  mlp_output_file = path.join(dataset_path, "mlp.json")

  with open(mlp_output_file, 'w') as outfile:
    json.dump(output_mlp, outfile)

def trainLogReg(data, dataset_path, g=1.):
  print("Start training for Kernel Log Reg...")
  training_X, training_y, validation_X, validation_y, test_X, test_y = data

  # Compute kernels;
  K_train = rbf(training_X, training_X, g)
  K_validation = rbf(validation_X, training_X)
  K_test = rbf(test_X, training_X)

  sgd_clf = SGDClassifier(loss='log', alpha=1e-4, random_state=0, verbose=1).fit(K_train, training_y)
  y_lr_holdout = sgd_clf.predict(K_validation)
  p_lr_holdout = sgd_clf.predict_proba(K_validation).T
  y_lr_test = sgd_clf.predict(K_test)
  p_lr_test = sgd_clf.predict_proba(K_test).T  # Need to align the first axis with y_pred

  header_sgd = ['y_pred'] + sgd_clf.classes_.tolist()
  # print(header_sgd)
  output_sgd_validation = np.vstack((y_lr_holdout, p_lr_holdout)).T
  output_sgd_test = np.vstack((y_lr_test, p_lr_test)).T
  output_sgd = {
    'coef': sgd_clf.coef_.tolist(),
    'intercept': sgd_clf.intercept_.tolist(),
    'output_header': header_sgd,
    'validation_output': output_sgd_validation.tolist(),
    'test_output': output_sgd_test.tolist()
  }

  sgd_output_file = path.join(dataset_path, "sgd.json")
  with open(sgd_output_file, 'w') as outfile:
    json.dump(output_sgd, outfile)


def main(base_path, n, g):
  for i in range(1,n + 1):
    print("Training models for dataset%d" % i)
    dataset_path = path.join(base_path, "CSpace%d" % i)
    data = load_data(dataset_path)
    # train models here;
    trainMLP(data, dataset_path)
    trainLogReg(data, dataset_path, g)

if __name__ =="__main__":
  base_path = sys.argv[1]
  n = int(sys.argv[2])
  g = float(sys.argv[3])
  main(base_path, n, g)
