"""
This code defines a deep neural network model based on a “Long Short-Term Memory” (LSTM) architecture:

Layer Type      Number of Layers
LSTM  		    3
BiLSTM 		    3
LSTM  		    3
Regression	    1

First, the dataset is split into 90% for training and the ret for testing.
As the dataset is imbalanced, I used K-means clustering on the target variable and computed weights
based on the resulting groups. Then, I used this weight vector on each training step to correct the result
of the "loss function".

A High-Performance-Computing-Cluster (HPC-Cluster) was used, employing parallelization.
The file "hpc_job_GPU_v1.5.sh" contains the code for executing this file in an HPC-Cluster.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from math import sqrt
from sklearn.cluster import KMeans

# Feature Set: 20 MFCC, 20 delta-delta, 1 spectral center, 12 chroma, 12 chroma cens
num_features = 65
# The size of the hidden layer
HIDDEN_SIZE = num_features * 4
# The batch size
# 75 * 9 = 675
BATCH_SIZE = 675
OUTPUT_DIM = 1
NUM_LAYERS = 3
# The learning rate
LEARNING_RATE = 0.0005
NUM_EPOCHS = 1000

SEED = 42
# Set seeds for python, numpy and torch
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def load_data(file_name):
    print('FILE EXIST')
    # 1407 * 5 = 7035
    # When the csv file is too large or when there is not enough memory, then better to load the csv in chunks
    # featuresDF = pd.read_csv(file_name, sep=';', dtype={'STUDENT': str}, chunksize=7035)
    featuresDF = pd.read_csv(file_name, sep=';', dtype={'STUDENT': str})
    return featuresDF


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm1 = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.lstm2 = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, bidirectional=True, dropout=0.2)
        self.lstm3 = nn.LSTM(self.hidden_dim*2, self.hidden_dim, 2, dropout=0.2)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        # lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        lstm_out, self.hidden = self.lstm1(input)
        lstm_out2, self.hidden = self.lstm2(lstm_out)
        lstm_out3, self.hidden = self.lstm3(lstm_out2)

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        # y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        y_pred = self.linear(lstm_out3)
        return y_pred


def cluster_target(target_var, kmeans):
    # kmeans = KMeans(n_clusters=15, n_jobs=-1, random_state=SEED)
    # target_var = target_var.reshape(-1, 1)
    # kmeans.fit(target_var)
    target_varDF = pd.DataFrame(target_var)
    pred = kmeans.predict(target_var)
    target_varDF['cluster'] = pred

    weightDF = pd.DataFrame()
    weightDF['cluster'] = target_varDF['cluster'].value_counts().index
    weightDF['count'] = target_varDF['cluster'].value_counts().values
    weightDF['weight'] = weightDF['count'].sum() / weightDF['count']

    weightDF_Dict = {}
    weightDF_Dict = {int(row.cluster): row.weight for index, row in weightDF.iterrows()}
    return weightDF_Dict


if __name__ == '__main__':

    # dtype = torch.FloatTensor
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor  # Uncomment this to run on GPU
    else:
        dtype = torch.float

    file = '../mfcc_data/featuresNorm_MFCC_Extended20.csv'

    features = load_data(file)
    # features = next(iter(features))

    # X = features.iloc[:, 1:66].values
    X = np.hstack((features.iloc[:, 1:66].values, features['FILE'].values.reshape(len(features), 1)))
    y = features['LABEL_LEVEL'].values

    # files_summary = features['FILE'].value_counts()
    # pd.DataFrame(files_summary).to_csv("file_summary.csv", sep=';')
    # label_summary = features['LABEL_LEVEL'].value_counts()
    # pd.DataFrame(label_summary).to_csv("label_summary.csv", sep=';')

    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    gss.get_n_splits(X, y, features['FILE'])
    for train_index, test_index in gss.split(X, y, features['FILE']):
        X_train_Or, X_test = X[train_index], X[test_index]
        y_train_Or, y_test = y[train_index], y[test_index]

    # featuresTEST = np.hstack((X_test, y_test.reshape(len(y_test), 1)))
    # pd.DataFrame(featuresTEST).to_csv("featuresTEST.csv")

    # This used to shuffle the training data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    sss.get_n_splits(X_train_Or, y_train_Or)
    for train_index, test_index in sss.split(X_train_Or, y_train_Or):
        X_train, X_test_ = X_train_Or[train_index], X_train_Or[test_index]
        y_train, y_test_ = y_train_Or[train_index], y_train_Or[test_index]

    X_train = np.vstack((X_train, X_test_))
    y_train = np.hstack((y_train, y_test_))

    print(pd.DataFrame(y_train)[0].value_counts())
    print(pd.DataFrame(y_test)[0].value_counts())

    kmeans = KMeans(n_clusters=15, n_jobs=-1, random_state=SEED)
    target_var = y_train.reshape(-1, 1)
    kmeans.fit(target_var)
    weightDF_Dict = cluster_target(target_var, kmeans)

    X_train = torch.from_numpy(X_train[:, 0:65].astype(np.float32)).type(dtype)
    y_train = torch.from_numpy(y_train.astype(np.float32)).type(dtype)

    # X_test = torch.from_numpy(X_test[:, 0:65].astype(np.float32)).type(dtype)
    # X_test = X_test.unsqueeze(0)
    # y_test = torch.from_numpy(y_test.astype(np.float32)).type(dtype)

    # Y_train = np.array(featuresDF['LABEL_LEVEL']).astype(np.float32)
    # Y_train = torch.from_numpy(Y_train).type(dtype)
    # Y_train = torch.from_numpy(Y_train)

    # featureDF = featuresDF.to_numpy()
    # X_train = featureDF[:, 1:66].astype(np.float32)
    # X_train = torch.from_numpy(X_train.astype(np.float32)).type(dtype)
    # X_train = torch.from_numpy(X_train)
    # X_train = X_train.unsqueeze(0)

    # train_X = torch.from_numpy(X_train).type(torch.Tensor)
    # featureDF.to_csv("featuresTEST.csv")

    lstm_model = LSTM(num_features, HIDDEN_SIZE, batch_size=BATCH_SIZE, output_dim=OUTPUT_DIM, num_layers=NUM_LAYERS)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        lstm_model = nn.DataParallel(lstm_model)
    lstm_model.to(device)
    loss_function = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)

    print(lstm_model)

    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print("\nTraining on GPU")
    else:
        print("\nNo GPU, training on CPU")

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    hist = np.zeros(NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        # Init hidden state - if you don't want a stateful LSTM (between epochs)
        train_loss = 0.0
        lstm_model.hidden = lstm_model.init_hidden()
        for i in range(num_batches):
            # correct = 0
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            lstm_model.zero_grad()

            X_train_batch = X_train[i * BATCH_SIZE: (i+1)*BATCH_SIZE, ]
            y_train_batch = y_train[i * BATCH_SIZE: (i+1)*BATCH_SIZE, ]
            X_train_batch = X_train_batch.unsqueeze(0)

            # Step 3. Run our forward pass.
            y_pred = lstm_model(X_train_batch)
            y_pred = y_pred.squeeze()

            # find the clusters numbers on the prediction
            clusters = kmeans.predict(y_pred.cpu().detach().numpy().reshape(-1, 1))
            # get the weight corresponding to each cluster
            weights = [weightDF_Dict[j] for j in clusters]
            weights = torch.tensor(weights).type(dtype)

            # Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
            loss = loss_function(y_pred, y_train_batch)
            # Multiply the loss by the weight
            loss = loss * weights
            loss = loss.mean()
            train_loss += loss.detach().item()
            # loss = loss.type(dtype)

            # if epoch % 20 == 0:
            #     print("Epoch ", epoch, "MSE: ", loss.item())
                # Y_train_CPU = Y_train.cpu().numpy()
                # y_pred_CPU = y_pred.cpu().detach().numpy()
                # print("Expl Var Score: {}".format(explained_variance_score(Y_train_CPU, y_pred_CPU)))
                # print("R2 Score: {}".format(r2_score(Y_train_CPU, y_pred_CPU)))
            # hist[epoch] = loss.item()

            # Zero out gradient, else they will accumulate between epochs
            optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Update parameters
            optimizer.step()
        if epoch % 20 == 0:
            print("Epoch ", epoch, "MSE: ", train_loss/num_batches)
            print("Epoch ", epoch, "train_loss: ", train_loss)


            # if epoch % 10 == 0:
            #     correct += (y_pred == Y_train).float().sum()
            #     accuracy = 100 * correct / X_train.shape[1]
            #     print("Accuracy = {}".format(accuracy))
            #     print(y_pred)

    # print(hist)

    # Saving the model
    torch.save(lstm_model.state_dict(), 'LSTM_BiLSTM_model_weighted_dropout.pytorch')

    # Load the model for evaluation
    lstm_model.load_state_dict(torch.load('LSTM_BiLSTM_model_weighted_dropout'))
    lstm_model = nn.DataParallel(lstm_model)
    lstm_model.to(device)
    lstm_model.eval()

    # Load Features test
    file = 'featuresTEST.csv'
    featuresTEST = pd.read_csv(file, sep=',', dtype={'STUDENT': str})
    featuresTEST = pd.DataFrame(featuresTEST)
    # X_test = featuresTEST.iloc[:, 0:66].values
    # y_test = featuresTEST.iloc[:, 67:68].values

    splits = featuresTEST.groupby(['65'])

    results = pd.DataFrame(columns=['yTEST_Mean', 'yPRED_Mean', 'RMSE', 'R2_Score'])
    for key, data in splits:
        print(key)
        print(data.shape)

        X_test = data.iloc[:, 0:65].values
        y_test = data.iloc[:, 67:68].values

        X_test = torch.from_numpy(X_test.astype(np.float32)).type(dtype)
        X_test = X_test.unsqueeze(0)
        y_test = torch.from_numpy(y_test.astype(np.float32)).type(dtype)
        y_test = y_test.squeeze()

        y_pred = lstm_model(X_test)
        y_pred = y_pred.squeeze()
        loss = loss_function(y_pred, y_test)

        print('Mean Real Value: ', y_test.mean())
        print('Mean Predicted: ', y_pred.mean())
        print("MSE_loss: ", loss.detach().item())
        y_test = y_test.cpu().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        print("Expl Var Score: {}".format(explained_variance_score(y_test, y_pred)))
        print("R2 Score: {}".format(r2_score(y_test, y_pred)))
        R2_Score = r2_score(y_test, y_pred)
        print("RMSE: {}".format(sqrt(mean_squared_error(y_test, y_pred))))
        RMSE = sqrt(mean_squared_error(y_test, y_pred))

        results.loc[len(results)] = [y_test.mean(), y_pred.mean(), RMSE, R2_Score]

        print('...')
        print('...')

    results.to_csv('LSTM_results.csv', sep=';', float_format='%.4f')