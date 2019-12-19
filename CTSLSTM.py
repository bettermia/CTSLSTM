from multiChannelLSTM import MultiChannelLSTM
from multiChannelLSTMFusion import Fusion

import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers import TimeDistributed, LSTM, ConvLSTM2D, MaxPooling3D
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import plot_model
from keras.optimizers import Adam

from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib
matplotlib.use('nbagg')
from matplotlib import pyplot
from utils import *
import sys
import os

class CTSLSTM(object):
    def __init__(self, hps, mode='train'):
        self.hps = hps
        self.mode = mode

    def run(self):
        # use specific gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = self.hps['gpu_id']
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        
        # load data
        self.loadData(self.hps['train_data_path'], self.hps['test_data_path'])
        print('--------------------------------')
        print('trainX.shape: ', self.trainX.shape)
        print('trainY.shape: ', self.trainY.shape)
        print('testX.shape: ', self.testX.shape)
        print('testY.shape: ', self.testY.shape)
        print('--------------------------------')

        # model construction
        model= self.buildModel(self.hps['output_path'])

        # plot the architecture of model 
        plot_model(model, to_file=self.hps['output_path'] + '/model.png', show_shapes=True)

        # 1) print the minimun RMSE and MAE, and figure out the best models with minimun RMSE.
        # 2) plot the RMSE of each epoch, and save as a figure.
        self.plot_loss(self.hps['output_path'])

    def compileModel(self):
        '''
        CTSLSTM, many-to-many architecture whith MultiChannelLSTM layers using ST-cell.
        '''
        model = Sequential()

        channels, cols = 1, 1
        model.add(Reshape((-1, channels, self.hps['n_input'], cols), 
                           input_shape=(self.hps['n_steps'], self.hps['n_input'])))
        
        model.add(MultiChannelLSTM(self.hps['n_hidden_in'], 
                                   return_sequences=True, 
                                   activation='relu', 
                                   dropout=self.hps['dropout_rate']))

        model.add(MultiChannelLSTM(self.hps['n_hidden_out'], 
                                   return_sequences=True, 
                                   activation='relu', 
                                   dropout=self.hps['dropout_rate']))

        model.add(TimeDistributed(Fusion(self.hps['n_fusion'], activation='relu')))
        model.add(TimeDistributed(Dense(self.hps['n_output'], activation='tanh')))

        optimizer = Adam(lr=self.hps['learning_rate'])
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        return model

    def buildModel(self, output_path):
        '''
        Train and save model.

        Parameters
        ----------
        output_path: The output path.

        Returns
        ----------
        Trained model.

        '''
        model = self.compileModel()

        save_model_path = output_path + '/model'
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        fname = os.path.join(save_model_path, '{}.h5'.format('{epoch:04d}'))
        model_checkpoint = ModelCheckpoint(fname, 
                                           monitor='loss',
                                           verbose=0, 
                                           save_best_only=False, 
                                           mode='min',
                                           period=1)

        model.fit(self.trainX,
                  self.trainY,
                  verbose=0, 
                  batch_size=self.hps['batch_size'], 
                  epochs=self.hps['epoch'],
                  callbacks=[model_checkpoint])

        return model

    def plot_loss(self, output_path):
        '''
        1) print the minimun RMSE and MAE, and figure out the best models with minimun RMSE.
        2) plot the RMSE of each epoch, and save as a figure.
        '''
        model = self.compileModel()

        trainHistory, testHistory = [], []
        cnt = 1
        while cnt <= self.hps['epoch']:
            fname = output_path + '/model/{:04d}'.format(cnt) + '.h5'
            model.load_weights(fname)

            trainPredict, testPredict = self.predict(model)

            trainY = []
            for i in range(len(self.trainY)):
                if i == 0:
                    trainY = self.trainY[i].reshape((self.trainY.shape[1],self.trainY.shape[2]))
                else:
                    trainY = np.append(trainY, 
                                       self.trainY[i].reshape((self.trainY.shape[1],self.trainY.shape[2])), 
                                       axis=1)

            testY = []
            for i in range(len(self.testY)):
                if i == 0:
                    testY = self.testY[i].reshape((self.testY.shape[1],self.testY.shape[2]))
                else:
                    testY = np.append(testY, 
                                      self.testY[i].reshape((self.testY.shape[1],self.testY.shape[2])), 
                                      axis=1)

            trainY = np.transpose(trainY)
            testY = np.transpose(testY)

            # inverse normalization
            trainY = inverse_scaler(trainY, 
                                    self.max_dataTrain[0:-self.hps['n_test_samples']*self.hps['n_output']], 
                                    self.min_dataTrain[0:-self.hps['n_test_samples']*self.hps['n_output']])
            testY = inverse_scaler(testY, 
                                   self.max_dataTest[-self.hps['n_test_samples']*self.hps['n_output']:], 
                                   self.min_dataTest[-self.hps['n_test_samples']*self.hps['n_output']:])
            
            trainMSE, trainMAE, trainMAPE, testMAPE, testMSE, testMAE = self.evalModel(trainPredict, testPredict, trainY, testY)
            self.setMin(trainMSE, trainMAE, trainMAPE, testMAPE, testMSE, testMAE, cnt)

            trainHistory.append(np.sqrt(trainMSE))
            testHistory.append(np.sqrt(testMSE))

            cnt += 1

        self.printMin()
        self.saveMin(output_path)

        # plot the RMSE of each epoch, and save as a figure
        pyplot.title('RMSE')

        pyplot.plot(trainHistory, label='train')
        pyplot.plot(testHistory, label='test')

        pyplot.legend()
        pyplot.savefig(output_path + '/loss.jpg')
        pyplot.clf()
    
    def predict(self, model):
        trainTemp = model.predict(self.trainX, batch_size=8192)
        testTemp = model.predict(self.testX, batch_size=2048)
        
        trainPredict = []
        for i in range(len(trainTemp)):
            if i == 0:
                trainPredict = trainTemp[i].reshape((trainTemp.shape[1],trainTemp.shape[2]))
            else:
                trainPredict = np.append(trainPredict, 
                                         trainTemp[i].reshape((trainTemp.shape[1],trainTemp.shape[2])),
                                         axis=1)

        testPredict = []
        for i in range(len(testTemp)):
            if i == 0:
                testPredict = testTemp[i].reshape((testTemp.shape[1],testTemp.shape[2]))
            else:
                testPredict = np.append(testPredict, 
                                        testTemp[i].reshape((testTemp.shape[1],testTemp.shape[2])),
                                        axis=1)
        
        trainPredict = np.transpose(trainPredict)
        testPredict = np.transpose(testPredict)

        testPredict = inverse_scaler(testPredict,
                                     self.max_dataTest[-self.hps['n_test_samples']*self.hps['n_input']:],
                                     self.min_dataTest[-self.hps['n_test_samples']*self.hps['n_input']:])
        trainPredict = inverse_scaler(trainPredict,
                                      self.max_dataTrain[0:-self.hps['n_test_samples']*self.hps['n_input']],
                                      self.min_dataTrain[0:-self.hps['n_test_samples']*self.hps['n_input']])        
        
        return trainPredict, testPredict
    
    def evalModel(self, trainPredict, testPredict, trainY, testY):
        trainMSE = mean_squared_error(trainY, trainPredict)
        trainMAE = mean_absolute_error(trainY, trainPredict)
        trainMAPE = mean_absolute_percentage_error(trainY, trainPredict)

        testMSE = mean_squared_error(testY[:,-1], testPredict[:,-1])
        testMAE = mean_absolute_error(testY[:,-1], testPredict[:,-1])
        testMAPE = mean_absolute_percentage_error(testY[:,-1], testPredict[:,-1])

        return trainMSE, trainMAE, trainMAPE, testMAPE, testMSE, testMAE
    
    def loadData(self, train_path, test_path):
        # ------ load training set ------
        train = pd.read_csv(train_path, header=None)

        # drop station name and time
        train = train.drop([0,1],axis=1)

        # calculate the maximun and minimun of each column
        self.max_dataTrain = pd.DataFrame(train.max(axis=1)).values
        self.min_dataTrain = pd.DataFrame(train.min(axis=1)).values
        
        # ------ load training set ------
        test = pd.read_csv(test_path,header=None)

        test = test.drop([0,1],axis=1)

        self.max_dataTest = pd.DataFrame(test.max(axis=1)).values
        self.min_dataTest = pd.DataFrame(test.min(axis=1)).values

        # normalization
        train = scaler(train.values, self.max_dataTrain, self.min_dataTrain)
        test = scaler(test.values, self.max_dataTest, self.min_dataTest)

        # construct training set
        rows = 0
        trainX, trainY = [], []
        while rows <= (len(train)-self.hps['n_input']):
            if rows == 0:
                temp = np.transpose(train[rows:rows+self.hps['n_input'], 0:self.hps['n_steps']])
                trainX = temp.reshape((1, self.hps['n_steps'], self.hps['n_input']))

                temp = np.transpose(train[rows:rows+self.hps['n_output'], self.hps['n_steps']:])            
                trainY = temp.reshape((1, self.hps['n_steps'], self.hps['n_output']))
            else:
                temp = np.transpose(train[rows:rows+self.hps['n_input'],0:self.hps['n_steps']])
                trainX = np.append(trainX,
                                   temp.reshape((1,self.hps['n_steps'],self.hps['n_input'])),
                                   axis=0)

                temp = np.transpose(train[rows:rows+self.hps['n_output'],self.hps['n_steps']:])
                trainY = np.append(trainY,
                                   temp.reshape((1,self.hps['n_steps'],self.hps['n_output'])),
                                   axis=0)

            rows = rows + self.hps['n_input']
        
        # construct test set
        rows = 0
        testX, testY = [], []
        while rows <= (len(test)-self.hps['n_input']):
            if rows == 0:
                temp = np.transpose(test[rows:rows+self.hps['n_input'],0:self.hps['n_steps']])
                testX = temp.reshape((1,self.hps['n_steps'],self.hps['n_input']))

                temp =  np.transpose(test[rows:rows+self.hps['n_output'],self.hps['n_steps']:])    
                testY = temp.reshape((1,self.hps['n_steps'],self.hps['n_output']))
            else:
                temp = np.transpose(test[rows:rows+self.hps['n_input'],0:self.hps['n_steps']])
                testX = np.append(testX,
                                  temp.reshape((1,self.hps['n_steps'],self.hps['n_input'])),
                                  axis=0)

                temp =  np.transpose(test[rows:rows+self.hps['n_output'],self.hps['n_steps']:])
                testY = np.append(testY,
                                  temp.reshape((1,self.hps['n_steps'],self.hps['n_output'])),
                                  axis=0)
            rows += self.hps['n_input']

        # split dataset into training set and test set
        self.trainX = trainX[0:-self.hps['n_test_samples']]
        self.trainY = trainY[0:-self.hps['n_test_samples']]

        self.testY = testY[-self.hps['n_test_samples']:]
        self.testX = testX[-self.hps['n_test_samples']:]
    
    def setMin(self, trainMSE, trainMAE, trainMAPE, testMAPE, testMSE, testMAE, cnt):
        if cnt == 1:
            self.trainRMSE = np.sqrt(trainMSE)
            self.testRMSE = np.sqrt(testMSE)

            self.trainMAE = trainMAE
            self.testMAE = testMAE 

            self.trainMAPE = trainMAPE
            self.testMAPE = testMAPE 

            self.bestEpoch = cnt
        else:
            if self.testRMSE > np.sqrt(testMSE):
                self.trainRMSE = np.sqrt(trainMSE)
                self.testRMSE = np.sqrt(testMSE)

                self.trainMAE = trainMAE
                self.testMAE = testMAE 

                self.trainMAPE = trainMAPE
                self.testMAPE = testMAPE 

                self.bestEpoch = cnt
    
    def printMin(self):
        print('--------------------------------')

        print('best epoch: ', self.bestEpoch)

        print('train RMSE:', self.trainRMSE)
        print('test RMSE: ', self.testRMSE)

        print('train MAE: ', self.trainMAE)
        print('test MAE: ', self.testMAE)

        print('train MAPE: ', self.trainMAPE)
        print('test MAPE: ', self.testMAPE)

        print('--------------------------------')

    def saveMin(self, output_path):
        with open(output_path + '/performance.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([self.testRMSE, self.testMAE, self.testMAPE])
    
