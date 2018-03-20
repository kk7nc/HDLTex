import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN"

import numpy as np
from keras.models import Sequential
import Data_helper
import BuildModel

if __name__ == "__main__":

    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    MEMORY_MB_MAX = 1600000 # maximum memory you can use
    MAX_SEQUENCE_LENGTH = 500 # Maximum sequance lenght 500 words
    MAX_NB_WORDS = 55000 # Maximum number of unique words
    EMBEDDING_DIM = 100 #embedding dimension you can change it to {25, 100, 150, and 300} but need to change glove version
    batch_size_L1 = 64 # batch size in Level 1
    batch_size_L2 = 64 # batch size in Level 2
    epochs = 1

    L1_model =0 # 0 is DNN, 1 is CNN, and 2 is RNN for Level 1
    L2_model =2 # 0 is DNN, 1 is CNN, and 2 is RNN for Level 2

    np.set_printoptions(threshold=np.inf)
    '''
    location of input data in two ways 
    1: Tokenizer that is using GLOVE
    1: loadData that is using couting words or tf-idf
    '''

    X_train, y_train, X_test, y_test, content_L2_Train, L2_Train, content_L2_Test, L2_Test, number_of_classes_L2,word_index, embeddings_index,number_of_classes_L1 =  \
            Data_helper.loadData_Tokenizer(MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)

    X_train_DNN, y_train_DNN, X_test_DNN, y_test_DNN, content_L2_Train_DNN, L2_Train_DNN, content_L2_Test_DNN, L2_Test_DNN, number_of_classes_L2_DNN = Data_helper.loadData()
    print("Loading Data is Done")
    #######################DNN Level 1########################
    if L1_model == 0:
        print('Create model of DNN')
        model = BuildModel.buildModel_DNN(X_train_DNN.shape[1], number_of_classes_L1, 8, 64, dropout=0.25)
        model.fit(X_train_DNN, y_train_DNN[:, 0],
                  validation_data=(X_test_DNN, y_test_DNN[:, 0]),
                  epochs=epochs,
                  verbose=2,
                  batch_size=batch_size_L1)

    #######################CNN Level 1########################
    if L1_model == 1:
        print('Create model of CNN')
        model = BuildModel.buildModel_CNN(word_index, embeddings_index,number_of_classes_L1,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,1)
        model.fit(X_train, y_train[:,0],
                  validation_data=(X_test, y_test[:,0]),
                  epochs=epochs,
                  verbose=2,
                  batch_size=batch_size_L1)
    #######################RNN Level 1########################
    if L1_model == 2:
        print('Create model of RNN')
        model = BuildModel.buildModel_RNN(word_index, embeddings_index,number_of_classes_L1,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM)
        model.fit(X_train, y_train[:,0],
                  validation_data=(X_test, y_test[:,0]),
                  epochs=epochs,
                  verbose=2,
                  batch_size=batch_size_L1)

    HDLTex = [] # Level 2 models is list of Deep Structure
    ######################DNN Level 2################################
    if L2_model == 0:
        for i in range(0, number_of_classes_L1):
            print('Create Sub model of ',i)
            HDLTex.append(Sequential())
            HDLTex[i] = BuildModel.buildModel_DNN(content_L2_Train_DNN[i].shape[1], number_of_classes_L2_DNN[i],2, 1024, dropout=0.5)
            HDLTex[i].fit(content_L2_Train_DNN[i], L2_Train_DNN[i],
                      validation_data=(content_L2_Test_DNN[i], L2_Test_DNN[i]),
                      epochs=epochs,
                      verbose=2,
                      batch_size=batch_size_L2)
    ######################CNN Level 2################################
    if L2_model == 1:

        for i in range(0, number_of_classes_L1):
            print('Create Sub model of ', i)
            HDLTex.append(Sequential())
            HDLTex[i] = BuildModel.buildModel_CNN(word_index, embeddings_index,number_of_classes_L2[i],MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,1)
            HDLTex[i].fit(content_L2_Train[i], L2_Train[i],
                          validation_data=(content_L2_Test[i], L2_Test[i]),
                          epochs=epochs,
                          verbose=2,
                          batch_size=batch_size_L2)
    ######################RNN Level 2################################
    if L2_model == 2:
        for i in range(0, number_of_classes_L1):
            print('Create Sub model of ', i)
            HDLTex.append(Sequential())
            HDLTex[i] = BuildModel.buildModel_RNN(word_index, embeddings_index,number_of_classes_L2[i],MAX_SEQUENCE_LENGTH,EMBEDDING_DIM)
            HDLTex[i].fit(content_L2_Train[i], L2_Train[i],
                          validation_data=(content_L2_Test[i], L2_Test[i]),
                          epochs=epochs,
                          verbose=2,
                          batch_size=batch_size_L2)








