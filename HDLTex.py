import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN"
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from operator import itemgetter
import numpy
from keras.models import Sequential
import Data_helper
import BuildModel
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain
import keras.backend.tensorflow_backend as K
os.environ['THEANO_FLAGS'] = "device=gpu1"

if __name__ == "__main__":

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    MEMORY_MB_MAX = 1600000
    MAX_SEQUENCE_LENGTH = 500
    MAX_NB_WORDS = 55000
    EMBEDDING_DIM = 100
    batch_size_L1 = 256
    batch_size_L2 = 256

    L1_model =0 # 0 is DNN, 1 is CNN, and 2 is RNN
    L2_model =1 # 0 is DNN, 1 is CNN, and 2 is RNN

    np.set_printoptions(threshold=np.inf)
    '''
    location of input data 
    '''

    if RNN==1 or CNN==1:
        X_train, y_train, X_test, y_test, content_L2_Train, L2_Train, content_L2_Test, L2_Test, number_of_classes_L2,word_index, embeddings_index,number_of_classes_L1 =  \
            Data_helper.loadData_Tokenizer(MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)
   ######################CNN################################
    if L2_model == 1:
        HDLTex = []
        seed = 7
        numpy.random.seed(seed)
        model = Sequential()
        for i in range(0, number_of_classes_L1):
            HDLTex.append(Sequential())
        for i in range(0, number_of_classes_L1):
            HDLTex[i] = BuildModel.buildModel_CNN(word_index, embeddings_index,number_of_classes_L2[i],MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,1)
            HDLTex[i].fit(content_L2_Train[i], L2_Train[i],
                          validation_data=(content_L2_Test[i], L2_Test[i]),
                          epochs=50,
                          batch_size=batch_size_L2)


        model = BuildModel.buildModel_CNN(word_index, embeddings_index,number_of_classes_L1,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,1)
        model.fit(X_train, y_train[:,0],
                  validation_data=(X_test, y_test[:,0]),
                  epochs=50,
                  batch_size=batch_size_L1)

    if L2_model == 1:
        HDLTex = []
        seed = 7
        numpy.random.seed(seed)
        model = Sequential()
        for i in range(0, number_of_classes_L1):
            HDLTex.append(Sequential())
        for i in range(0, number_of_classes_L1):
            HDLTex[i] = BuildModel.buildModel_CNN(word_index, embeddings_index,number_of_classes_L2[i],MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,1)
            HDLTex[i].fit(content_L2_Train[i], L2_Train[i],
                          validation_data=(content_L2_Test[i], L2_Test[i]),
                          epochs=50,
                          batch_size=batch_size_L2)


        model = BuildModel.buildModel_CNN(word_index, embeddings_index,number_of_classes_L1,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,1)
        model.fit(X_train, y_train[:,0],
                  validation_data=(X_test, y_test[:,0]),
                  epochs=50,
                  batch_size=batch_size_L1)



    ######################RNN################################
    if L2_model == 1:
        seed = 7
        HDLTex = []
        numpy.random.seed(seed)
        model = Sequential()
        for i in range(0, number_of_classes_L1):
            HDLTex.append(Sequential())
        for i in range(0, number_of_classes_L1):
            print("Run Model %d for %d Lables",i,number_of_classes_L2[i])
            HDLTex[i] = BuildModel.buildModel_RNN(word_index, embeddings_index,number_of_classes_L2[i],MAX_SEQUENCE_LENGTH,EMBEDDING_DIM)
            HDLTex[i].fit(content_L2_Train[i], L2_Train[i],
                          validation_data=(content_L2_Test[i], L2_Test[i]),
                          epochs=5,
                          batch_size=batch_size_L2)
            score = HDLTex[i].evaluate(content_L2_Test[i], L2_Test[i], batch_size=batch_size_L2)
           # y_proba = HDLTex[i].predict_classes(content_L2_Test[i])
            #score = accuracy_score(L2_Test[i], y_proba)
            #print(y_proba)
            #print(score)

        model = BuildModel.buildModel_RNN(word_index, embeddings_index,number_of_classes_L1,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM)
        model.fit(X_train, y_train[:,0],
                  validation_data=(X_test, y_test[:,0]),
                  epochs=1,
                  batch_size=batch_size_L1)
        model.predict_classes(X_test)

    #######################DNN########################
    if L1_model == 1:
        X_train, y_train, X_test, y_test, content_L2_Train, L2_Train, content_L2_Test, L2_Test, number_of_classes_L2 = Data_helper.loadData(fname,fnamek,fnameL2,number_of_classes_L1)
        print("Loading Data is Done")
        seed = 7
        numpy.random.seed(seed)
        HDLTex = []
        print(number_of_classes_L2)
        model = BuildModel.buildModel_DNN(X_train.shape[1], number_of_classes_L1,8, 64, dropout=0.25)
        model.fit(X_train, y_train[:,0],
                  validation_data=(X_test, y_test[:,0]),
                  epochs=2000,
                  batch_size=batch_size_L1)

        for i in range(0, number_of_classes_L1):
            HDLTex.append(Sequential())


            HDLTex[i] = BuildModel.buildModel_DNN(content_L2_Train[i].shape[1], number_of_classes_L2[i],2, 1024, dropout=0.5)
            HDLTex[i].fit(content_L2_Train[i], L2_Train[i],
                      validation_data=(content_L2_Test[i], L2_Test[i]),
                      epochs=20,
                      batch_size=batch_size_L2)

        print(X_train.shape)








