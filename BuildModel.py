from keras.models import Sequential
from keras.models import Model
import numpy as np
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional,SimpleRNN
'''
buildModel_DNN(nFeatures, nClasses, nLayers=3,Numberof_NOde=100, dropout=0.5)
Build Deep neural networks Model for text classification
Shape is input feature space
nClasses is number of classes
nLayers is number of hidden Layer
Number_Node is number of unit in each hidden layer
dropout is dropout value for solving overfitting problem
'''
def buildModel_DNN(Shape, nClasses, nLayers=3,Number_Node=100, dropout=0.5):
    model = Sequential()
    model.add(Dense(Number_Node, input_dim=Shape))
    model.add(Dropout(dropout))
    for i in range(0,nLayers):
        model.add(Dense(Number_Node, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='RMSprop',
                  metrics=['accuracy'])

    return model

'''
def buildModel_RNN(word_index, embeddings_index, nClasses, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
word_index in word index , 
embeddings_index is embeddings index, look at data_helper.py 
nClasses is number of classes, 
MAX_SEQUENCE_LENGTH is maximum lenght of text sequences, 
EMBEDDING_DIM is an int value for dimention of word embedding look at data_helper.py 
'''
def buildModel_RNN(word_index, embeddings_index, nClasses, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
    model = Sequential()
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))
    model.add(GRU(100,dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(nClasses, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    return model
'''
def buildModel_CNN(word_index,embeddings_index,nClasses,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,Complexity=0):
word_index in word index , 
embeddings_index is embeddings index, look at data_helper.py 
nClasses is number of classes, 
MAX_SEQUENCE_LENGTH is maximum lenght of text sequences, 
EMBEDDING_DIM is an int value for dimention of word embedding look at data_helper.py 
Complexity we have two different CNN model as follows 
Complexity=0 is simple CNN with 3 hidden layer
Complexity=2 is more complex model of CNN with filter_length of [3, 4, 5, 6, 7]
'''
def buildModel_CNN(word_index,embeddings_index,nClasses,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,Complexity=1):
    if Complexity==0:
        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=True)
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        embedded_sequences = embedding_layer(sequence_input)

        x = Conv1D(256, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(256, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(256, 5, activation='relu')(x)
        x = MaxPooling1D(35)(x)  # global max pooling
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        preds = Dense(nClasses, activation='softmax')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
    else:
        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=True)

        convs = []
        filter_sizes = [3, 4, 5, 6, 7]

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        for fsz in filter_sizes:
            l_conv = Conv1D(128, filter_length=fsz, activation='relu')(embedded_sequences)
            l_pool = MaxPooling1D(5)(l_conv)
            convs.append(l_pool)

        l_merge = Merge(mode='concat', concat_axis=1)(convs)
        l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)
        l_pool1 = MaxPooling1D(5)(l_cov1)
        l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
        l_pool2 = MaxPooling1D(30)(l_cov2)
        l_flat = Flatten()(l_pool2)
        l_dense = Dense(128, activation='relu')(l_flat)
        preds = Dense(nClasses, activation='softmax')(l_dense)
        model = Model(sequence_input, preds)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

    return model
