import re
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import WOS_input as WOS
import Download_Glove as GloVe
import numpy as np
import os


''' Location of the dataset'''
path_WOS = WOS.download_and_extract()
GLOVE_DIR = GloVe.download_and_extract()
print(GLOVE_DIR)

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def text_cleaner(text):
    text = text.replace(".", "")
    text = text.replace("[", " ")
    text = text.replace(",", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("\"", "")
    text = text.replace("-", "")
    text = text.replace("=", "")
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    return text.lower()


def loadData_Tokenizer(MAX_NB_WORDS,MAX_SEQUENCE_LENGTH):

    
    fname = os.path.join(path_WOS,"WebOfScience/WOS5736/X.txt")
    fnamek = os.path.join(path_WOS,"WebOfScience/WOS5736/YL1.txt")
    fnameL2 = os.path.join(path_WOS,"WebOfScience/WOS5736/YL2.txt")

    with open(fname) as f:
        content = f.readlines()
        content = [clean_str(x) for x in content]
    content = np.array(content)
    with open(fnamek) as fk:
        contentk = fk.readlines()
    contentk = [x.strip() for x in contentk]
    with open(fnameL2) as fk:
        contentL2 = fk.readlines()
        contentL2 = [x.strip() for x in contentL2]
    Label = np.matrix(contentk, dtype=int)
    Label = np.transpose(Label)
    number_of_classes_L1 = np.max(Label)+1 #number of classes in Level 1

    Label_L2 = np.matrix(contentL2, dtype=int)
    Label_L2 = np.transpose(Label_L2)
    np.random.seed(7)

    Label = np.column_stack((Label, Label_L2))

    number_of_classes_L2 = np.zeros(number_of_classes_L1,dtype=int) #number of classes in Level 2 that is 1D array with size of (number of classes in level one,1)


    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(content)
    sequences = tokenizer.texts_to_sequences(content)
    word_index = tokenizer.word_index

    print('Found %s unique tokens.' % len(word_index))

    content = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    indices = np.arange(content.shape[0])
    np.random.shuffle(indices)
    content = content[indices]
    Label = Label[indices]
    print(content.shape)

    X_train, X_test, y_train, y_test = train_test_split(content, Label, test_size=0.2, random_state=0)

    L2_Train = []
    L2_Test = []
    content_L2_Train = []
    content_L2_Test = []
    '''
    crewate #L1 number of train and test sample for level two of Hierarchical Deep Learning models
    '''
    for i in range(0, number_of_classes_L1):
        L2_Train.append([])
        L2_Test.append([])
        content_L2_Train.append([])
        content_L2_Test.append([])

        X_train = np.array(X_train)
        X_test= np.array(X_test)
    for i in range(0, X_train.shape[0]):
        L2_Train[y_train[i, 0]].append(y_train[i, 1])
        number_of_classes_L2[y_train[i, 0]] = max(number_of_classes_L2[y_train[i, 0]],(y_train[i, 1]+1))
        content_L2_Train[y_train[i, 0]].append(X_train[i])

    for i in range(0, X_test.shape[0]):
        L2_Test[y_test[i, 0]].append(y_test[i, 1])
        content_L2_Test[y_test[i, 0]].append(X_test[i])

    for i in range(0, number_of_classes_L1):
        L2_Train[i] = np.array(L2_Train[i])
        L2_Test[i] = np.array(L2_Test[i])
        content_L2_Train[i] = np.array(content_L2_Train[i])
        content_L2_Test[i] = np.array(content_L2_Test[i])

    embeddings_index = {}
    '''
    For CNN and RNN, we used the text vector-space models using $100$ dimensions as described in Glove. A vector-space model is a mathematical mapping of the word space
    '''
    Glove_path = os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')
    print(Glove_path)
    f = open(Glove_path, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            print("Warnning"+str(values)+" in" + str(line))
        embeddings_index[word] = coefs
    f.close()
    print('Total %s word vectors.' % len(embeddings_index))
    return (X_train, y_train, X_test, y_test, content_L2_Train, L2_Train, content_L2_Test, L2_Test, number_of_classes_L2,word_index,embeddings_index,number_of_classes_L1)





def loadData():
    WOS.download_and_extract()
    fname = os.path.join(path_WOS,"WebOfScience/WOS5736/X.txt")
    fnamek = os.path.join(path_WOS,"WebOfScience/WOS5736/YL1.txt")
    fnameL2 = os.path.join(path_WOS,"WebOfScience/WOS5736/YL2.txt")
    with open(fname) as f:
        content = f.readlines()
        content = [text_cleaner(x) for x in content]
    with open(fnamek) as fk:
        contentk = fk.readlines()
    contentk = [x.strip() for x in contentk]
    with open(fnameL2) as fk:
        contentL2 = fk.readlines()
        contentL2 = [x.strip() for x in contentL2]
    Label = np.matrix(contentk, dtype=int)
    Label = np.transpose(Label)
    number_of_classes_L1 = np.max(Label)+1  # number of classes in Level 1

    Label_L2 = np.matrix(contentL2, dtype=int)
    Label_L2 = np.transpose(Label_L2)
    np.random.seed(7)
    print(Label.shape)
    print(Label_L2.shape)
    Label = np.column_stack((Label, Label_L2))

    number_of_classes_L2 = np.zeros(number_of_classes_L1,dtype=int)

    X_train, X_test, y_train, y_test  = train_test_split(content, Label, test_size=0.2,random_state= 0)

    vectorizer_x = CountVectorizer()
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()

    L2_Train = []
    L2_Test = []
    content_L2_Train = []
    content_L2_Test = []

    for i in range(0, number_of_classes_L1):
        L2_Train.append([])
        L2_Test.append([])
        content_L2_Train.append([])
        content_L2_Test.append([])


    for i in range(0, X_train.shape[0]):
        L2_Train[y_train[i, 0]].append(y_train[i, 1])
        number_of_classes_L2[y_train[i, 0]] = max(number_of_classes_L2[y_train[i, 0]],(y_train[i, 1]+1))
        content_L2_Train[y_train[i, 0]].append(X_train[i])

    for i in range(0, X_test.shape[0]):
        L2_Test[y_test[i, 0]].append(y_test[i, 1])
        content_L2_Test[y_test[i, 0]].append(X_test[i])

    for i in range(0, number_of_classes_L1):
        L2_Train[i] = np.array(L2_Train[i])
        L2_Test[i] = np.array(L2_Test[i])
        content_L2_Train[i] = np.array(content_L2_Train[i])
        content_L2_Test[i] = np.array(content_L2_Test[i])
    return (X_train,y_train,X_test,y_test,content_L2_Train,L2_Train,content_L2_Test,L2_Test,number_of_classes_L2)
