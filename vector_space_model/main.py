from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import nltk
from nltk.stem import PorterStemmer

reload(sys)
sys.setdefaultencoding('utf8')

def getFiles():
    files = []
    for i in range(8):
        files.append(open('../texto' + str(i + 1) + '.txt'))

    return files

def getOutTfFiles():
    files = []
    for i in range(8):
        files.append(open('../texto' + str(i + 1) + '_tf_out.txt', 'w'))

    return files

def getOutTfIdfFiles():
    files = []
    for i in range(8):
        files.append(open('../texto' + str(i + 1) + '_tfidf_out.txt', 'w'))

    return files


def stemmer(files):
    stemmed_texts = []
    for file in files:
        words = nltk.word_tokenize(file.read())

        stem_text = ""

        for w in words:
            stem_text = stem_text + " " + ps.stem(w)

        stemmed_texts.append(stem_text)

    return stemmed_texts


def write_files(feature_names, files, weight):
    index = 0

    for file in files:

        for fn in feature_names:
            file.write(fn + "\t")

        file.write("\n\n")

        for f in weight[index]:
            file.write(str(f) + "\t")

        index = index + 1


files = getFiles()


ps = PorterStemmer()
stemmed_texts = stemmer(files)


vectorizer = CountVectorizer(stop_words='english')
tfidf_vectorizer = TfidfVectorizer(stop_words='english')


X = vectorizer.fit_transform(stemmed_texts)
Y = tfidf_vectorizer.fit_transform(stemmed_texts)

tf_files = getOutTfFiles()
tfidf_files = getOutTfIdfFiles()

index = 0
feature_names = vectorizer.get_feature_names()
tf = X.toarray()
tfidf = Y.toarray()




write_files(feature_names, tf_files, tf)
write_files(feature_names, tfidf_files, tfidf)


inverted_file = open('../inverted.txt' , 'w')
tf_transpose = tf.transpose()

index = 0

for fn in feature_names:
    inverted_file.write(fn + ' -> ')

    for i in range(len(tf_transpose[0])):
        inverted_file.write('\t\ttexto' + str(i + 1) + ': ' + str(tf_transpose[index][i]))

    inverted_file.write('\n')

    index = index + 1
