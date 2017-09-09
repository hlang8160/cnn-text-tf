import numpy as np
from tensorflow.contrib import learn
max_document_length = 4
x_text = [
    'i love you',
    'me too',
    'you love me too'
]
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
print x
print len(vocab_processor.vocabulary_)
