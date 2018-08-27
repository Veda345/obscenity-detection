# Obscenity Detection
DNN model for text classification.


Update paths and params in **config.txt**:
- `neg` - path to negative dataset (in case of obscenity detection - normal posts)
- `pos` - path to positive dataset (in case of obscenity detection - obscene posts)
- `neg_obfv` - path to obfuscated negative dataset (in case of obscenity detection - obfuscated normal posts)
- `pos_obf` - path to obfuscated positive dataset (in case of obscenity detection - obfuscated obscene posts)
- `test_neg` - path to test negative dataset (in case of obscenity detection - validation normal posts)
- `test_pos` - path to test positive dataset (in case of obscenity detection - validation obscene posts)
- `w2v` - path to word2vec model
- `train_with_obf` - set true if you want to train on both initial and obfuscated data
- `epochs` - number of training epochs

You can obtain all required datasets with **transform_data.py**.
Use **transformation_config.txt** to set params for **transform_data.py**. 
- `vocab` - path to vocabulary
- `vocab_test_roots` - list of roots (e.g. ху,еб)
- `init_data` - path to dataset

To **divide initial vocab** into train and test parts use `separate_vocab(vocab_path, vocab_roots)`. 
Test part of vocabulary will contain all words with provided roots. 
The result files for train and test vocabulary will be named as initial_vocabulary with suffixes *«_train»* and *«_test»* 
respectively. 

To **preprocess initial dataset** use `preprocess_data(init_file_path)`. 
The result file will be saved in the same directory with *«_preprocessed»* suffix and will contain lower-case posts 
without <br> tags.

To **divide initial dataset** into negative and positive parts use `label_data_with_vocab(vocab_path, processed_file_path)`. 
The result files will contain posts for negative and positive classes (e.g. normal and obscene posts or obscene train and 
obscene test posts) and will be named as initial file with suffixes *«_negative»* and *«_positive»* respectively. 
You can use full vocabulary or train and test parts of vocabulary to repeat experiments todo .

To **obfuscate dataset** use `obfuscate_data(init_file_path)`. The result file will be saved in the same directory with 
*«_obf»* suffix and will contain posts obfuscated in following ways:
* removal of characters;
* insertion of characters;
* changing characters to similar digits;
* changing characters to homoglyphs;
* duplication of characters;

You can **train w2v model** with **gensim_w2v.py**. 
Provide path to full dataset in **transformation_config.txt**. The result model will be trained as `Word2Vec(sentences, size=300, window=5, min_count=20, iter=10)` and saved in the same directory as dataset with suffix *«_w2v_model»*. 


## DNN model
![model scheme](todo)
Model contains two branches working on word and character levels respectively. 

On **character level** we use one-hot encoding with 102 valid characters including Cyrillic and Latin alphabets, punctuation and whitespace. The length of the user-generated post is limited to 1000 characters: the characters after the 1000-th are ignored. Thus, each post (sequence of its symbols) is associated with a matrix of size 1000x102, which is then fed to the input of a sequence of convolutional and max-pooling layers.

On **word level** we use pre-trained Word2Vec model to create embeddings for neural network. Embeddings map the original sequence of words to a sequence of vectors of dimension 300. The length of the user-generated post is limited to 20 words. Thus, each post is mapped to a matrix of size 20x300.

Description of the model:
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input (InputLayer)               (None, 1000, 102)     0                                            
____________________________________________________________________________________________________
conv1d_1 (Conv1D)                (None, 991, 128)      130688      input[0][0]                      
____________________________________________________________________________________________________
max_pooling1d_1 (MaxPooling1D)   (None, 330, 128)      0           conv1d_1[0][0]                   
____________________________________________________________________________________________________
words_input (InputLayer)         (None, 20)            0                                            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 42240)         0           max_pooling1d_1[0][0]            
____________________________________________________________________________________________________
emb_layer (Embedding)            (None, 20, 300)       343313100   words_input[0][0]                
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 32)            1351712     flatten_1[0][0]                  
____________________________________________________________________________________________________
flatten_2 (Flatten)              (None, 6000)          0           emb_layer[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 32)            0           dense_1[0][0]                    
____________________________________________________________________________________________________
densed_emb_layer (Dense)         (None, 1024)          6145024     flatten_2[0][0]                  
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 1056)          0           dropout_1[0][0]                  
                                                                   densed_emb_layer[0][0]           
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 2)             2114        concatenate_1[0][0]              
====================================================================================================
Total params: 350,942,638
Trainable params: 7,629,538
Non-trainable params: 343,313,100
____________________________________________________________________________________________________

```
