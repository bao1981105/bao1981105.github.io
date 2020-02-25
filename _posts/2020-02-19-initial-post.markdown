---
title:  "Decode Bengali: initial post"
date:   2020-02-18 
tags: [convolutional neural network, Kaggle]
header:
  image: "images/bengali.png"
  caption: "Photo Credit: hindustantimes"

excerpt: "Bengali.AI Handwritten Grapheme Classification competition"
---
### Background
Bengali is the 5th most spoken language in the world with hundreds of million of speakers. It’s the official language of Bangladesh and the second most spoken language in India. Considering its reach, there’s significant business and educational interest in developing AI that can optically recognize images of the language handwritten. This challenge hopes to improve on approaches to Bengali recognition.

Optical character recognition is particularly challenging for Bengali. While Bengali has 49 letters (to be more specific 11 vowels and 38 consonants) in its alphabet, there are also 18 potential diacritics, or accents. This means that there are many more graphemes, or the smallest units in a written language. The added complexity results in ~13,000 different grapheme variations (compared to English’s 250 graphemic units).

Bangladesh-based non-profit Bengali.AI is focused on helping to solve this problem. They build and release crowdsourced, metadata-rich datasets and open source them through research competitions. Through this work, Bengali.AI hopes to democratize and accelerate research in Bengali language technologies and to promote machine learning education.

We have access to the image of a handwritten Bengali grapheme and separately classify three constituent elements in the image: grapheme root, vowel diacritics, and consonant diacritics. There are 168 different grapheme roots, 11 vowel diacritics, and 7 consonant diacritics.

### EDA
In the training set, we have a label for each of three components of graphemes and the outputs from our EDA code list the count of root, consonant diacritics, and vowel diacritics, from the most used to the least used.

![alt text](https://i.ibb.co/KW4JwdQ/root.png "root count")
![alt text](https://i.ibb.co/jHZ5JdK/conso.png "consonant diacritics count")
![alt text](https://i.ibb.co/DGTF8m0/vowel.png "vowel diacritics count")

### Model
In order to classify three constituent elements in the image of a handwritten Bengali grapheme: grapheme root, vowel diacritics, and consonant diacritics, we started with a model using multiple convolutional layers and fully connected layers and evaluated the performance of our model.        
Credit to [Bengali Graphemes: Starter EDA+ Multi Output CNN kernel](https://www.kaggle.com/kaushal2896/bengali-graphemes-starter-eda-multi-output-cnn)

```python
# all images are resized to 64*64
inputs = Input(shape = (64, 64, 1))

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1))(inputs)
model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = Dropout(rate=0.3)(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = Dropout(rate=0.3)(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = Dropout(rate=0.3)(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=256, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = Dropout(rate=0.3)(model)

model = Flatten()(model)
model = Dense(1024, activation = "relu")(model)
model = Dropout(rate=0.3)(model)
dense = Dense(512, activation = "relu")(model)

head_root = Dense(168, activation = 'softmax')(dense)
head_vowel = Dense(11, activation = 'softmax')(dense)
head_consonant = Dense(7, activation = 'softmax')(dense)

model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

* We first create three Conv2D layers with 32 filters, each 3 by 3, strides of 1 in both directions, and “same” padding, and applied the ReLU activation function to its outputs, and batch normalization with a momentum of 0.15. Then we added a max pooling layer to subsample the input image in order to reduce the computational load, the memory usage, and the number of parameters and thereby limiting the risk of overfitting. We chose to go with a 2 by 2 max pooling. We also included a dropout layer with a dropout rate of 30%.
* Then, we repeated the above structure 3 more times, each time doubling the number of filters as we approach the output layer while max pooling to reduce data loss.
* Lastly, we flattened the outputs from convolutional layers and used the fully connected network which is composed of two hidden layers and three output layers. Again, we added a dropout layer with a rate of 30% after the first hidden layer for the purpose of avoiding risk of overfitting.

### Performance
![alt text](https://i.ibb.co/4Sbvrqx/Training-Dataset-3-Loss.png "Logo Title Text 1")
![alt text](https://i.ibb.co/w0XBmYY/Training-Dataset-3-Accuracy.png "Logo Title Text 1")
* Validation accuracy for root reached 94.85%
* Validation accuracy for vowel diacritic reached 98.11%
* Validation accuracy for consonant diacritic reached 98.81% 

Since the image files are quite large, the model was trained on four files. Dataset 3 is the last file containing the images. We can see that losses continue to decrease for both training set and test set. Accuracy scores for root, vowel, and consonant are slowly increasing. No overfitting is observed. 

### Shortcomings
* We should consider different architectures such as ResNet and go through various possible combinations of layers to evaluate and compare different models.
* We should take cross validation into consideration for tuning the hyperparameters.
* Different dropout rates and filter size should be assessed for better performance of the model.
* Since the training process is lengthy, we could consider increasing the number of strides to reduce model’s computational complexity.
* Furthermore, when conducting image processing, we could also use the original image size without having to resize the grid to 64 by 64.

### Next Steps
We will evaluate more architectures and consider different ways of constructing the convolutional layers of our model. More hyperparameters such as filter number, dropout rates, number of strides and image resize should be carefully considered and further evaluated to reduce computation complexity and increase model performance. We will also try to ensemble the best models or try to use transfer learning (if available) to boost model performance.  

Stay tuned.

### Team
We are NNPlayer, a group of students from Brown University. We are one of the team participating the Bengali.AI Handwritten Grapheme Classification competition [kaggle.com/c/bengaliai-cv19](https://www.kaggle.com/c/bengaliai-cv19/).

You can find out more great work done by our team member at GitHub:
 
[Hannah Han](https://github.com/bao1981105) 

[Daisy Du](https://github.com/daisydu97) 

[Justin Tian](https://github.com/quicklearnerjustin)

You can find out the starter code for our project at [github.com/bao1981105/decode-bengali](https://github.com/bao1981105/decode-bengali)
