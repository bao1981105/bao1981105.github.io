---
title:  "Decode Bengali: final post"
date:   2020-03-04
tags: [Kaggle]
header:
  image: "images/bengali.png"
  caption: "Photo Credit: hindustantimes"

excerpt: "Bengali.AI Handwritten Grapheme Classification competition"
---

The first model, a plain vanilla CNN model achieves a score of 0.9516. Its architecture is shown below. 
![alt text](https://i.ibb.co/19XdnQR/architecture.png "baseline")

***   
To boost the performance of our model, we tried several newer models with more complex architecture. After testing out that our models can train, we used the virtual machine (GPU: NVIDIA Tesla P100) on GCP to do the heavy lifting for us and let it train for hours. After the training is done, we will save the model weights and load it to Kaggle for inference. 

### ResNet
![alt text](https://i.ibb.co/3dMJxLT/r1.png)

ResNet (Residual Neural Network) is an artificial neural network which builds on a convolutional neural network. It uses skip connections to build a shortcut to send the input past several convolutional layers.  

**Why ResNet**
In image classification problems, it is common to use CNN to solve these problems. Especially, as the depth of the network and the complexity of the model increase, the network performs better and predicts more accurately. However, as the number of layers increases, another problem might arise. If the network is too deep, the gradients from where the loss function is calculated easily shrink to zero after several applications of the chain rule. There would be nearly no update on the value and therefore, no learning is being performed. ResNet could solve this vanishing gradient problem as it reuses activations from a previous layer until the adjacent layer learns its weights. Also, after solving the problem of vanishing gradients, ResNet will speed up the training process as well.

**Construction**
Now, let us take a look at the Residual Neural Network we trained to classify the Bengali graphemes. We read the data and resized them as we did before. At first, we constructed 3 layers, in each of which there are 32 feature maps and the kernel has a size of 3. We also choose zero padding to maintain the size of the inputs and the ReLu activation. Then, after batch normalization and max pooling, we built another layer, which includes 32 feature maps as before but with the size of filter equal to 5. Then, we did dropout with the dropout rate equal to 0.3. That was how we did regularization. 
Then, we started to construct layers of 64 feature maps. The first two of such layers had 3 by 3 filters, while the third layer had 5 by 5 filters. We also added zero padding and ReLu activation methods. At the end of the third layer, we did dropout again.
After building the basic layers, we extended the network with the residual unit. 
![alt text](https://i.ibb.co/zfB5CVM/r2.png)

In the first unit, we had two convolutional layers. Each layer had 128 feature maps and each filter had a size of 3. But one thing we need to notice is that the first layer had the stride equal to 2. Thus, we had to make a convolutional layer in the skip connection to match the size of the output after the skip step and the size of the output after the two layers. Then, we continued to construct 4 similar units. The only difference was that all the strides were set to one and the skip connection was set to the identity function. 
Later, we constructed 6 more units, similar to what we just did before. In the first unit, the first layer was constructed and it contained two convolutional layers. Both of the layers had 256 feature maps with 3 by 3 filters. It was just that the first layer’s stride was 2, while the second filter had stride equal to 1. Then, we made a convolutional layer in the skip connection to match the size of the output after the skip step and the size of the output after the two layers. We also made 5 more similar units. They were the same as the first unit except that all the strides were set to one and the skip connection function was equal to identity. 

To make the network computationally stronger, we constructed 7 more units, almost repeating the previous process. In the first unit, the first layer was constructed and it contained two convolutional layers. Both of the layers had 512 feature maps with 3 by 3 filters. It was just that the first layer’s stride was 2, while the second filter had stride equal to 1. Then, we made a convolutional layer in the skip connection to match the size of the output after the skip step and the size of the output after the two layers. We continued to produce 6 more similar units. They were the same as the first unit except that all the strides were set to one and the skip connection function was equal to identity. 
In summary, we constructed 5+6+7=18 residual units and we had 18X2=36 convolutional layers in total.
After flattening, the overall LB score of the ResNet is 0.957.

![alt text](https://i.ibb.co/cgmhskt/r3.png)

***
### EfficientNetB3

LB Score: 0.9681   

We know that usually a CNN with a larger number of layers can hold richer details about the image and therefore is usually more accurate than a model with a fewer number of layers. Using wider CNN and an image with larger input size could lead to higher accuracy, but the gain in accuracy tends to saturate after certain threshold. 

With limited computing resources, we want to strike a balance between model depth, model width, and image resolution and build a model that achieves a satisfactory score without using too much computing power. 

Image processing: simple resize, shrink the image from 137 * 236 to 95 * 165 （preserves ratio instead of crop center), we have not used data augmentation this time. 

```python
def resize_image(img, w, h, new_w, new_h):
    img = 255 - img
    img = (img * (255.0 / img.max())).astype(np.uint8)
    img = img.reshape(h, w)
    image_resized = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)
    return image_resized    
```

Training process: we use stratified k fold, and allocate 1/6 of training images to to validation set and 5/6 to training set, train for 50 epochs, and a batch_size of 56. If after 5 consecutive epochs, validation loss for the root is still higher than previous lowest level, reduce the learning rate by 25% until it reaches the minimum of 1e-5. 


```python
input = Input(shape = input_shape)

x_model = efn.EfficientNetB3(weights = 'imagenet', include_top = False, input_tensor = input, pooling = None, classes = None)

for layer in x_model.layers:
	layer.trainable = True

lambda_layer = Lambda(generalized_mean_pool_2d)
lambda_layer.trainable_weights.extend([gm_exp])
x = lambda_layer(x_model.output)
    
grapheme_root = Dense(168, activation = 'softmax', name = 'root')(x)
vowel_diacritic = Dense(11, activation = 'softmax', name = 'vowel')(x)
consonant_diacritic = Dense(7, activation = 'softmax', name = 'consonant')(x)


model = Model(inputs = x_model.input, outputs = [grapheme_root, vowel_diacritic, consonant_diacritic])
```
```python
model.compile(optimizer = Adam(lr = 0.00016),
                loss = {'root': 'categorical_crossentropy',
                        'vowel': 'categorical_crossentropy',
                        'consonant': 'categorical_crossentropy'},
                loss_weights = {'root': 0.40,        
                                'vowel': 0.30,
                                'consonant': 0.30},
                metrics = {'root': ['accuracy', tf.keras.metrics.Recall()],
                            'vowel': ['accuracy', tf.keras.metrics.Recall()],
                            'consonant': ['accuracy', tf.keras.metrics.Recall()] })
```

The model architecture was borrowed from this [kernel](https://www.kaggle.com/nxrprime/keras-efficientnet-b3-with-image-preprocessing).

***
### Ensemble

We ensemble three models from above and one model from this [kernel](https://www.kaggle.com/h030162/version1-0-9696) and achieved a slightly higher score of 0.9702. 

```python
preds1 = model1.predict_generator(data_generator_test)
preds2 = model2.predict_generator(data_generator_test)
preds3 = model3.predict_generator(data_generator_test)
preds4 = model4.predict_generator(data_generator_test)
for i, image_id in zip(range(len(test_files)), test_files):
        for subi, col in zip(range(len(preds1)), tgt_cols):
            sub_preds1 = preds1[subi]
            sub_preds2 = preds2[subi]
            sub_preds3 = preds3[subi]
            sub_preds4 = preds4[subi]

            row_ids.append(str(image_id)+'_'+col)
            sub_pred_value = np.argmax((sub_preds1[i] + sub_preds2[i] + sub_preds3[i] + preds4[subi][i]) / 4)
            targets.append(sub_pred_value)
```

Current ranking: 330/1861, top 18%.

Next steps: we are still tuning our DenseNet-121 model with image input of 224 * 224 * 3. Will try out some newest state-of-the-art data augmentation techniques used by top team: cutout, cutmix, etc.      
