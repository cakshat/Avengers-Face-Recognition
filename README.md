# Avengers-Face-Recognition
This is a neural network I trained to recognize the faces of 6 Avengers characters: (Clint Barton, Tony Stark, Steve Rogers, Bruce Banner, Natasha Romanoff, and Thor)...sorry Nick Fury :(

# Compiling data
Data was used from Kaggle. There are 6 folders of 6 people (Clint Barton, Tony Stark, Steve Rogers, Bruce Banner, Natasha Romanoff, and Thor) each having close to 100 images. These images were then preprocessed and normalized. Normalisation modifies the image values so that their mean is 0 and standard deviation is 1. It serves to ensure that the model consistently learns the features of the images.

# Model
The structure of the neural network model_c is as follows:

1. Input layer: Shape -> (200, 200, 1), it accepts a 200x200 grayscale (channels=1) image as input.
2. Convolutional layers:The network is made up of 6 convolutional layers, each of which has 64, 128, 256, 512, 512, and 512 filters. All of the convolutional layers have 3x3 kernels. All of the convolutional layers have the same padding, which means that each layer's output has the same shape as its input. All of the convolutional layers use the ReLU function as their activation function.
3. Max pooling layers: There are 5 max-pooling layers in the network, and each one has a kernel size of 2x2 and a stride = 2. The max pooling layers make the feature maps that the convolutional layers produce smaller.
4. Dense layers: The network has 2 dense layers, with 1024 and 6 neurons, respectively. The activation function for the first dense layer is ReLU, and the activation function for the second dense layer is softmax. The softmax activation function is used for classification tasks, where the output of the network is a probability distribution over the possible classes.

The network uses L2 regularization with a regularization coefficient of 0.001.

The network is compiled using SGD optimizer and categorical cross-entropy loss function. Stochastic gradient descent (SGD) is a simple yet powerful optimization algorithm for training machine learning models. It works by iteratively updating the model's parameters in the direction of the negative gradient of the loss function.

SGD worked better than Adam and RMSprop hence it was used in this case.

The learning rate and momentum are the two most important hyperparameters for the SGD optimizer. The learning rate determines how big the changes to the properties of the model are. The motion makes it harder for the optimizer to get stuck in a local minimum. The category cross-entropy loss function is a loss function that is used to classify things.

# Evaluation

I experimented with different neural network structures (simple and complex) and also with different values of hyperparameters like learning rate, momentum and decay. As mentioned earlier SGD optimizer gave the best results compared to Adam and RMSprop (Adam and RMSprop showed accuracy saturation at around 18-20%). The model was trained for BATCH_SIZE = 16 and EPOCHS = 30. After experimenting with different batch sizes and epochs these values were found to give high accuracy. 
1. Test loss: 1.72
2. Overall accuracy on test data: 67.8%
   
After evaluating accuracy per class for the entire dataset the class-wise accuracies were as follows:
1. steve_rogers (Chris Evans): 84.88%
2. tony_stark (Robert Downey Jr.): 67.0%
3. clint_barton (Jeremy Renner): 94.0%
4. bruce_banner (Mark Ruffalo): 80.0%
5. thor (Chris Hemsworth): 89.0%
6. Natasha_Romanoff (Scarlett Johansson): 94.0%

Thus, overall on the test set and class-wise on the entire dataset, it gives not bad results.

# Further improvements
1. Much more vast dataset would help train the model effectively. The split among train, validation, and test was around 468, 59 & 59 respectively. Hence, it was a bit difficult to get an idea of the actual accuracy of each class as in the test set each class had only about 8-10 images.
2. Further experimentation with neural network architecture, regularization, activation functions, and hyperparameter could give more accurate model.
