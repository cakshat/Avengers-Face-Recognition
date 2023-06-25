# Avengers-Face-Recognition
This is a neural network trained to recognize faces of 6 Avengers characters: (Clint Barton, Tony Stark, Steve Rogers, Bruce Banner, Natasha Romanoff, and Thor)...sorry Nick Fury :(

# Compiling data
Data was used from Kaggle. There are 6 folders of 6 people (Clint Barton, Tony Stark, Steve Rogers, Bruce Banner, Natasha Romanoff, Thor) each having close to 100 images. These images were then preprocessed and normalized. Normalization is a process that adjusts the values of the images so that they have a mean of 0 and a standard deviation of 1. This is important for deep learning models, as it helps to ensure that the model learns the features of the images in a consistent way.

The caer.preprocess_from_dir() function is used to preprocess a dataset of images from a directory.
The to_categorical() function in the Keras library is used to convert a class vector (integers) to a binary class matrix. This is a common preprocessing step for deep learning models that use categorical cross-entropy as their loss function.

# Model
The structure of the neural network model_c is as follows:

1. Input layer: The input layer has shape (200, 200, 1), which means that it accepts a 200x200 grayscale (channels=1) image as input.
2. Convolutional layers: The network has 6 convolutional layers, each with 64, 128, 256, 512, 512, and 512 filters, respectively. The kernel size for all of the convolutional layers is 3x3. The padding for all of the convolutional layers is the same, which means that the output of each convolutional layer has the same shape as the input. The activation function for all of the convolutional layers is ReLU.
3. Max pooling layers: The network has 5 max-pooling layers, each with a kernel size of 2x2 and a stride of 2. The max pooling layers reduce the size of the feature maps output by the convolutional layers.
4. Dense layers: The network has 2 dense layers, each with 1024 and 6 neurons, respectively. The activation function for the first dense layer is ReLU, and the activation function for the second dense layer is softmax. The softmax activation function is used for classification tasks, where the output of the network is a probability distribution over the possible classes.
   
In total, the network has 13 layers. The first 6 layers are convolutional layers, the next 5 layers are max-pooling layers, and the last 2 layers are dense layers. The network has a total of 1,251,848 parameters.

The network is configured to use L2 regularization with a regularization coefficient of 0.001. L2 regularization is a technique that helps to prevent overfitting by adding a penalty to the loss function that is proportional to the square of the weights of the network.

The network is compiled using the SGD optimizer and the categorical cross-entropy loss function. Stochastic gradient descent (SGD) is a simple yet powerful optimization algorithm for training machine learning models. It works by iteratively updating the model's parameters in the direction of the negative gradient of the loss function.

SGD worked better than Adam and RMSprop hence it was used in this case.

The SGD optimizer has two main hyperparameters: the learning rate and the momentum. The learning rate controls how large the updates to the model's parameters are. The momentum helps to prevent the optimizer from getting stuck in local minima. The categorical cross-entropy loss function is a loss function that is used for classification tasks.

# Evaluation

The model was trained for BATCH_SIZE = 16 and EPOCHS = 30. After experimenting with different batch sizes and epochs these values were found to give high accuracy. 
1. Test loss: 1.72
2. Overall accuracy on test data: 67.8%
   
After evaluating accuracy per class for the entire dataset the class-wise accuracies were as follows:
1. steve_rogers (Chris Evans): 84.88%
2. tony_stark (Robert Downey Jr.): 67.0%
3. clint_barton (Jeremy Renner): 94.0%
4. bruce_banner (Mark Ruffalo): 80.0%
5. thor (Chris Hemsworth): 89.0%
6. Natasha_Romanoff (Scarlett Johansson): 94.0%
