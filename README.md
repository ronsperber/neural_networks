## Basic Neural Network implementation from scratch using numpy



### Setup

1. Create the conda environment
```bash
conda env create -f environment.yml
```

2. Activate the conda environment
```bash
conda activate neural_networks
```

3. Install the package in editable mode
```bash
pip install -e .
```

#### Modules (all in my_nn)
1.  `activations.py` : contains all activation functions used. All part of a class called Activation. Note that softmax is not directly implemented as an activation function. Multiclass classification has a loss function that combines softmax with categorical cross entropy
   1. `loss.py` : contains loss functions used. All in a class called Loss.
   1. `metrics.py` : contains metrics used either directly to be observed in training or after training
   1. `nn.py` : contains some helper functions, the Dense() class, the Dropout() class and a class called FeedForward() used to create a network.
   1. `data/sample_data.py` : function to generate/save/plot some sample data to test the neural network on

#### Usage
To create a new dense layer, used Dense(number of inputs, number of outputs, name of activation function). 

To create a dropout layer, use Dropout(drop_probability)

To create a network with layer_1, layer_2,..., layer_n: model = FeedForward(layer_1, layer_2, ..., layer_n)
   
#### training a model
For now, both the feature data and target must be numpy arrays.

Feature set X: dimensions (m, n) : m is number of samples, n is number of features

Target y: dimensions (m, 1): m is number of samples

If y is for a classification task, y should have integer values 0,1,...(number of classes - 1). Binary would just be 0,1. For now, we do not use one-hot encoded targets for multiclass classification

To train: model.fit(X,y, loss_fn=loss_function_name) is the most basic way to do it, but there are additional possible parameters:
- epochs : number of epochs to train (default is 10)
- verbose : whether to print intermediate statistics for batchs in the middle of an epoch (default is False)
- learning_rate : learning rate for the model (default is 0.01)
- batch_size : size of batches to use for training (default is 32)
- print_every : When verbose is True, how oftne to print out batch results (default is 10)
- val_size : If you want to separate part of the training data to be a validation set, this controls the size. Can be a number in [0,1] (for a fraction) or an integer (default is None)
- val_set : If you want to pass a separate (X_val, y_val) for validation (default is None)
- clip_value : When not none, gradients will be clipped to the interval [-clip_value, clip_value] (default is None)

Some notes about multiclass classification:
- The model expects integer numbers to describe classes, not one-hot encoded versions
- You can either use raw logits (```activation="linear"```) or softmax (```activation="softmax"```)
- Using softmax anywhere other than the final layer will cause an error


#### Sample notebooks
The `notebooks` folder contains some sample notebooks to see how this works
- `datavisualizations.ipynb` : Visualizations of the different targets for the data (all use the same feature space)
- `binary_parabola.ipynb` : Runs a model on a data set with binary classification where the true decision boundary is a parabola.
- `binary_pl.ipynb` : Runs a model on a data set with binary classification where the true decision boundary is an absolute value graph.
- `multiclass.ipynb` : Runs a model on a data set with multiclass classification where the decision boundaries are 2 branches of a hyperbola
- `regression.ipynb` : Runs a model on a data set with a regression, where the true function is z = x<sup>2</sup> + y<sup>2</sup>

