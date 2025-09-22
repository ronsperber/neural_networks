# my_nn: A Simple Neural Network Implementation from Scratch with NumPy
This library implements feedforward neural networks using only NumPy, for learning and experimentation with classification and regression tasks
### Setup

1. Create the conda environment
```bash
conda env create -f environment.yml
```

2. Activate the conda environment
```bash
conda activate neural_networks
```

3. (Optional, but recommended) Install the package in editable mode
```bash
pip install -e .
```

#### Modules (all in my_nn)
- `activations.py`: contains all activation functions, implemented as subclasses of `Activation`. Softmax is handled in the multiclass loss function, not directly as an activation.
- `loss.py`: contains loss functions, subclasses of `Loss`.
- `metrics.py`: metrics used for evaluation during or after training.
- `nn.py`: contains `Dense`, `Dropout`, and `FeedForward` classes for building networks.
- `data/sample_data.py`: functions to generate, save, and plot sample datasets.


#### Usage
To create a new dense layer, use Dense(number of inputs, number of outputs, name of activation function). 

Example:
First the imports
```python
from my_nn.nn import Dense, Dropout, FeedForward
```

```python
layer = Dense(2, 5, activation="relu")
```

To create a dropout layer, use Dropout(drop_probability)
Example:
```python
dropout = Dropout(0.2)
```

To create a network with layer_1, layer_2,..., layer_n: model = FeedForward(layer_1, layer_2, ..., layer_n)
Example:
```python
dense_1 = Dense(4 ,10, activation="relu")
dense_2 = Dense(10, 1, activation="sigmoid")
model = FeedForward(dense_1, dense_2)
```
   
#### training a model
For now, both the feature data and target must be numpy arrays.

Feature set X: dimensions (m, n) : m is number of samples, n is number of features

Target y: dimensions (m, 1): m is number of samples

If y is for a classification task, y should have integer values 0,1,...(number of classes - 1). Binary would just be 0,1. For now, we do not use one-hot encoded targets for multiclass classification

To train: model.fit(X,y, loss_fn=loss_function_name) is the most basic way to do it, but there are additional possible parameters:
- epochs : number of epochs to train (default is 10)
- verbose : whether to print intermediate statistics for batches in the middle of an epoch (default is False)
- learning_rate : learning rate for the model (default is 0.01)
- batch_size : size of batches to use for training (default is 32)
- print_every : When verbose is True, how often to print out batch results (default is 10)
- val_size : If you want to separate part of the training data to be a validation set, this controls the size. Can be a number in [0,1] (for a fraction) or an integer (default is None)
- val_set : If you want to pass a separate (X_val, y_val) for validation (default is None)
- clip_value : When not none, gradients will be clipped to the interval [-clip_value, clip_value] (default is None)

A simple example:
```python
model.fit(X_train, y_train, loss_fn="categorical_cross_entropy", epochs=20)
y_pred = model.predict(X_test)
```

Some notes about multiclass classification:
- The model expects integer numbers to describe classes, not one-hot encoded versions
- You can either use raw logits (```activation="linear"```) or softmax (```activation="softmax"```)
  e.g.:
  ```python
  # Using raw logits
   layer2 = Dense(3, 3, activation="linear")
   # Or using softmax
   layer2 = Dense(3, 3, activation="softmax")
  ```

- Using softmax anywhere other than the final layer will cause an error


#### Sample notebooks
The `notebooks` folder contains some sample notebooks to see how this works
- `datavisualizations.ipynb` : Visualizations of the different targets for the data (all use the same feature space)
- `binary_parabola.ipynb` : Runs a model on a data set with binary classification where the true decision boundary is a parabola.
- `binary_pl.ipynb` : Runs a model on a data set with binary classification where the true decision boundary is an absolute value graph.
- `multiclass.ipynb` : Runs a model on a data set with multiclass classification where the decision boundaries are 2 branches of a hyperbola
- `regression.ipynb` : Runs a model on a data set with a regression, where the true function is z = x<sup>2</sup> + y<sup>2</sup>

