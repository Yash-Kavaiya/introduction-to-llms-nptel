# Lec 06 : Introduction to PyTorch

# PyTorch and NumPy Code Explanation

Let's go through this code line by line with detailed explanations:

## Imports and Basic PyTorch Operations

```python
import torch
from torch import tensor
import numpy as np
```
- Imports the PyTorch library for deep learning and tensor operations
- Directly imports the tensor function from torch for convenience
- Imports NumPy with the standard 'np' alias for numerical computing

```python
torch.__version__
```
- Returns the installed PyTorch version string

```python
tensor0 = torch.tensor(1)
```
- Creates a scalar tensor containing the integer value 1
- This is a 0-dimensional tensor (just a single value)

```python
tensor0
```
- Evaluates and displays the tensor0 variable

```python
tensor0.ndim
```
- Returns the number of dimensions of tensor0
- Result will be 0 since it's a scalar tensor (0-dimensional)

## Automatic Differentiation Example

```python
x = torch.tensor(2.0, requires_grad=True)
```
- Creates a tensor with value 2.0
- The `requires_grad=True` parameter tells PyTorch to track operations on this tensor for gradient computation
- This is essential for automatic differentiation in neural networks

```python
y = 3 * torch.sigmoid(x) + 5
```
- Creates a mathematical expression using x:
  - `torch.sigmoid(x)` applies the sigmoid function (σ(x) = 1/(1+e^(-x)))
  - Multiplies the result by 3
  - Adds 5
- The result is stored in tensor y
- Since x has requires_grad=True, y will also track gradients

```python
y.backward()
```
- Computes the gradient of y with respect to x (dy/dx)
- This is the power of automatic differentiation - PyTorch automatically calculates the derivative

```python
print(x.grad)
```
- Prints the gradient value stored in x.grad
- The gradient is: 3 * sigmoid(x) * (1 - sigmoid(x))
- For x=2.0, sigmoid(x)≈0.88, so gradient≈3 * 0.88 * 0.12≈0.32

## NumPy Einstein Summation (einsum)

```python
import numpy as np
A = [[1, 5], [3, 7]]
B = [[2, -1], [4, 2]]
np.einsum('ij, ij -> ', A, B)
```
- Creates two 2×2 matrices A and B
- Uses Einstein summation notation to perform a specific operation:
  - 'ij, ij -> ' means: multiply corresponding elements and sum everything
  - Calculation: (1×2 + 5×(-1) + 3×4 + 7×2) = 2 - 5 + 12 + 14 = 23
- This is equivalent to np.sum(np.multiply(A, B)) or the Frobenius inner product

## Key Concepts Covered
- Tensor creation and manipulation in PyTorch
- Automatic differentiation for gradient computation
- The sigmoid activation function
- Einstein summation for efficient array operations

# PyTorch Tensors: Comprehensive Guide

Let's analyze each line of code that demonstrates tensor creation and manipulation in PyTorch:

## Creating Tensors of Different Dimensions

```python
# Scalar - tensor with zero dimensions
tensor0 = torch.tensor(4)
tensor0.ndim  # Returns 0
```
- Creates a scalar (0D) tensor with the value 4
- `.ndim` shows it has 0 dimensions

```python
# Vector - tensor with one dimension
tensor1 = torch.tensor([6,8,0,1,2])
tensor1.ndim  # Returns 1
```
- Creates a vector (1D) tensor with 5 elements
- `.ndim` returns 1, confirming it's one-dimensional

```python
# Matrix - tensor with two dimensions
tensor2 = torch.tensor([[0, 1, 7],[4, 2, 4]])
tensor2.ndim  # Returns 2
```
- Creates a 2×3 matrix (2D tensor)
- `.ndim` returns 2, confirming it's two-dimensional

## Understanding Tensor Shape and Properties

```python
print(f'vector:\n{tensor1}\tNo. of dimensions: {tensor1.ndim}\tShape: {tensor1.shape}\n')
```
- Prints the vector tensor, its dimensions, and shape ([5])
- Shape represents the size in each dimension

```python
print(f'matirx:\n{tensor2}\tNo. of dimensions: {tensor2.dim()}\tShape: {tensor2.size()}\n')
```
- Prints the matrix tensor, its dimensions, and shape ([2, 3])
- Note: `.dim()` is equivalent to `.ndim`
- `.size()` is equivalent to `.shape`

```python
A = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
A.shape  # Returns torch.Size([3, 3])
```
- Creates a 3×3 matrix and shows its shape

## Multiple Ways to Create Tensors

```python
A = torch.rand((3,5))  # Creates a 3×5 tensor with random values between 0 and 1
```

```python
size = (3,4)
tensor4 = torch.empty(size)  # Creates an uninitialized 3×4 tensor (values are whatever is in memory)
```

```python
tensor5 = torch.rand(size)  # Creates a 3×4 tensor with random values from uniform distribution [0,1)
```

```python
tensor6 = torch.zeros(size)  # Creates a 3×4 tensor filled with zeros
```

```python
tensor7 = torch.ones(size)  # Creates a 3×4 tensor filled with ones
```

## Tensor Data Types

```python
tensor4 = torch.rand(6,2)
print(tensor4)
```
- Creates a 6×2 tensor with random values
- Default data type is float32

```python
tensor5 = torch.rand(1, 2, 3, dtype=torch.float16)
print(tensor5)
```
- Creates a 1×2×3 tensor with random values using float16 data type (half precision)

```python
tensor4.type(torch.double)  # Converts tensor4 to double precision (float64)
tensor4.dtype  # Returns the data type
```

## Converting from NumPy Arrays

```python
import numpy as np
example_array = np.array([[9,3],[0,4]])
```

```python
tensor8 = torch.from_numpy(example_array)  # Creates a tensor that shares memory with array
```
- Important: Changes to `example_array` will affect `tensor8` (memory is shared)

```python
tensor9 = torch.tensor(example_array)  # Creates a new tensor (no memory sharing)
```
- Changes to `example_array` won't affect `tensor9`

```python
example_array *= 2  # Modifies the numpy array
print(example_array)  # Shows modified array
print(tensor8)  # Shows modified tensor (affected by numpy change)
print(tensor9)  # Shows original values (not affected by numpy change)
```

## Creating Tensors from Other Tensors

```python
tensor10 = torch.ones_like(A)  # Creates tensor of ones with same shape as A
```

## Device Configuration (CPU/GPU)

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
- Automatically selects GPU if available, otherwise uses CPU

```python
tensor11 = torch.ones(3,7).to(device)  # Creates tensor and moves to selected device
```

```python
tensor11 = torch.zeros(3,7, device=device)  # Creates tensor directly on selected device
```

```python
tensor23 = torch.tensor([[2,32,7],[123,23,24]])
tensor23.size()  # Returns torch.Size([2, 3])
```

## Key Concepts
- Tensors are multi-dimensional arrays (generalizations of vectors and matrices)
- PyTorch offers multiple ways to create and manipulate tensors
- Tensor operations can be performed on GPU for faster computation
- Tensor shapes describe how data is arranged across dimensions

# Tensor Indexing and Slicing in PyTorch

Let's analyze each line of code demonstrating how to access elements in PyTorch tensors:

## Creating a 3D Tensor

```python
A = torch.tensor([[[5,7,2,4],[1,2,3,4]], 
                 [[-5,-7,-2,-4],[-1,-2,-3,-4]], 
                 [[15,17,12,14],[11,12,13,14]]])
```
- Creates a 3D tensor with shape (3,2,4)
- This represents 3 "blocks", each containing 2 rows of 4 elements

```python
A.dim()  # Returns 3
```
- Confirms this is a 3-dimensional tensor

```python
A.shape  # Returns torch.Size([3, 2, 4])
```
- Shows tensor has 3 elements in first dimension, 2 in second, 4 in third

## Single Element Access with Multiple Brackets

```python
print(A[0][1][3])  # Returns 4
```
- Accesses element at:
  - First block (index 0)
  - Second row (index 1)
  - Fourth element (index 3)

```python
print(A[2][0][2])  # Returns 12
```
- Accesses element at:
  - Third block (index 2)
  - First row (index 0)
  - Third element (index 2)

## Accessing Entire Rows

```python
print(A[0][1])  # Returns tensor([1, 2, 3, 4])
```
- Returns complete second row from first block

```python
print(A[2][1])  # Returns tensor([11, 12, 13, 14])
```
- Returns complete second row from third block

## Error Cases (Commented Out)

```python
# print(A[3])  # Error - index out of bounds
```
- Would cause error as tensor only has 3 blocks (indices 0,1,2)

```python
# print(A[1][2][3][4])  # Error - too many dimensions
```
- Would cause error - accessing more dimensions than tensor has

## More Element Access

```python
print(A[0][1][2])  # Returns 3
```
- Accesses element at first block, second row, third column

## 2D Tensor Indexing and Slicing

```python
tensor2  # Displays the 2D tensor defined earlier
```

```python
tensor2.dim()  # Returns 2
```

```python
tensor2.size()  # Returns torch.Size([2, 3])
```

```python
tensor2[0]  # Returns first row of tensor2
```
- Gets entire first row

```python
tensor2[1,0]  # Returns element at second row, first column
```
- Comma notation is equivalent to multiple brackets

## Slicing Operations

```python
tensor2[:, :2]  # Returns all rows, but only first two columns
```
- The colon `:` means "all indices in this dimension"
- `:2` means "indices from 0 up to (but not including) 2"

```python
tensor2[0, :]  # Returns first row with all columns
```
- Gets complete first row

```python
tensor2[:,2]  # Returns all rows, but only third column
```
- Gets complete third column

## Key Concepts
- Indexing in PyTorch uses zero-based counting
- You can access elements with multiple brackets `A[i][j][k]` or commas `A[i,j,k]`
- Slicing with `:` allows selecting ranges of elements
- Out of bounds indices cause runtime errors

  # PyTorch Tensor Operations and Manipulations

Let's break down these tensor operations and manipulations:

## Basic Arithmetic Operations

```python
tensor12 = torch.ones(2, 3)
tensor13 = torch.rand(2, 3)

print(tensor12)  # 2×3 tensor filled with ones
print(tensor13)  # 2×3 tensor with random values between 0 and 1
```

### Element-wise Operations

```python
# Element-wise addition
tensor14 = tensor12 + tensor13  # Equivalent to torch.add(tensor12, tensor13)
print(tensor14)
```
- Adds corresponding elements from both tensors
- Requires tensors of compatible shapes

```python
# Element-wise subtraction
tensor15 = tensor12 - tensor13  # Equivalent to torch.sub(tensor12, tensor13)
print(tensor15)
```
- Subtracts tensor13 elements from tensor12 elements

```python
# Element-wise multiplication
tensor16 = tensor12 * tensor13  # Equivalent to torch.mul(tensor12, tensor13)
print(tensor16)
```
- Multiplies corresponding elements

```python
# Element-wise division
tensor17 = tensor12 / tensor12  # Equivalent to torch.div(tensor12, tensor12)
print(tensor17)
```
- Divides corresponding elements
- This example divides tensor by itself, resulting in all ones

## Tensor Shape Manipulation

```python
x = torch.randint(0, 3, (4, 5))  # 4×5 tensor with random integers from 0 to 2
```
- `randint(low, high, size)` creates tensor with random integers
- Range is [low, high-1]

```python
# Reshaping tensors
y = x.view(20)       # Reshapes to a 1D tensor with 20 elements
z = x.view(-1, 10)   # Reshapes to 2×10 tensor (uses -1 to auto-compute rows)
print(x.size(), y.size(), z.size())
```
- `view` changes the shape without changing data
- `-1` tells PyTorch to infer that dimension's size

```python
a = torch.arange(9)   # Creates a tensor with values [0,1,2,3,4,5,6,7,8]
a = a.reshape(3, 3)   # Reshapes to a 3×3 matrix
```
- `reshape` is similar to `view` but can copy data if needed

## Tensor Concatenation

```python
b = torch.randint(0, 9, (3, 3))  # 3×3 tensor with random integers from 0 to 8
```

```python
c = torch.cat((a, b), dim=1)  # Concatenates horizontally (along columns)
```
- Results in a 3×6 tensor
- First 3 columns from a, next 3 from b

```python
d = torch.cat((a, b), dim=0)  # Concatenates vertically (along rows)
```
- Results in a 6×3 tensor
- First 3 rows from a, next 3 from b

## Reduction Operations

```python
p = torch.randint(0, 9, (2, 3, 5))  # 3D tensor with shape 2×3×5
```

```python
p.sum()  # Returns the sum of all elements in the tensor
```

```python
p.sum(dim=0)  # Sums along first dimension (reducing from shape [2,3,5] to [3,5])
```
- Collapses the first dimension by summing
- For each position in the remaining dimensions, adds values from the reduced dimension

```python
p.sum(dim=0).shape  # Shows torch.Size([3, 5])
```
- Confirms the new shape after reduction

## Key Concepts
- Element-wise operations maintain tensor shapes
- Shape manipulation changes how elements are organized
- Concatenation combines tensors along specified dimensions
- Reduction operations like sum collapse dimensions by applying a function
# Understanding PyTorch Autograd

This code demonstrates PyTorch's automatic differentiation system (autograd) which calculates derivatives for neural network training.

## The Math Problem

The code works with this function and its derivative:
$$y=3\sigma(x) + 5$$
$$\frac{\partial y}{ \partial x} = 3*\sigma(x)(1-\sigma(x))$$

When x = 2, the derivative is:
$$\frac{\partial y}{ \partial x} = 3*\sigma(2)(1-\sigma(2))=3*0.8808*(1-0.8808)=0.3149$$

## Basic Tensor Without Gradient Tracking

```python
import torch  
x = torch.tensor(2.0)
x.requires_grad, x.is_leaf  # Returns (False, True)
y = 3 * torch.sigmoid(x) + 5
y.requires_grad, y.is_leaf  # Returns (False, True)
```
- `requires_grad=False` by default, so no gradients are tracked
- `is_leaf=True` indicates this is a user-created tensor

## Enabling Gradient Tracking

```python
import torch  
x = torch.tensor(2.0, requires_grad=True)  
x.requires_grad, x.is_leaf  # Returns (True, True)
y = 3 * torch.sigmoid(x) + 5
y  # Tensor with value ~7.64
y.requires_grad, y.is_leaf  # Returns (True, False)
```
- Setting `requires_grad=True` enables gradient tracking
- Operations on a tensor with gradient tracking also track gradients
- `y` is not a leaf tensor since it was created through operations

## The Computation Graph

```python
print(x.grad_fn)  # None (leaf tensors don't have grad_fn)
print(y.grad_fn)  # Shows AddBackward or similar
```
- `grad_fn` shows the backward function used for gradient computation
- Leaf tensors (like `x`) have no `grad_fn`
- Derived tensors (like `y`) have a `grad_fn` based on their creation operation

## Computing Gradients

```python
print(x.grad)  # None (before backward)
y.backward()   # Compute gradients
print(x.grad)  # tensor(~0.3149)
```
- `backward()` computes gradients of `y` with respect to tensors with `requires_grad=True`
- The result in `x.grad` matches our manual calculation (~0.3149)

## Resetting Gradients

```python
x.grad.zero_()  # Resets gradient to zero
y = 3 * torch.sigmoid(x) + 5  # Recompute y
y.backward()    # Compute gradients again
x.grad          # Same gradient value as before
```
- Gradients accumulate by default, so we need to zero them between backward passes
- `zero_()` is an in-place operation that zeros out the tensor

## More Complex Example

```python
a = torch.rand(2, 5, requires_grad=True)  # 2×5 random tensor
b = a * a + a + 5  # Element-wise operations
c = b.mean()  # Mean of all elements in b
b.retain_grad()  # Keep b's gradients (non-leaf tensors don't retain gradients by default)

print(a.grad)  # None (before backward)
c.backward()   # Compute gradients of c with respect to a
print(a.grad)  # Shows dc/da gradient
b.grad         # Shows dc/db gradient
```
- For complex computation graphs, gradients flow backward from the final output
- `retain_grad()` tells PyTorch to keep gradients for non-leaf tensors
- The gradients are calculated for all tensors with `requires_grad=True`

## Key Concepts
- PyTorch builds a dynamic computation graph for tracking operations
- Autograd computes exact derivatives automatically
- Gradients flow backward from the output to all tensors requiring gradients
- This system enables training neural networks by computing weight updates

# Linear Regression with Gradient Descent in PyTorch

This code implements a linear regression model using gradient descent to find the optimal parameters. Let's break down each section:

## Generating Training Data

```python
# Generate train data based on y = 5 * x + 3
x = torch.linspace(-1.0, 1.0, 15).reshape(15, 1)
w = torch.tensor([5])  # True weight value
b = torch.tensor([3])  # True bias value
y = w * x + b  # Generate target values
print(x.shape)  # torch.Size([15, 1])
print(x)
```
- Creates 15 evenly spaced values between -1 and 1, formatted as a column vector
- Defines the true parameters (w=5, b=3) that we'll try to recover through training
- Generates target values using the formula y = w*x + b

## Parameter Initialization

```python
# Parameter initialization
w = torch.randn(size=(1,1), requires_grad=True)  # Random initial weight with gradient tracking
b = torch.randn(size=(1,1), requires_grad=True)  # Random initial bias with gradient tracking
print(x+b)  # Tensor addition

# Alternate initialization without gradient tracking (ISSUE: won't work for training)
w = torch.randn(size=(1,1))  # No requires_grad=True
b = torch.randn(size=(1,1))  # No requires_grad=True
```
- First initializes parameters with gradient tracking enabled
- Then re-initializes without gradient tracking (this is problematic - see below)

## Model Definition

```python
print(x)
print(w)
print(b)

# Perform a test operation
y = x + b
print(y)

# Define the forward function (model prediction)
def forward(x):
    return w * x + b

# Define the loss function (Mean Squared Error)
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

print('w:', w)
print('b:', b)
```
- Defines the linear model as `w * x + b`
- Uses Mean Squared Error (MSE) for the loss function

## Training with Gradient Descent

```python
# Define hyperparameters
learning_rate = 0.03
num_epochs = 180

# Train the model
for epoch in range(num_epochs):
    # Forward pass
    y_pred = forward(x)
    
    # Compute loss
    l = loss(y, y_pred)
    
    # Backward pass (compute gradients)
    l.backward()
    
    # Update parameters using gradient descent
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # Reset gradients for next iteration
    w.grad.zero_()
    b.grad.zero_()
    
    # Print progress every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'epoch {epoch+1}: w = {w.item():.3f}, b = {b.item():.3f}, loss = {l.item():.3f}')
```
- Performs gradient descent for 180 epochs
- Each epoch:
  1. Makes predictions using the current model
  2. Calculates the loss (error)
  3. Computes gradients with `backward()`
  4. Updates parameters in the direction that reduces loss
  5. Resets gradients to zero for the next iteration
- Prints progress every 10 epochs

## Important Note: Issue in the Code

There's a critical problem in this code: parameters are initialized without gradient tracking (`requires_grad=True`) but the training loop tries to access their gradients.

For this to work correctly, you must initialize parameters with:
```python
w = torch.randn(size=(1,1), requires_grad=True)
b = torch.randn(size=(1,1), requires_grad=True)
```

Otherwise, `w.grad` and `b.grad` will be `None`, causing errors during training.

## Key Concepts
- Gradient descent iteratively adjusts parameters to minimize loss
- Parameters must have `requires_grad=True` for gradient tracking
- `backward()` computes parameter gradients
- Parameters are updated using the formula: `parameter -= learning_rate * parameter.grad`
- Gradients must be zeroed after each update to avoid accumulation

# Neural Networks with PyTorch and MNIST

Let's break down this neural network implementation for handwritten digit classification:

## Imports and Setup

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
- Imports core PyTorch modules, optimization tools, and dataset utilities
- Sets up device (GPU if available, otherwise CPU)
- `print(dir(datasets))` lists all available datasets in torchvision

## Hyperparameters

```python
hidden_size = 400     # Neurons in hidden layer
num_epochs = 8        # Training iterations over dataset
batch_size = 32       # Examples per mini-batch
learning_rate = 0.0001
```

## Loading MNIST Dataset

```python
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
```
- Downloads MNIST if needed, stores in ./data
- Applies ToTensor transform to convert images to normalized tensors [0,1]

## Exploring Dataset Properties

```python
print(train_dataset.classes)         # Class labels (digits 0-9)
print(train_dataset.data.shape)      # torch.Size([60000, 28, 28])
print(train_dataset.targets.shape)   # torch.Size([60000])

print(test_dataset.classes)
print(test_dataset.data.shape)       # torch.Size([10000, 28, 28])
print(test_dataset.targets.shape)    # torch.Size([10000])
```
- 60,000 training images, 10,000 test images
- Each image is 28×28 pixels

## Creating DataLoaders

```python
in_features = 784   # 28*28 flattened images
out_features = 10   # 10 digit classes

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
```
- DataLoaders handle batching, shuffling, and loading

## Visualizing Data

```python
data = iter(train_dataloader)
imgs, labels = next(data)
print(imgs.shape)    # torch.Size([32, 1, 28, 28])
print(labels.shape)  # torch.Size([32])

# Displays 10 sample images with their labels
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(imgs[i][0], cmap='grey')
    plt.xlabel(f'Label = {labels[i].item()}')
plt.show()
```

## Defining the Neural Network

```python
class BasicNeuralNet(nn.Module):
    def __init__(self, hidden_size):
        super(BasicNeuralNet, self).__init__()
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(in_features, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, out_features)
    
    def forward(self, x):
        out = self.layer1(x)
        out = torch.relu(out)
        out = self.layer2(out)
        return out
```
- Creates a 2-layer feedforward neural network
- First layer: 784→400 with ReLU activation
- Second layer: 400→10 for digit classification

## Model Initialization

```python
model = BasicNeuralNet(hidden_size).to(device)

w1, b1, w2, b2 = list(model.parameters())
print(w1.shape)  # torch.Size([400, 784])
print(b1.shape)  # torch.Size([400])
```
- Creates model and moves to configured device
- Extracts weights and biases to examine shapes

## Loss Function and Optimizer

```python
criterion = nn.CrossEntropyLoss()  # Appropriate for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

## Training Loop

```python
total_steps = len(train_dataloader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        # Reshape images to (batch_size, 784)
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % 300 == 0:
            print(f'Epoch {epoch}, Step {i}/{total_steps}, Loss: {loss.item():.3f}')
```
- For each epoch, processes batches of images
- Flattens each 28×28 image to a 784-element vector
- Computes predictions, loss, and gradients
- Updates model parameters

## Evaluation

```python
with torch.no_grad():
    correct = 0
    num_samples = len(test_dataset)
    
    for imgs, labels in test_dataloader:
        imgs = imgs.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
    
    acc = correct / num_samples
    print(f'Accuracy: {100*acc} %')
```
- Disables gradient tracking for evaluation
- Processes test dataset batch by batch
- Calculates classification accuracy

## Architecture Summary
- Input layer: 784 neurons (flattened 28×28 images)
- Hidden layer: 400 neurons with ReLU activation
- Output layer: 10 neurons (digit classes 0-9) 
- Loss: Cross-entropy for multi-class classification
- Optimizer: Adam with learning rate 0.0001

