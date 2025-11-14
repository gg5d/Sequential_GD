# Sequential Gradient Descent Comparison

This project compares three different ways of training a simple neural network:

1. **normalGD** – Uses PyTorch autograd for backprop and does manual parameter updates (no optimizer).  
2. **pytorchGD** – Re-implements the forward and backward pass using PyTorch Tensors, updates all weights at once.  
3. **pytorchSGD** – Similar to `pytorchGD` but updates the second layer before calculating the first layer gradients (sequential style).  

The purpose of this project is not just to train MNIST, but to explore how different learning rules behave, especially those that deviate from standard gradient descent.

---

## Why

At the moment, most artificial neural networks are being trained with standard gradient descent using backpropagation.  Autograd libraries like PyTorch automate and speed up the manipulation of differentiable functions, but even when dispensed with the mathematical underpinnings and tensor manipulation, autograd libraries are developed on the original thesis of **simultaneous** gradient updates.  


Biological learning, on the other hand, is thought to be more "sequential" and less simultaneous. The idea of **sequential gradient descent** is to update the weights of one layer during the update before computing the gradients of the previous layer. That decision is more similar to how biological neurons may be adapting.  


By writing backpropagation in PyTorch Tensors manually, we circumvent all of the consequences of using autograd, we can explore these other alternative update rules. This allows us to ask the question, *What happens if we train ANNs with more biologically-inspired learning rules?*

---

## Results so far

- In the context of a simple **3-layer ANN**, the use of sequential gradient descent using PyTorch SGD yields **faster convergence** at almost all connector/HP test cases (minus the very last test case in this cell) than regular, “push the button” PyTorch gradient descent.  
- Surprisingly, in this study, PyTorch GD (batch update rule) is faster than PyTorch autograd backpropagation too.  
- For now, we are not measuring accuracy—**it is important to point out that the different rules for updating weights lead to significantly different training dynamics**.  

The meaning of the text is clear from the plots generated that examine the curve of training loss from each different update rule. The accuracy of the evaluation is printed after training.

---

## Next steps

- Scale up the ANN to more layers and units to see if the trends hold in larger networks.  
- Make the architecture more flexible and support GPU acceleration by rewriting NumPy operations with PyTorch tensors.
- Move beyond MNIST and test on harder datasets (e.g. CIFAR-10).  
- Experiment with different architectures like CNNs and compare whether sequential updates still give faster convergence.  

---

## Updates

**November 14th 2025**

- PyTorch tensor implementations of both standard gradient descent and sequential gradient descent currently show **no significant difference** in behavior.
- Switching to a classification task was attempted, but the results so far remain almost indistinguishable between the two update strategies.
- To better understand the cause, I’ve gone back to basics: testing a **simple, naive neural network for classification** from scratch. *Building up incrementally to pinpoint the issue.*


**October 7th 2025**

- Rewrote all the NumPy code to use **PyTorch tensors** instead. This makes it possible to run everything on the GPU while keeping the same manual backprop logic.  
- Made the network architecture **fully flexible**. The code now builds however many layers you list in `layer_sizes`, so you don’t need to hardcode each layer anymore.  
- Added support for different activation functions. You can now pick between **sigmoid**, **ReLU**, or **leaky ReLU** when running `compare_methods()`.  
- The training code now works with **MNIST**, **FashionMNIST**, and **CIFAR-10**, with the right input size and normalization handled automatically.

Overall, the project is now more flexible, faster, and easier to scale without changing the core idea of comparing sequential vs. standard gradient descent.


## Current Working Files
- `pytorch_SGD.py`: PyTorch Tensors implementation of a simple neural network based around 3 layers.
- `pytorch_SGD_multilayer.py`: pytorch_SGD.py but with 5 layers instead of 3.
- `pytorch_SGD_scalable.py`: pytorch_SGD.py but with any amount of defined layers.
- `naive_higham.py`: Naive (3 layer) implementation of the Neural Network introduced in the Higham Paper
- `naivetest.py`: naive_higham.py but done using classes rather than copy and pasting the same code twice.
- `higham_SGD_ANN.py`: Older implementation of the ANN from the Higham paper, updated to fit the needs of testing SGD (currently unknown if the correct is accurate)


## Requirements

- Python 3.8+  
- PyTorch  
- torchvision  
- matplotlib  
- numpy  
- mplcursors (optional, for interactive plots)

Install with:

```bash
pip install torch torchvision matplotlib numpy mplcursors
