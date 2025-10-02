# Sequential Gradient Descent Comparison

This project compares three different ways of training a simple neural network on the MNIST dataset:

1. **normalGD** – Uses PyTorch autograd for backprop and does manual parameter updates (no optimizer).  
2. **numpyGD** – Re-implements the forward and backward pass in NumPy, updates all weights at once.  
3. **numpySGD** – Similar to `numpyGD` but updates the second layer before calculating the first layer gradients (sequential style).  

The purpose of this project is not just to train MNIST, but to explore how different learning rules behave, especially those that deviate from standard gradient descent.

---

## Why

At the moment, most artificial neural networks are being trained with standard gradient descent using backpropagation.  Autograd libraries like PyTorch automate and speed up the manipulation of differentiable functions, but even when dispensed with the mathematical underpinnings and tensor manipulation, autograd libraries are developed on the original thesis of **simultaneous** gradient updates.  


Biological learning, on the other hand, is thought to be more "sequential" and less simultaneous. The idea of **sequential gradient descent** is to update the weights of one layer during the update before computing the gradients of the previous layer. That decision is more similar to how biological neurons may be adapting.  


By writing backpropagation in NumPy manually, we circumvent all of the consequences of using autograd, we can explore these other alternative update rules. This allows us to ask the question, *What happens if we train ANNs with more biologically-inspired learning rules?*

---

## Results so far

- In the context of a simple **3-layer ANN**, the use of sequential gradient descent using NumPy SGD yields **faster convergence** at almost all connector/HP test cases (minus the very last test case in this cell) than regular, “push the button” PyTorch gradient descent.  
- Surprisingly, in this study, NumPy GD (batch update rule) is faster than PyTorch autograd backpropagation too.  
- For now, we are not measuring accuracy—**it is important to point out that the different rules for updating weights lead to significantly different training dynamics**.  

The meaning of the text is clear from the plots generated that examine the curve of training loss from each different update rule. The accuracy of the evaluation is printed after training.

---

## Next steps

- Scale up the ANN to more layers and units to see if the trends hold in larger networks.  
- Make the architecture more flexible and support GPU acceleration by rewriting NumPy operations with PyTorch tensors.  
- Move beyond MNIST and test on harder datasets (e.g. CIFAR-10).  
- Experiment with different architectures like CNNs and compare whether sequential updates still give faster convergence.  

---

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
