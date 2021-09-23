# jax-for-hamiltonian
This notebook contains a brief introduction to [JAX](https://github.com/google/jax) and a demonstration of how it can be used to define and solve the equations of motion of a simple mass-spring system. I wrote this up in the process of learning to create wave propagation simulations from scratch.

Relevant features of JAX, as described in the JAX repository:
- "automatically differentiate native Python and NumPy code" with the `grad` function (I used this to automatically generate the equations of motion without having to compute them by hand)
- "automatic vectorization" with `vmap` function (I used this to simplify the function definitions and overall code structure)

I also looked into using the `jit` ("just-in-time") decorator to speed up function calls, but `jit` is not compatible with control flow operations like if-else statements, which I wanted to use for the boundary conditions.

**View the Jupyter notebook on [nbviewer](https://nbviewer.jupyter.org/github/notchia/jax-for-hamiltonian/blob/main/jax_for_hamiltonian.ipynb)**
