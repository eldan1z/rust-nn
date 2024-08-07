# Neural Network from Scratch in Rust

This project implements a simple neural network from scratch using Rust. The goal is for the neural network to learn the XOR function, which means the network should output values close to [0.0] or [1.0] depending on the input pairs. After successful training, the network should approximate these results accurately.


## Modules

- **activations.rs**: Contains the activation functions.
- **backprop.rs**: Contains the backpropagation logic.
- **lib.rs**: Defines the neural network structure and integrates the activation functions and backpropagation logic.
- **main.rs**: Contains the main function to train and test the neural network.

## Getting Started

### Prerequisites

- Rust (Installation instructions can be found [here](https://www.rust-lang.org/tools/install))

### Installing

1. Clone the repository
    ```sh
    git clone https://github.com/eldan1z/rust_nn.git
    cd rust_nn
    ```

2. Add dependencies in `Cargo.toml`
    ```toml
    [dependencies]
    rand = "0.8"
    ```

3. Create the necessary source files with the provided content.

### Running the Project

1. Build and run the project using Cargo

   ```sh
   cargo run
   ```

### Example Output

```plaintext
[0.0, 0.0] -> [0.001]
[0.0, 1.0] -> [0.999]
[1.0, 0.0] -> [0.999]
[1.0, 1.0] -> [0.001]
```
