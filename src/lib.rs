extern crate rand;

pub mod activation;
pub mod backprop;

use activation::sigmoid;
use rand::distributions::{Distribution, Uniform};

pub struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    pub fn new(num_inputs: usize) -> Neuron {
        let mut rng = rand::thread_rng();
        let between = Uniform::from(-1.0..1.0);
        Neuron {
            weights: (0..num_inputs).map(|_| between.sample(&mut rng)).collect(),
            bias: between.sample(&mut rng),
        }
    }

    pub fn forward(&self, inputs: &Vec<f64>) -> f64 {
        let mut sum = 0.0;
        for (w, &input) in self.weights.iter().zip(inputs) {
            sum += w * input;
        }
        sum + self.bias
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(num_neurons: usize, num_inputs: usize) -> Layer {
        Layer {
            neurons: (0..num_neurons).map(|_| Neuron::new(num_inputs)).collect(),
        }
    }

    pub fn forward(&self, inputs: &Vec<f64>) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|neuron| sigmoid(neuron.forward(inputs)))
            .collect()
    }
}

pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layer_sizes: &Vec<usize>) -> NeuralNetwork {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(layer_sizes[i + 1], layer_sizes[i]));
        }
        NeuralNetwork { layers }
    }

    pub fn predict(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut outputs = inputs.clone();
        for layer in &self.layers {
            outputs = layer.forward(&outputs);
        }
        outputs
    }

    pub fn train(
        &mut self,
        inputs: &Vec<Vec<f64>>,
        targets: &Vec<Vec<f64>>,
        epochs: usize,
        learning_rate: f64,
    ) {
        for _ in 0..epochs {
            for (input, target) in inputs.iter().zip(targets) {
                // Forward pass
                let mut activations = vec![input.clone()];
                let mut zs = vec![];

                for layer in &self.layers {
                    let z: Vec<f64> = layer.forward(&activations.last().unwrap());
                    zs.push(z.clone());
                    activations.push(z.iter().map(|&z| sigmoid(z)).collect());
                }

                // Backward pass
                self.backpropagate(&activations, &zs, target, learning_rate);
            }
        }
    }
}
