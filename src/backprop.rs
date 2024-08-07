use crate::activation::sigmoid_derivative;
use crate::NeuralNetwork;

impl NeuralNetwork {
    pub fn backpropagate(
        &mut self,
        activations: &Vec<Vec<f64>>,
        zs: &Vec<Vec<f64>>,
        target: &Vec<f64>,
        learning_rate: f64,
    ) {
        let mut delta: Vec<f64> = activations
            .last()
            .unwrap()
            .iter()
            .zip(target.iter())
            .map(|(a, y)| a - y)
            .collect();

        for l in (0..self.layers.len()).rev() {
            let sp: Vec<f64> = zs[l].iter().map(|&z| sigmoid_derivative(z)).collect();
            delta = delta.iter().zip(sp.iter()).map(|(d, s)| d * s).collect();

            let prev_activation = &activations[l];
            let layer = &mut self.layers[l];

            for (i, neuron) in layer.neurons.iter_mut().enumerate() {
                for j in 0..neuron.weights.len() {
                    neuron.weights[j] -= learning_rate * delta[i] * prev_activation[j];
                }
                neuron.bias -= learning_rate * delta[i];
            }

            if l > 0 {
                let prev_layer = &self.layers[l - 1];
                delta = prev_layer
                    .neurons
                    .iter()
                    .map(|neuron| neuron.weights.iter().zip(&delta).map(|(w, d)| w * d).sum())
                    .collect();
            }
        }
    }
}
