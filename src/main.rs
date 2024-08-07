use rust_nn::NeuralNetwork;

fn main() {
    let mut nn = NeuralNetwork::new(&vec![2, 2, 1]); // 2 inputs, one hidden layer with 2 neurons, 1 output

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    nn.train(&inputs, &targets, 10000, 0.1);

    for input in &inputs {
        let output = nn.predict(input);
        println!("{:?} -> {:?}", input, output);
    }
}
