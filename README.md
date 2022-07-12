# neural_net_perceptron
Created a perceptron to better understand the inner workings of AI in Python. Employed a single perceptron and its nonlinear extension (neural networks) to detect whether or not images contain animals.

For part 1 (perceptron): The perceptron model is a linear function that tries to separate data into two or more classes. It does this by learning a set of weight coefficients and then adding a bias. classifyPerceptron() will take as input the training data, training labels, development data, learning rate, and maximum number of iterations. It returns a list of labels for the development data. Only NumPy is used program the perceptron, no non-standard libraries were used.

For part 2 (neural nets): PyTorch and NumPy were uesd to implement the neural net. Input is fed into the net from the input layer and the data is passed through the hidden layers and out to the output layer. The hidden layer contains a ReLU activation function, and cross-entropy is used as a loss function. There are 32 hidden units (h=32) and 3072 input units, one for each channel of each pixel in an image (d=(32)<sup>2</sup>(3)=3072).

Achieved .80 dev-set accuracy for part 1, and 0.84 dev-set accuracy for part 2 for the boolean classifier for whether or not the image contains an animal.
` 
python3 mp2_part1.py -h 

> optional arguments:

>   -h, --help            show this help message and exit

>   --dataset DATASET_FILE

>                         the directory of the training data

>   --method METHOD       classification method, ['perceptron']

>   --lrate LRATE         Learning rate - default 1.0

>   --max_iter MAX_ITER   Maximum iterations - default 10




python3 mp2_part1.py -h 

> optional arguments:

>   -h, --help            show this help message and exit

>   --dataset DATASET_FILE

>                         directory containing the training data

>   --max_iter MAX_ITER   Maximum iterations: default 500

>   --seed SEED           seed source for randomness

`
