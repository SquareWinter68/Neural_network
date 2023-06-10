using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks.Neural_networks
{
    public struct Trasnposed_matrix
    {
        // This is done to avoid actually transposing a matrix, which has a time complexity of
        // O(n^2)
        private readonly double[][] matrix;

        public Trasnposed_matrix(double[][] _matrix)
        {
            matrix = _matrix;
        }

        public double[] this[int row]
        {
            get { return matrix[row]; }
        }

        public double this[int row, int column]
        {
            get { return matrix[column][row]; }
            set { matrix[column][row] = value; }
        }

        public int Rows { get { return matrix[0].Length; } }
        public int Columns { get { return matrix.Length; } }

    }


    internal class Neural_network
    {
        private double[] input_layer;
        private Layer[] layers;
        private double[] output_error;
        private double[][] layer_errors;
        Linear_algebra lin_alg = new Linear_algebra();

        // Output error stores an array of gradients of error for every node in the output layer
        // This gradient is the essential part in calculating weight and bias gradients

        public Neural_network(int input_layer_size, int[] hidden_layers, int output_layer_size)
        {
            if (input_layer_size == 0)
            {
                throw new ArgumentException("The input layer must be provided");
            }

            //Makes and populates the layers list based on input data
            layers = new Layer[hidden_layers.Length + 1];
            layers[0] = new Layer(input_layer_size, hidden_layers[0], null);
            for (int i = 1; i < hidden_layers.Length; i++)
            {
                layers[i] = new Layer(layers[i - 1].output_size, hidden_layers[i], null);
            }
            
            layers[layers.Length - 1] = new Layer(layers[layers.Length - 2].output_size, output_layer_size, null);
            

        }

        public double[] calculate_outputs(double[] inputs)
        {

            foreach(Layer layer in layers)
            {
                inputs = layer.calculate_outputs(inputs);
            }
            return inputs;
        }


        public double sigmoid_derivative(double x)
        {
            
            double numerator = Math.Pow(Math.E, -x);

            return numerator / ((1 + numerator) * (1 + numerator));
        }

        public double[] sigmoid_derivative_vector(double[] vector)
        {
            return vector.Select(element => sigmoid_derivative(element)).ToArray();
        }

        public double cost_data_point(Data_point data_point)
        {
            //estimates how the network is doing for a single data point (which usually consits of an array of inputs, and expected outputs to test against)
            double[] activations = calculate_outputs(data_point.inputs);
            // Cx = 1/2 * (expected_values_vecor - predcted_values)^2
            // vector subtraction, vector squared is dot product of the same vecor
            // (y-a) vector

            double[] error = lin_alg.vector_subtraction(data_point.expected_outputs, activations);
            return lin_alg.dot_product(error, error)/2.0;
            
            // deviding by two for the derivative to come to cancle the two out
            // thereby saving hundreds of milliseconds by avoiding a vector operation 
        }

        public double[] cost_data_point_derivative(Data_point data_point)
        {
            // Derivative 2 * (a - y), remember two's cancle out
            double[] activations = calculate_outputs(data_point.inputs);
            return lin_alg.vector_subtraction(activations, data_point.expected_outputs);
        }


        public double average_cost(Data_point[] data_points)
        {
            //Iterates through all of the training data and gets the average cost of all datapoints
            //this is of more interest than a single data point beacuse the network need to be 
            //optimized for the whole dataset.
            double cost = 0d;
            foreach (Data_point point in data_points)
            {
                cost += cost_data_point(point);
            }
            return cost / data_points.Length;
        }

        public double average_cost_parallel(Data_point[] data_points)
        {
            double sum = data_points.AsParallel().Sum(point => cost_data_point(point));

            return sum / data_points.Length;
        }


        public double[] average_cost_derivative(Data_point[] data_points)
        {
            double[] average_gradients = Enumerable.Range(0, layers[^1].output_size)
                                                   .Select(element => 0.0)
                                                   .ToArray();

            object lockObj = new object();

            Parallel.ForEach(data_points,() => Enumerable.Range(0, layers[^1].output_size).Select(element => 0.0).ToArray(),
                
                (point, loopState, localState) =>
                {
                    double[] gradient = cost_data_point_derivative(point);
                    return lin_alg.vector_addition(localState, gradient);
                },
                
                localState =>
                {
                    lock (lockObj)
                    {
                        average_gradients = lin_alg.vector_addition(average_gradients, localState);
                    }
                });

            //double factor = 1.0 / data_points.Length;
            average_gradients = lin_alg.vector_scalar_multiplication(average_gradients, 1d / data_points.Length);

            output_error = average_gradients;
            return output_error;
        }


        public double[][] backpropagation(Data_point[] data,double learnig_rate)
        {
            //Console.WriteLine(average_cost_parallel(data));
            //average_cost_derivative(data);
            double[][] layer_errors = new double[layers.Length][];

            layer_errors[^1] = output_error;
            for (int layer = layers.Length-2; layer >= 0; layer--)
            {
                //lin_alg.matrix_vector_multiplication_vectorized(lin_alg.transpose_matrix(layers[layer].get_weights()), layer_errors[layer + 1])
                // layer = weights[layer+1]T * layer_error[layer+1] hadamard dc/dz
                //output_layer_error();

                double[] mat_mult = lin_alg.matrix_vector_multiplication(lin_alg.transpose_matrix(layers[layer + 1].weights), layer_errors[layer + 1]);
                layer_errors[layer] = lin_alg.hadamard_vector_product(mat_mult, sigmoid_derivative_vector(layers[layer].biases));

                // Update biases
                //layers[layer].biases -= layer_errors[layer];
                layers[layer].biases = lin_alg.vector_subtraction_with_second_vector_scaling_factor(layers[layer].biases, layer_errors[layer], learnig_rate);

                double[][] weight_gradients = lin_alg.row_vector_column_vector_multiplication(layer_errors[layer + 1], layers[layer].outputs);

                // Update weights
                layers[layer].weights = lin_alg.matrix_subtraction_with_second_matrix_scaling_factor(layers[layer].weights, weight_gradients, learnig_rate);
            }
            return layer_errors;
            Console.WriteLine(average_cost_parallel(data));
        }

       

        public double[] get_output_layer_activations()
        {
            return layers[-1].outputs;
        }

    }
}
