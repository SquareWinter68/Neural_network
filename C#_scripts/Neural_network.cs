using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks.Neural_networks
{
    internal class Neural_network
    {
        private double[] input_layer;
        private Layer[] layers;
        private double[] output_layer_error;
        Linear_algebra lin_alg = new Linear_algebra();

        public Neural_network(double[] input_layer, int[] hidden_layers, int output_layer_size)
        {
            if(input_layer == null || input_layer.Length == 0)
            {
                throw new ArgumentException("The input layer must be provided");
            }
            this.input_layer = input_layer;
            layers = new Layer[hidden_layers.Length + 1];
            layers[0] = new Layer(input_layer.Length, hidden_layers[0], null, null);
            for(int i = 1; i < hidden_layers.Length; i++)
            {
                layers[i] = new Layer(layers[i-1].output_size, hidden_layers[i], null, null);
            }
            layers[layers.Length-1] = new Layer(layers[layers.Length-2].output_size, output_layer_size, null, null);
            calculate_outputs(input_layer);
            
        }

        public double[] calculate_outputs(double[] inputs)
        {
            layers[0].inputs = inputs;
            for(int i = 1; i < layers.Length; i++)
            {
                layers[i].inputs = layers[i - 1].calculate_outputs_vectorized();
            }

            return layers[layers.Length - 1].calculate_outputs_vectorized();
        }
        
        public double cost_single_point(double expected_value, double activation_value)
        {
            // Cost(y, a) = (y - a)^2
            double error = (expected_value - activation_value);
            return error * error;
        }

        public double cost_single_point_derivative(double expected_value, double activation_value)
        {
            //d/da Cost(y, a) = 2 * (y - a) * -1 = 2 * (a - y)
            return 2 * (activation_value - expected_value);
        }

        public double cost_data_point(Data_point data_point)
        {
            //estimates how the network is doing for a single data point (which usually consits of an array of inputs, and expected outputs to test against)
            double[] outputs = calculate_outputs(data_point.inputs);
            // Cx = 1/2 * (expected_values_vecor - predcted_values)^2
            // vector subtraction, vector squared is dot product of the same vecor
            // (y-a) vector

            double[] error = lin_alg.vector_subtraction(data_point.expected_outputs, outputs);
            return lin_alg.dot_product(error, error);
        }

        public double[] cost_data_point_derivative(Data_point data_point)
        {
            // Cost function is of the form C(y, a) = (y-a)^2
            // Its partial derivative with respect to a is
            //2*(y-a)*-1 = 2(a-y)
            double[] outputs = calculate_outputs(data_point.inputs);
            Layer layer = layers[layers.Length - 1];

            output_layer_error = lin_alg.hadamard_vector_product(lin_alg.vector_subtraction(outputs, data_point.expected_outputs), layer.weighted_sums);
            return output_layer_error;
        }

        public double average_cost(Data_point[] data_points)
        {
            //Iterates through all of the training data and gets the average cost of all datapoints
            //this is of more interest than a single data point beacuse the network need to be 
            //optimized for the whole dataset.
            double cost = 0d;
            foreach(Data_point point in data_points)
            {
                cost += cost_data_point(point);
            }
            return cost/data_points.Length;
        }
            
        
    }
}
