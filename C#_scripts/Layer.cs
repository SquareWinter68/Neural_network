using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks.Neural_networks
{
    internal class Layer
    {
        public int input_size { get; private set; }
        public int output_size { get; private set; }
        public double[]? outputs { get; private set; }
        public double[][]? weights { get; set; }
        public double[]? biases { get; set;}
        public double[]? weighted_sums { get; private set; }


        public Layer(int input_size, int output_size, double[]? outputs)
        {
            this.input_size = input_size;
            this.output_size = output_size;
            this.outputs = outputs;


            if (weights == null)
            {
                populate_weights();
            }
            if (biases == null)
            {
                populate_biases();
            }
        }


        private void populate_weights()
        {
            Random random = new Random();
            // Since every output node has connections/weights with every of the input nodes
            // a list of lists the size of output nodes is first generated where
            // each index n corresponds to a list of all connections the output node 
            //weights = Enumerable.Range(0, this.output_size).Select(_ => new List<Double>()).ToList();
            List<List<Double>> temp_weights = Enumerable.Range(0, this.output_size).Select(_ => new List<double>()).ToList();
            foreach (List<Double> list_ in temp_weights)
            {
                list_.AddRange(Enumerable.Range(0, this.input_size).Select(_ => random.NextDouble()).ToList());
            }
            weights = temp_weights.Select(_ => _.ToArray()).ToArray();
        }

        private void populate_biases()
        {
            Random random = new Random();
            biases = Enumerable.Range(0, output_size).Select(_ => random.NextDouble()).ToArray();
        }

        public double sigmoid_activation(double x)
        {
            return 1d / (1d + Math.Pow(Math.E, -x));
        }

        public double[] sigmoid_activation_vectorized(double[] weighted_inputs)
        {
            // Applays the sigmoid function to all of the weighted inputs
            return weighted_inputs.Select(x => sigmoid_activation(x)).ToArray();
        }

        public double[] calculate_outputs(double[] inputs)
        {
            // Must write explanation
            Linear_algebra linear_Algebra = new Linear_algebra();
            weighted_sums = linear_Algebra.vector_addition(linear_Algebra.matrix_vector_multiplication_vectorized(weights, inputs), biases);
            outputs = sigmoid_activation_vectorized(weighted_sums);
            return outputs;
        }

        public double node_cost(double outout_activation, double expected_output)
        {
            return Math.Pow(outout_activation - expected_output, 2);
        }

    }
}
