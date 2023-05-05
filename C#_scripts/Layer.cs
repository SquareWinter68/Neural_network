using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks.Neural_networks
{
    internal class Layer
    {
        public int input_size { get; set; }
        public int output_size { get; set; }
        public double[]? inputs { get; set; }
        public double[]? outputs { get; set; }
        private double[][]? weights = null;
        private double[]? biases = null;
        public double[]? weighted_sums = null;
        

        public Layer(int input_size, int output_size, double[]? inputs, double[]? outputs)
        {
            this.input_size = input_size;
            this.output_size = output_size;
            this.inputs = inputs;
            this.outputs = outputs;

            if (inputs == null)
            {
                //throw new ArgumentNullException(nameof(inputs), "Input array cannot be null");
                //populate_inputs();
            }

            if (weights == null)
            {
                populate_weights();
            }
            if(biases == null)
            {
                populate_biases();
            }
        }

        public void populate_inputs()
        {
            if (inputs == null)
            {
                Random random = new Random();

                inputs = Enumerable.Range(0, this.input_size)
                                              .Select(_ => random.NextDouble()).ToArray();
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

        public Double sigmoid_activation(Double x)
        {
            return 1d / (1d + Math.Pow(Math.E, -x));
        }

        public double[] sigmoid_activation_vectorized(double[] weighted_inputs)
        {
            return weighted_inputs.Select(x => sigmoid_activation(x)).ToArray();
        }

        public double[] calculate_outputs()
        {
            outputs = new double[output_size];
            for(int node_index = 0; node_index < output_size; node_index++)
            {
                double weighted_input = biases[node_index];
                foreach(var weight in weights[node_index].Select((value, index)=>(value, index)))
                {
                    weighted_input += weight.value * inputs[weight.index];
                }
                outputs[node_index] = sigmoid_activation(weighted_input);
            }
            return outputs;
        }

        public double[] calculate_outputs_vectorized()
        {
            Linear_algebra linear_Algebra = new Linear_algebra();
            weighted_sums = linear_Algebra.vector_adition(linear_Algebra.matrix_vector_multiplication(weights, inputs), biases);
            outputs = sigmoid_activation_vectorized(weighted_sums);
            return outputs;
        }

       public double node_cost(double outout_activation, double expected_output)
        {
            return Math.Pow(outout_activation - expected_output, 2);
        } 
       
    }
}
