using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks.Neural_networks
{
    internal class Neural_network
    {
        Layer[] layers;

        public Neural_network(int hidden_layer_count)
        {
            layers = new Layer[hidden_layer_count];
        }

        public double[] calculate_outputs(double[] inputs)
        {
            layers[0].inputs = inputs;
            foreach(var layer in layers)
            {

            }
            throw new NotImplementedException();
        }
    }
}
