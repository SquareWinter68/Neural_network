using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks.Neural_networks
{
    internal class Data_point
    {
        public double[] inputs { get; set; }
        public double[] expected_outputs { get; set; }

        public Data_point(double[] inputs, double[] expected_outputs)
        {
            this.inputs = inputs;
            this.expected_outputs = expected_outputs;
        }
    }
}
