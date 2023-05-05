using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks.Neural_networks
{
    internal class Select_vs_for
    {
        public delegate double[] Performance();
        public double function(double x)
        {
            return Math.Pow(x, 2) + 2 * x - 3;
        }
        public string Time(Performance preformance)
        {
            Stopwatch stopwatch = new Stopwatch();
            
            stopwatch.Start();
            preformance();
            stopwatch.Stop();

            TimeSpan timeSpan = stopwatch.Elapsed;
            return $"H:{timeSpan.Hours} m:{timeSpan.Minutes} s:{timeSpan.Seconds} ms:{timeSpan.Milliseconds}";
        }

        public void Time_them_both()
        {      
            Performance performance = () => Enumerable.Range(0, 9000000).Select(_ => function(_)).ToArray();   
            Performance performance1 = () =>
            {
                double[] values = new double[9000000];
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = function(i);
                }
                return values;
            };
            Console.WriteLine(Time(performance));
            Console.WriteLine(Time(performance1));

        }

    }
}
