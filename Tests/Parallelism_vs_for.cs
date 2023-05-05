using NUnit.Framework;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace Neural_networks.Neural_networks
{
    internal class Parallelism_vs_for
    {
        public double[] vector_adition_for(double[] vector1, double[] vector2)
        {
            if (vector1.Length != vector2.Length)
            {
                throw new ArgumentException("No No");
            }
            return vector1.Select((value, index) => value + vector2[index]).ToArray();
        }

        public double[] vector_addition_parallel(double[] vector1, double[] vector2)
        {
            if (vector1.Length != vector2.Length)
            {
                throw new ArgumentException("Vectors have to be of the same length");
            }

            double[] result = new double[vector1.Length];

            Parallel.ForEach(Partitioner.Create(0, vector1.Length), range =>
            {
                for (int i = range.Item1; i < range.Item2; i++)
                {
                    result[i] = vector1[i] + vector2[i];
                }
            });
            return result;
        }
        public void test_vector_adition_performance()
        {
            Random random = new Random();
            int vector_size = 500;
            (int, double)[] times_for = new(int, double)[100];
            (int, double)[] times_parallel = new (int, double)[100];
            for (int i = 0; i < 100; i++)
            {
                double averagetime_for = 0d;
                double averagetime_parallel = 0d;
                
                // this loop is used to average out the times
                for (int j = 0; j < 10; j++)
                {
                    double[] vector1 = Enumerable.Range(0, vector_size).Select(_ => random.NextDouble()).ToArray();
                    double[] vector2 = Enumerable.Range(0, vector_size).Select(_ => random.NextDouble()).ToArray();
                    Stopwatch stopwatch1 = Stopwatch.StartNew();
                    double[] result = vector_adition_for(vector1, vector2);
                    stopwatch1.Stop();
                    averagetime_for += stopwatch1.ElapsedMilliseconds;

                    Stopwatch stopwatch2 = Stopwatch.StartNew();
                    double[] result2 = vector_addition_parallel(vector1, vector2);
                    stopwatch2.Stop();
                    averagetime_parallel += stopwatch2.ElapsedMilliseconds;

                    //CollectionAssert.AreEqual(result, result2);
                }
                times_for[i] = (vector_size, averagetime_for/10);
                times_parallel[i] = (vector_size, averagetime_parallel/10);
                vector_size += 10000;
            }

            using (StreamWriter file = new StreamWriter("C:\\Users\\Vukasin\\Desktop\\py_N_N's\\C#_nn's\\data1.csv"))
            {
                String file_contents = "Vector_size,for_loop_time,parallel_time\n";
                for(int i = 0; i < times_for.Length; i++)
                {
                    file_contents += $"{times_for[i].Item1},{times_for[i].Item2},{times_parallel[i].Item2}\n";
                }
                file.WriteLine(file_contents);
            }

            
            
            //CollectionAssert.AreEqual(result, result2);

            //Console.WriteLine($"for {massive} elements the for loop took {for_loop} ms, whlie the parrallel version took {parallel} ms");
        }

        public void test()
        {
            Random random = new Random();
            
            for (int i = 0; i<10; i++)
            {
                int vector_size = random.Next(100, 500);//(160000, 160500);
                double[] vector1 = Enumerable.Range(0, vector_size).Select(_ => random.NextDouble()).ToArray();
                double[] vector2 = Enumerable.Range(0, vector_size).Select(_ => random.NextDouble()).ToArray();

                Stopwatch stopwatch2 = Stopwatch.StartNew();
                double[] result2 = vector_addition_parallel(vector1, vector2);
                stopwatch2.Stop();

                Stopwatch stopwatch1 = Stopwatch.StartNew();
                double[] result = vector_adition_for(vector1, vector2);
                stopwatch1.Stop();
                //Console.WriteLine($"for loop:{stopwatch1.ElapsedMilliseconds}");

                Console.WriteLine($"for:{stopwatch1.ElapsedMilliseconds}  parallel:{stopwatch2.ElapsedMilliseconds}");
            }

        }
    }
}
