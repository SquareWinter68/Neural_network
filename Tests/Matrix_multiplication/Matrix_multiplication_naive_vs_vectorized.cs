using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;

namespace Neural_networks.Neural_networks.Performance_tests
{
    internal class Matrix_multiplication_naive_vs_vectorized
    {
        public double[][] matrix_multiplication_naive(double[][] matrix1, double[][] matrix2)
        {
            if (matrix1[0].Length != matrix2.Length)
            {
                throw new ArgumentException("Arrays must be of same shape");
                //Change this error later
            }

            double[][] result = new double[matrix1.Length][];

            for (int _ = 0; _ < matrix1.Length; _++)
            {
                result[_] = new double[matrix2[0].Length];
            }

            // using for loops instead of foreach because of performance

            for (int row_number_matrix_1 = 0; row_number_matrix_1 < matrix1.Length; row_number_matrix_1++)
            {

                for (int column_number_matrix_2 = 0; column_number_matrix_2 < matrix2[0].Length; column_number_matrix_2++)
                {
                    double row_col_result = 0d;

                    for (int i = 0; i < matrix1[0].Length; i++)
                    {
                        row_col_result += matrix1[row_number_matrix_1][i] * matrix2[i][column_number_matrix_2];
                    }
                    result[row_number_matrix_1][column_number_matrix_2] = row_col_result;
                }

            }

            return result;
        }

        // Special case when multiplying a matrix with a vector.
        // To avod returning an array of arrays that each contaiin only one element
        public double[] matrix_vector_multiplication_naive(double[][] matrix, double[] vector)
        {
            if (matrix[0].Length != vector.Length)
            {
                throw new ArgumentException("The column of the matrix and the vector have to be of the same size");
            }
            double[] result = new double[matrix.Length];
            for(int row = 0; row < matrix.Length; row++)
            {
                double temp_result = 0d;
                for(int element = 0; element < matrix[0].Length; element++)
                {
                    temp_result += matrix[row][element] * vector[element];
                }
                result[row] = temp_result;
            }
            return result;
        }

        public double[][] matrix_multiplication_vectorized(double[][] matrix1, double[][] matrix2)
        {
            throw new NotImplementedException();
        }

        public double[] matrix_vector_multiplication_vectorized(double[][] matrix, double[] vector_)
        {
            if (matrix[0].Length != vector_.Length)
            {
                throw new ArgumentException("The column of the matrix and the vector have to be of the same size");
            }
            double[] result = new double[matrix.Length];
            int vector_size = Vector<double>.Count;

            for(int row = 0; row < matrix.Length; row++)
            {
                double temp_result = 0d;
                int element = 0;

                for(;element <= matrix[0].Length - vector_size; element += vector_size)
                {
                    Vector<double> row_vector = new Vector<double>(matrix[row], element);
                    Vector<double> vector = new Vector<double>(vector_, element);
                    temp_result += Vector.Dot(row_vector, vector);
                }

                for(; element < matrix[0].Length; element++)
                {
                    temp_result += matrix[row][element] * vector_[element];
                }
                
                result[row] = temp_result;
            }
            return result;
        }

        public double[] matrix_vector_multiplication_vectorized_parallel(double[][] matrix, double[] vector_)
        {
            if (matrix[0].Length != vector_.Length)
            {
                throw new ArgumentException("The column of the matrix and the vector have to be of the same size");
            }
            double[] result = new double[matrix.Length];
            int vector_size = Vector<double>.Count;

            Parallel.For(0, matrix.Length, row =>
            {
                double temp_result = 0d;
                int element = 0;

                for (; element <= matrix[0].Length - vector_size; element += vector_size)
                {
                    Vector<double> row_vector = new Vector<double>(matrix[row], element);
                    Vector<double> vector = new Vector<double>(vector_, element);
                    temp_result += Vector.Dot(row_vector, vector);
                }

                for (; element < matrix[0].Length; element++)
                {
                    temp_result += matrix[row][element] * vector_[element];
                }

                result[row] = temp_result;
            });

            return result;
        }


        public double[] hadamard_product_naive(double[] vector1, double[] vector2)
        {
            if(vector1.Length != vector2.Length)
            {
                throw new ArgumentException("Vectors must be of the same dimension");
            }
            for (int i = 0; i < vector1.Length; i++)
            {
                vector1[i] = vector1[i] * vector2[i];
            }
            return vector1;
        }

        public static double[] hadamard_product_parallel(double[] vector1, double[] vector2)
        {
            if (vector1.Length != vector2.Length)
            {
                throw new ArgumentException("Vectors must be of the same dimension");
            }

            int vectorLength = vector1.Length;
            int vectorWidth = Vector<double>.Count;
            double[] result = new double[vectorLength];
            int i = 0;

            // Process vectors in chunks of Vector<double>.Count elements in parallel
            Parallel.For(0, vectorLength / vectorWidth, j =>
            {
                int startIndex = j * vectorWidth;
                var v1 = new Vector<double>(vector1, startIndex);
                var v2 = new Vector<double>(vector2, startIndex);
                (v1 * v2).CopyTo(result, startIndex);
                i += vectorWidth;
            });

            // Process remaining elements outside of the parallel loop
            for (; i < vectorLength; i++)
            {
                result[i] = vector1[i] * vector2[i];
            }

            return result;
        }

        public void test_matrix_vector_multiplication_performance(int number_of_runs)
        {
            Random random = new Random();
            (int, int, long)[] naive_results = new (int, int, long)[number_of_runs];
            (int, int, long)[] vectorized_results = new (int, int, long)[number_of_runs];
            int min_size = 1000;
            for(int i = 0; i < number_of_runs; i++)
            {
                int matrix_rows = random.Next(min_size, min_size*2);
                int matrix_cols = random.Next(min_size, min_size*2);
                Stopwatch construction = Stopwatch.StartNew();
                double[][] matrix = Enumerable.Range(0, matrix_rows).Select(_ => Enumerable.Range(0, matrix_cols).Select(elemrnt => random.NextDouble()).ToArray()).ToArray();
                double[] vector = Enumerable.Range(0, matrix_cols).Select(_ => random.NextDouble()).ToArray();
                construction.Stop();

                Stopwatch naive = Stopwatch.StartNew();
                double[] result1 = matrix_vector_multiplication_naive(matrix, vector);
                //double[] result1 = matrix_vector_multiplication_vectorized(matrix, vector);
                naive.Stop();
                Stopwatch fast = Stopwatch.StartNew();
                //double[] result2 = matrix_vector_multiplication_vectorized(matrix, vector);
                double[] result2 = matrix_vector_multiplication_vectorized_parallel(matrix, vector);
                fast.Stop();

                naive_results[i] = new(matrix_rows, matrix_cols, naive.ElapsedMilliseconds);
                vectorized_results[i] = new(matrix_rows, matrix_cols, fast.ElapsedMilliseconds);
                if (i % 10 == 0)
                {
                    min_size += 100;
                }
            }
            Array.Sort(naive_results, (x, y) => (x.Item1 * x.Item2).CompareTo(y.Item1 * y.Item2));
            Array.Sort(vectorized_results, (x, y) => (x.Item1 * x.Item2).CompareTo(y.Item1 * y.Item2));

            using (StreamWriter file = new StreamWriter("C:\\Users\\Vukasin\\Desktop\\py_N_N's\\C#_nn's\\normal_vs_vectorized.csv"))
            {
                //string contents = "Rows,Columns,Vector,Speed normal,Speed vectorized,Divisible by 4\n";
                string contents = "Rows,Columns,Vector,Speed normal,Speed parallel vectorized,Divisible by 4\n";
                for (int i = 0; i < naive_results.Length; i++)
                {
                    contents += $"{naive_results[i].Item1},{naive_results[i].Item2},{naive_results[i].Item2},{naive_results[i].Item3},{vectorized_results[i].Item3},{vectorized_results[i].Item2 % 4 == 0}\n";
                }
                file.WriteLine(contents);
            }
            
            //CollectionAssert.AreEqual(result1, result2);

            //Console.WriteLine($"The naive approach took {naive.ElapsedMilliseconds} ms while the fast approach took {fast.ElapsedMilliseconds} ms.\nFor a vector size of {matrix_rows}\nIt took {construction.ElapsedMilliseconds} to construct the matrix and the vector\nTotal time:{construction.ElapsedMilliseconds + naive.ElapsedMilliseconds + fast.ElapsedMilliseconds} ms\nSIMD vector size:{Vector<double>.Count}");
        }

        public void discard()
        {
            double[] result = matrix_vector_multiplication_naive(new double[][] {new double[] { 1, 2, 3}, new double[] { 4, 5, 6 }, new double[] { 7, 8, 9 } }, new double[] { 1, 2, 3 });
            double[] result1 = matrix_vector_multiplication_vectorized(new double[][] { new double[] { 1, 2, 3 }, new double[] { 4, 5, 6 }, new double[] { 7, 8, 9 } }, new double[] { 1, 2, 3 });
            print_vector(result);
            print_vector(result1);
            //Vector<double> vector = new Vector<double>(new[] { 1.0, 2.0, 4.0, 0.0 });
        }

        public void print_vector(double[] vector)
        {
            foreach(double element in vector)
            {
                Console.Write($"{element} ");
            }
            Console.WriteLine();
        }
    }
}
