using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks.Neural_networks
{
    internal class Linear_algebra
    {
        public double[] vector_addition(double[] vector1, double[] vector2)
        {
            if ((vector1.Length != vector2.Length))
            {
                throw new ArgumentException("Vectors have to be of the same length");
            }
            double[] result = new double[vector1.Length];
            int vectorsize = Vector<double>.Count;
            int i = 0;
            for (; i < vector1.Length - vectorsize; i += vectorsize)
            {
                Vector<double> vec1 = new Vector<double>(vector1, i);
                Vector<double> vec2 = new Vector<double>(vector2, i);
                var temp_res = vec1 + vec2;
                temp_res.CopyTo(result, i);
            }

            if (i != vector1.Length)
            {
                for (; i < vector1.Length; i++)
                {
                    result[i] = vector1[i] + vector2[i];
                }
            }
            return result;
        }

        public double[] vector_subtraction(double[] vector1, double[] vector2)
        {
            if ((vector1.Length != vector2.Length))
            {
                throw new ArgumentException("Vectors have to be of the same length");
            }
            double[] result = new double[vector1.Length];
            int vectorsize = Vector<double>.Count;
            int i = 0;
            for (; i < vector1.Length - vectorsize; i += vectorsize)
            {
                Vector<double> vec1 = new Vector<double>(vector1, i);
                Vector<double> vec2 = new Vector<double>(vector2, i);
                var temp_res = vec1 - vec2;
                temp_res.CopyTo(result, i);
            }

            if (i != vector1.Length)
            {
                for (; i < vector1.Length; i++)
                {
                    result[i] = vector1[i] - vector2[i];
                }
            }
            return result;
        }

        public double[] vector_subtraction_with_second_vector_scaling_factor(double[] vector1, double[] vector2, double scaling_factor)
        {
            if ((vector1.Length != vector2.Length))
            {
                throw new ArgumentException("Vectors have to be of the same length");
            }
            double[] result = new double[vector1.Length];
            int vectorsize = Vector<double>.Count;
            int i = 0;
            for (; i < vector1.Length - vectorsize; i += vectorsize)
            {
                Vector<double> vec1 = new Vector<double>(vector1, i);
                Vector<double> vec2 = new Vector<double>(vector2, i);
                var temp_res = vec1 - (scaling_factor * vec2);
                temp_res.CopyTo(result, i);
            }

            if (i != vector1.Length)
            {
                for (; i < vector1.Length; i++)
                {
                    result[i] = vector1[i] - (scaling_factor * vector2[i]);
                }
            }
            return result;
        }

        public double[] vector_scalar_multiplication(double[] vector, double scalar)
        {
            return vector.Select(_ => _ * scalar).ToArray();
        }

        public double dot_product(double[] vector1, double[] vector2)
        {
            if (vector1.Length != vector2.Length)
            {
                throw new ArgumentException("Vectors must be of the same shape/dimension");
            }
            double dot = 0;

            for (int i = 0; i < vector1.Length; i++)
            {
                dot += vector1[i] * vector2[i];
            }
            return dot;
        }

        public List<Double> cross_product(List<Double> vector1, List<Double> vector2)
        {
            throw new NotImplementedException();
        }

        public double[] hadamard_vector_product(double[] vector1, double[] vector2)
        {
            if (vector1.Length != vector2.Length)
            {
                throw new ArgumentException("Vectors must be of same dimension");
            }
            //return vector1.Select((value, index) => (value * vector2[index])).ToArray();
            //need faster implementation
            double[] result = new double[vector1.Length];
            for (int i = 0; i < vector1.Length; i++)
            {
                result[i] = vector1[i] * vector2[i];
            }
            return result;
        }

        public double[][] matrix_subtraction_with_second_matrix_scaling_factor(double[][] matrix1, double[][] matrix2, double matrix2_scaliing_factor)
        {
            if (!(matrix1.Length == matrix2.Length && matrix1[0].Length == matrix2[0].Length))
            {
                throw new ArgumentException("Matrices must be of same shape");
            }
            int vectorsize = Vector<double>.Count;
            for(int row = 0; row < matrix1.Length; row++)
            {
                int element = 0;
                for(; element <= matrix1[0].Length - vectorsize; element += vectorsize)
                {
                    Vector<double> mat1_vector = new Vector<double>(matrix1[row], element);
                    Vector<double> mat2_vector = new Vector<double>(matrix2[row], element);
                    mat2_vector *= matrix2_scaliing_factor;
                    var temp_res = mat1_vector - mat2_vector;
                    temp_res.CopyTo(matrix1[row], element);
                }
                
                for(; element < matrix1[0].Length; element++)
                {
                    matrix1[row][element] = matrix1[row][element] - matrix2_scaliing_factor * matrix2[row][element];
                }
            }
            return matrix1;
        }

        public double[][] matrix_multiplication(double[][] matrix1, double[][] matrix2)
        {
            if (matrix1[0].Length != matrix2.Length)
            {
                throw new ArgumentException("Arrays must be of same shape");
                //Change this error later
            }
            
            double[][] result = new double[matrix1.Length][];

            for(int _ = 0; _ < matrix1.Length; _++)
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

        public double[] matrix_vector_multiplication(double[][] matrix, double[] vector)
        {
            // this function supposes the vector it is given is transposed
            // the user should also suppose the vector they are returned is transposed
            
            if (matrix[0].Length != vector.Length)
            {
                throw new ArgumentException("Arrays must be of same shape");
            }

            double[] result = new double[matrix.Length];

            for(int matrix_row = 0; matrix_row < matrix.Length; matrix_row++)
            {
                double current_result = 0d;
                for (int matrix_column = 0; matrix_column < matrix[0].Length; matrix_column++)
                {
                    current_result += matrix[matrix_row][matrix_column] * vector[matrix_column];
                }
                result[matrix_row] = current_result;
            }

            return result;
            //reminder this result is transposed!
        }

        public double[] matrix_vector_multiplication_vectorized(double[][] matrix, double[] vector_)
        {
            double[] result = new double[matrix.Length];
            int vector_size = Vector<double>.Count;

            for (int row = 0; row < matrix.Length; row++)
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
            }
            return result;
        }

        public double[] matrix_vector_multiplication_vectorized(Trasnposed_matrix matrix, double[] vector_) {
            if (matrix.Columns != vector_.Length)
            {
                throw new ArgumentException("The columnsize and vector length must match");
            }
            int vectorsize = Vector<double>.Count;
            double[] result = new double[matrix.Rows];
            for (int row = 0; row < matrix.Rows; row++)
            {
                double temp_result = 0d;
                int element = 0;
                for (; element <= matrix.Columns - vectorsize; element += vectorsize)
                {
                    Vector<double> row_vector = new Vector<double>(matrix[row], element);
                    Vector<double> vector = new Vector<double>(vector_, element);
                    temp_result += Vector.Dot(row_vector, vector);
                }
                
                for(; element < matrix.Columns; element ++)
                {
                    temp_result += matrix[row][element] * vector_[element];
                }

                result[row] = temp_result;
            }
            return result;
        }

        public double[][] row_vector_column_vector_multiplication(double[] row_vector, double[] column_vector)
        {
            double[][] result = new double[row_vector.Length][];

            for (int row = 0; row < row_vector.Length; row++)
            {
                result[row] = new double[column_vector.Length];
                for (int column = 0; column < column_vector.Length; column++)
                {
                    result[row][column] = row_vector[row] * column_vector[column];
                }
            }
            return result;
        }


        public double[][] transpose_matrix(double[][] matrix)
        {
            double[][] result = new double[matrix[0].Length][];
            for (int row = 0; row < result.Length; row++)
            {
                result[row] = new double[matrix.Length];
            }
            
            for(int matrix_row = 0; matrix_row < matrix.Length; matrix_row++)
            {
                for(int matrix_column = 0; matrix_column < matrix[0].Length; matrix_column++)
                {
                    result[matrix_column][matrix_row] = matrix[matrix_row][matrix_column];
                }
            }
            return result;
        }
        
    }
}
