using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks.Neural_networks
{
    internal class Linear_algebra
    {
        public double[] vector_adition(double[] vector1, double[] vector2)
        {
            if (vector1.Length != vector2.Length)
            {
                throw new ArgumentException("Vectors have to be of the same length");
            }

            return vector1.Select((value, index) => (value + vector2[index])).ToArray();
        }

        public double[] vector_subtraction(double[] vector1, double[] vector2)
        {
            if ((vector1.Length != vector2.Length)){
                throw new ArgumentException("Vectors have to be of the same length");
            }
            return vector1.Select((value, index) => (value - vector2[index])).ToArray();
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

        public double[] hadarmad_vector_product(double[] vector1, double[] vector2)
        {
            if(vector1.Length != vector2.Length)
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
    }
}
