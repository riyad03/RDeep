using System;

namespace RigidWare.Mathf
{
	public class MathCal
	{
        public static class Functions
        {
            public static float Sigmoide(float x)
            {
                float res = (float)(1 / (1 + Math.Exp(-(double)x)));
                return res;
            }

            public static double Sigmoide(double x)
            {
                double res = (1 / (1 + Math.Exp(-(double)x)));
                return res;
            }
            public static float Accuracy(double[] X, bool[] Y)
            {
                float TruePrediction = 0;
                float AllPrediction = 0;

                for (int i = 0; i < X.Length; i++)
                {
                    if (X[i] == 1 && Y[i] || X[i] != 1 && !Y[i])
                    {
                        TruePrediction += 1;
                    }

                    AllPrediction += 1;
                }
                return TruePrediction / AllPrediction;
            }
            public static float Accuracy(double[] X, double[] Y)
            {
                float TruePrediction = 0;
                float AllPrediction = 0;

                for (int i = 0; i < X.Length; i++)
                {
                    if (X[i] == Y[i])
                    {
                        TruePrediction += 1;
                    }
                    AllPrediction += 1;
                }
                return TruePrediction / AllPrediction;
            }
            public static double LogLoss(int a, int y, int i)
            {
                double ll = 0;
                for (int x = 0; x < i; x++)
                {
                    ll += (y * Math.Log(a) + (1 - y) * Math.Log(1 - a));
                }
                return ll;
            }
        }
        public static class Matrix
        {
            public static double[,] SetMatrix(double x, int m, int n)
            {
                double[,] Mat = new double[m, n];
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        Mat[i, j] = x;
                    }
                }
                return Mat;
            }
            public static double[,] Log(double[,] x)
            {
                double[,] res = new double[x.GetLength(0), x.GetLength(1)];

                for (int i = 0; i < x.GetLength(0); i++)
                {
                    for (int j = 0; j < x.GetLength(1); j++)
                    {
                        res[i, j] = Math.Log(x[i, j]);
                    }
                }
                return (res);

            }

            public static double[,] MatrixMultip(double[,] x, double[,] y)
            {

                double[,] result = new double[x.GetLength(0), y.GetLength(1)];

                if (x.GetLength(1) == y.GetLength(0))
                {
                    for (int i = 0; i < x.GetLength(0); i++)
                    {

                        for (int c = 0; c < y.GetLength(1); c++)
                        {
                            result[i, c] = 0;
                            for (int j = 0; j < x.GetLength(1); j++)
                            {
                                /*Console.Write(j + "  ");
                                Console.WriteLine(y[j, c]);
                                Console.Write(result[i, c] + " + ");*/

                                result[i, c] += x[i, j] * y[j, c];
                                //Console.WriteLine(x[i, j] + " * "+y[j, c] +") = "+ result[i, c]);



                            }
                        }
                    }

                    return result;
                }
                else
                {


                    throw new InvalidOperationException("the number of columns of the first matrix is different from the number of rows of the second matrix ");

                }


            }
            public static double[,] MatrixMultipbyScalar(double y, double[,] x)
            {
                double[,] res = new double[x.GetLength(0), x.GetLength(1)];
                for (int i = 0; i < x.GetLength(0); i++)
                {
                    for (int j = 0; j < x.GetLength(1); j++)
                    {
                        res[i, j] = x[i, j] * y;
                    }
                }
                return res;
            }
            public static double[,] MatrixAdd(double[,] x, double[,] y)
            {
                double[,] sum = new double[x.GetLength(0), x.GetLength(1)];
                if (x.GetLength(0) == y.GetLength(0) && x.GetLength(1) == y.GetLength(1))
                {
                    for (int i = 0; i < x.GetLength(0); i++)
                        for (int j = 0; j < x.GetLength(1); j++)
                            sum[i, j] = x[i, j] + y[i, j];

                    return sum;
                }
                else
                {
                    throw new InvalidOperationException("The size of the first Matrix should be equal to size of second matrice");
                }
            }
            public static double[,] MatrixAddToScalar(double y, double[,] x)
            {
                double[,] result = new double[x.GetLength(0), x.GetLength(1)];
                for (int i = 0; i < x.GetLength(0); i++)
                {
                    for (int j = 0; j < x.GetLength(1); j++)
                    {
                        result[i, j] = x[i, j] + y;
                    }
                }
                return result;
            }
            public static double[,] MatrixTrans(double[,] x)
            {
                double[,] result = new double[x.GetLength(1), x.GetLength(0)];

                for (int i = 0; i < result.GetLength(0); i++)
                {
                    for (int j = 0; j < result.GetLength(1); j++)
                    {
                        result[i, j] = x[j, i];
                    }
                }
                return result;
            }
            public static double[,] Concatenate(double[,] A, double[,] B, bool Vertical = true)
            {
                if (Vertical)
                {
                    if (A.GetLength(0) == B.GetLength(0))
                    {
                        double[,] res = new double[A.GetLength(0), A.GetLength(0) + B.GetLength(1) - 1];
                        for (int i = 0; i < A.GetLength(0); i++)
                        {
                            for (int j = 0; j < A.GetLength(1); j++)
                            {
                                res[i, j] = A[i, j];
                            }
                            for (int j = A.GetLength(1); j < A.GetLength(1) + B.GetLength(1); j++)
                            {
                                res[i, j] = B[i, j - A.GetLength(1)];
                            }
                        }
                        return res;
                    }
                    else
                    {

                        throw new InvalidOperationException("the number of rows of the first Array must be equal to that of the second Array");

                    }

                }
                else
                {
                    if (A.GetLength(1) == B.GetLength(1))
                    {
                        double[,] res = new double[A.GetLength(0) + B.GetLength(0), A.GetLength(1)];
                        for (int i = 0; i < A.GetLength(1); i++)
                        {
                            for (int j = 0; j < A.GetLength(0); j++)
                            {
                                res[j, i] = A[j, i];
                            }
                            for (int j = A.GetLength(0); j < A.GetLength(0) + B.GetLength(0); j++)
                            {

                                res[j, i] = B[j - A.GetLength(0), i];
                            }
                        }
                        return res;
                    }
                    else
                    {
                        throw new InvalidOperationException("the number of columns of the first Array must be equal to that of the second Array");
                    }
                }
            }

            public static double[,] Sigmoid(double[,] x)
            {
                double[,] res = new double[x.GetLength(0), x.GetLength(1)];

                for (int i = 0; i < x.GetLength(0); i++)
                {
                    for (int j = 0; j < x.GetLength(1); j++)
                    {

                        res[i, j] = MathCal.Sigmoid(x[i, j]);
                    }
                }
                return (res);
            }

        }
        public static class Arr
        {
            public static double[] Setlinspace(double min, double max, int length)
            {
                length -= 1;
                double[] res = new double[length];
                double s = 0;
                for (int i = 0; i < length; i++)
                {
                    s += (min + max) / (length);
                    res[i] = s;
                }
                return res;
            }
            public static double[] Setarrange(double min, double max, double step)
            {
                int length = (int)(max / step) - 1;
                double[] res = new double[length];
                double s = 0;
                for (int i = 0; i < length; i++)
                {
                    s += step;
                    res[i] = s;
                }
                return res;
            }
            public static double[] Ravel(double[,] x)
            {
                double[] res = new double[x.Length];
                int k = 0;
                for (int i = 0; i < x.GetLength(0); i++)
                {
                    for (int j = 0; j < x.GetLength(1); j++)
                    {
                        res[k] = x[i, j];
                        k++;
                    }
                }
                return res;
            }
            public static bool[] Ravel(bool[,] x)
            {
                bool[] res = new bool[x.Length];
                int k = 0;
                for (int i = 0; i < x.GetLength(0); i++)
                {
                    for (int j = 0; j < x.GetLength(1); j++)
                    {
                        res[k] = x[i, j];
                        k++;
                    }
                }
                return res;
            }
            public static float[] Ravel(float[,] x)
            {
                float[] res = new float[x.Length];
                int k = 0;
                for (int i = 0; i < x.GetLength(0); i++)
                {
                    for (int j = 0; j < x.GetLength(1); j++)
                    {
                        res[k] = x[i, j];
                        k++;
                    }
                }
                return res;
            }
            public static int[] Ravel(int[,] x)
            {
                int[] res = new int[x.Length];
                int k = 0;
                for (int i = 0; i < x.GetLength(0); i++)
                {
                    for (int j = 0; j < x.GetLength(1); j++)
                    {
                        res[k] = x[i, j];
                        k++;
                    }
                }
                return res;
            }


        }
        public static double Factorial(int n)
        {
            if (n > 1)
            {
                return n * Factorial(n - 1);
            }
            else
            {
                return 1;
            }
        }
        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }
        public static double RandomSND()
        {
            Random rand = new Random();
            float y = rand.Next(10, 400);
            y = y / 1000;
            int x = rand.Next(0, 2);
            if (x == 0)
            {
                return Math.Sqrt(-2 * Math.Log(Math.Sqrt(2 * 3.14) * y));
            }
            else
            {
                return -Math.Sqrt(-2 * Math.Log(Math.Sqrt(2 * 3.14) * y));
            }
        }
        
    }
}
