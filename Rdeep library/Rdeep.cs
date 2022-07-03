using System;
using System.Collections.Generic;
using System.Threading;
using System.IO;
using RigidWare.Mathf;

namespace RigidWare.RDeep
{
	public class Rdeep
	{
        /// <summary>
        /// Randomizes Weight values from the standard normalization distribution.
        /// </summary>
        /// <param name="X"></param>
        /// <returns>Returns 2D Array Weight values from the standard normalization distribution.</returns>
        public static double[,] InitialisationW(double[,] X)
        {
            Console.WriteLine("Initializing: 0%");
            double[,] W = new double[X.GetLength(1), 1];
            for (int i = 0; i < W.GetLength(0); i++)
            {
                double x = MathCal.RandomSND();

                W[i, 0] = x;
                Thread.Sleep(2);
                Console.WriteLine(i * 100 / W.GetLength(0) + "%");
            }


            return W;
        }
        /// <summary>
        /// Randomizes a Bias value from the standard normalization distribution.
        /// </summary>
        /// <param name="X"></param>
        /// <returns>Returns a 2D arrays whose elements are equal to the Bias value.</returns>
        public static double[,] InitialisationB(double[,] X)
        {
            double[,] B = new double[X.GetLength(1), 1];
            B[0, 0] = MathCal.RandomSND();
            for (int i = 0; i < B.GetLength(0); i++)
            {
                for (int j = 0; j < B.GetLength(1); j++)
                {
                    B[i, j] = B[0, 0];
                }
            }
            return B;
        }
        /// <summary>
        /// Reads Weight values that have been saved in FileName.
        /// </summary>
        /// <param name="FileName"></param>
        /// <returns>Returns Weight values from FileName.</returns>
        /// <exception cref="FileNotFoundException"></exception>
        public static double[,] ReadW(string FileName)
        {
            int l,c;
            if (File.Exists(FileName))
            {
                string[] r = File.ReadAllLines(FileName);
                l = r.Length;
                double[,] w = new double[l, r[0].Split(';').Length-1];

                for (int i = 0; i < r.Length; i++) {
                    r[i]=r[i].Remove(r[i].Length - 1);
                    if (r[i].Contains(";"))
                    {
                        string[] s = r[i].Split(';');
                        c = s.Length;

                        for (int j = 0; j < c; j++)
                        {
                            if (s[j] != "")
                            {
                                w[i, j] = double.Parse(s[j]);
                                Console.WriteLine("w "+w[i, j]);
                            }

                        }
                    }
                    else
                    {
                        w[i,0]= double.Parse(r[i]);
                        
                    }
                    
                }
                
                return w;
            }
            else
            {
                throw new FileNotFoundException(FileName+" doesn't Exist");
            }
            //double[,] w =
        }

        /// <summary>
        /// Reads Bias values that have been saved in FileName.
        /// </summary>
        /// <param name="FileName"></param>
        /// <returns>Returns Bias values from FileName.</returns>
        /// <exception cref="FileNotFoundException"></exception>
        public static double ReadB(string FileName)
        {
            return double.Parse(File.ReadAllText(FileName));
        }
        /// <summary>
        /// Calculates the probabilty of prediction between 0 and 1.
        /// </summary>
        /// <param name="X"></param>
        /// <param name="W"></param>
        /// <param name="B"></param>
        /// <returns>Returns an 2D array of probability values. </returns>
        public static double[,] Model(double[,] X, double[,] W, double[,] B)
        {

            double[,] mult = MathCal.Matrix.MatrixMultip(X, W);
           
            double[,] z = MathCal.Matrix.MatrixAddToScalar(B[0, 0], mult);
           
            double[,] a = MathCal.Matrix.Sigmoid(z);


            return a;
        }

        static double LogLoss(double[,] a, double[,] y)
        {
            double sum = 0;

            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    double e = 1e-15;
                    Console.WriteLine(a[i, j]);
                    sum += y[i, j] * Math.Log(a[i, j] + e) + (1 - y[i, j]) * Math.Log(1 - a[i, j]);

                }
            }

            return -sum / y.GetLength(0);
        }

        static double[,] Gradients(double[,] y, double[,] a, double[,] x)
        {

            double[,] X = MathCal.Matrix.MatrixTrans(x);
            double[,] A = MathCal.Matrix.MatrixMultipbyScalar(-1, a);
            double[,] Y = MathCal.Matrix.MatrixAdd(y, A);
            double[,] res = MathCal.Matrix.MatrixMultip(X, Y);
           
            return MathCal.Matrix.MatrixMultipbyScalar(-1f / y.GetLength(0), res);
        }
        static double[,] Gradients(double[,] y, double[,] a)
        {


            double[,] A = MathCal.Matrix.MatrixMultipbyScalar(-1, a);
            double[,] AaddY = MathCal.Matrix.MatrixAdd(y, A);
            double[,] res = MathCal.Matrix.MatrixMultipbyScalar(-1f / y.GetLength(0), AaddY);

            return res;
        }

        static double[,] UpdateW(double[,] dw, double[,] w, double LearningRate)
        {
            double[,] dl = MathCal.Matrix.MatrixMultipbyScalar(-LearningRate, dw);
            double[,] W = MathCal.Matrix.MatrixAdd(w, dl);
            return W;
        }
        static double[,] UpdateB(double[,] db, double[,] b, double LearningRate)
        {

            double Bgradient = 0, bbrocast = 0;
            //brocasting
            for (int i = 0; i < db.GetLength(0); i++)
            {
                for (int j = 0; j < db.GetLength(1); j++)
                {
                    Bgradient += db[i, j];
                }
            }

            bbrocast = b[0, 0];
            double Scalarres = bbrocast + (-LearningRate * Bgradient);
            double[,] res = new double[b.GetLength(0), b.GetLength(1)];
            for (int i = 0; i < res.GetLength(0); i++)
            {
                for (int j = 0; j < res.GetLength(1); j++)
                {
                    res[i, j] = Scalarres;
                    //Console.WriteLine("up"+res[i,j]);
                }
            }

            return res;
        }
        /// <summary>
        /// Calculates predictions in boolean form.
        /// </summary>
        /// <param name="X"></param>
        /// <param name="W"></param>
        /// <param name="b"></param>
        /// <returns>Returns an 2D array of probability values in boolean form.</returns>
        public static bool[,] Predict(double[,] X, double[,] W, double[,] b)
        {
            double[,] A = Model(X, W, b);
            bool[,] result = new bool[A.GetLength(0), A.GetLength(1)];

            for (int i = 0; i < result.GetLength(0); i++)
            {
                for (int j = 0; j < result.GetLength(1); j++)
                {
                    result[i, j] = (A[i, j] >= 0.5);
                    

                }
                Console.WriteLine(i * 100 / result.GetLength(0) + "%");
            }
            return result;
        }

        /// <summary>
        /// Trains an artificial neuron for a number of iterations depending on the learning rate.
        /// </summary>
        /// <param name="X"></param>
        /// <param name="Y"></param>
        /// <param name="Learning_rate"></param>
        /// <param name="n_iter"></param>
        /// <returns>Returns an array of logloss values. </returns>
        public static double[] Artificial_neuron(double[,] X, double[,] Y, double Learning_rate = 0.1, int n_iter = 100)
        {
            List<double> L = new List<double>();

            double[,] w = InitialisationW(X);

            double[,] b = InitialisationB(X);
            

            for (int i = 0; i < n_iter; i++)
            {
                Console.WriteLine("A_N : "+i*100 / n_iter + "%");
                double[,] A = Model(X, w, b);
                L.Add(LogLoss(A, Y));
                double[,] dw = Gradients(Y, A, X);
                double[,] db = Gradients(Y, A);
                w = UpdateW(dw, w, Learning_rate);
                b = UpdateB(db, b, Learning_rate);
                
            }
            File.WriteAllText(Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) + @"\bias.txt",b[0,0]+"");
            string currentData = "";
            for (int i = 0; i < w.GetLength(0); i++)
            {
                
            
                for (int j=0;j< w.GetLength(1); j++)
                {
                    currentData += w[i, j]+";";
                    
                }
                currentData += "\n";
                
            }
            File.WriteAllText(Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) + @"\weights.txt",
                         currentData );
            bool[,] pred = Predict(X, w, b);
            
            
            //Console.WriteLine("percentage: " + MathCal.Functions.Accuracy(MathCal.Arr.Ravel(Y), MathCal.Arr.Ravel(pred)));
            double[] res = new double[L.Count];
            for (int i = 0; i < res.Length; i++)
            {
                res[i] = L[i];
            }

            return res;
        }


    
    }
}
