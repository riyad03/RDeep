using RigidWare.Mathf;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;

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

        public static double[][,] InitialisationW(double[,] X, int n1, int n2)
        {
            Console.WriteLine("Initializing: 0%");
            double[][,] res = new double[2][,];
            double[,] W1 = new double[n1, X.GetLength(0)];
            for (int i = 0; i < W1.GetLength(0); i++)
            {
                for (int j = 0; j < W1.GetLength(1); j++)
                {
                    double x = MathCal.RandomSND();

                    W1[i, j] = x;
                    //Thread.Sleep(2);
                    Console.WriteLine(i * 100 / W1.GetLength(0) + "%");
                }
            }
            double[,] W2 = new double[n2, n1];
            for (int i = 0; i < W2.GetLength(0); i++)
            {
                for (int j = 0; j < W2.GetLength(1); j++)
                {
                    double x = MathCal.RandomSND();

                    W2[i, j] = x;
                    //Thread.Sleep(2);
                    Console.WriteLine(i * 100 / W2.GetLength(0) + "%");
                }
            }

            res[0] = W1;
            res[1] = W2;
            return res;
        }
        /// <summary>
        /// Randomizes a Bias value from the standard normalization distribution.
        /// </summary>
        /// <param name="X"></param>
        /// <returns>Returns a 2D arrays whose elements are equal to the Bias value.</returns>
        public static double[,] InitialisationB(double[,] X)
        {
            double[,] B = new double[X.GetLength(0), 1];
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
        public static double[][,] InitialisationB(double[,] X, int n1, int n2)
        {
            double[,] B1 = new double[n1, 1];
            double[,] B2 = new double[n2, 1];
            B1[0, 0] = MathCal.RandomSND();
            for (int i = 0; i < B1.GetLength(0); i++)
            {
                for (int j = 0; j < B1.GetLength(1); j++)
                {
                    B1[i, j] = B1[0, 0];
                }
            }
            B2[0, 0] = MathCal.RandomSND();
            for (int i = 0; i < B2.GetLength(0); i++)
            {
                for (int j = 0; j < B2.GetLength(1); j++)
                {
                    B2[i, j] = B2[0, 0];
                }
            }
            double[][,] res = new double[2][,];
            res[0] = B1;
            res[1] = B2;
            return res;
        }

        public static void WriteParameters(double[][,] w, double[][,] b)
        {
            string[] sep = { "a", "b" };
            File.WriteAllText(Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) + @"\bias.txt", b[0][0, 0] + "");
            string currentData = "";
            for (int i = 0; i < w[0].GetLength(0); i++)
            {


                for (int j = 0; j < w[i].GetLength(0); j++)
                {

                    for (int k = 0; k < w[i].GetLength(1); k++)
                    {
                        currentData += w[i][j, k] + sep[i];
                    }
                    currentData += "\n";
                }


            }
            File.WriteAllText(Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) + @"\weights.neuNE",
                         currentData);
        }
        public static double[][,] ReadW1(string FileName)
        {
            int k, l, c;
            if (File.Exists(FileName))
            {
                char[] sep = { 'a', 'b' };
                string[] r = File.ReadAllLines(FileName);
                l = r.Length;
                double[][,] w = new double[sep.Length - 1][,];

                for (int m = 0; m < sep.Length; m++)
                {
                    for (int i = 0; i < r.Length; i++)
                    {
                        r[i] = r[i].Remove(r[i].Length - 1);
                        if (r[i].Contains(sep[m].ToString()))
                        {
                            string[] s = r[i].Split(sep[m]);
                            c = s.Length;

                            for (int j = 0; j < c; j++)
                            {
                                if (s[j] != "")
                                {
                                    w[m][i, j] = double.Parse(s[j]);
                                }

                            }
                        }
                        else
                        {
                            w[m][i, 0] = double.Parse(r[i]);

                        }
                    }

                }
                return w;
            }
            else
            {
                throw new FileNotFoundException(FileName + " doesn't Exist");
            }
        }
        /// <summary>
        /// Reads Weight values that have been saved in FileName.
        /// </summary>
        /// <param name="FileName"></param>
        /// <returns>Returns Weight values from FileName.</returns>
        /// <exception cref="FileNotFoundException"></exception>
        public static double[,] ReadW(string FileName)
        {
            int l, c;
            if (File.Exists(FileName))
            {
                string[] r = File.ReadAllLines(FileName);
                l = r.Length;
                double[,] w = new double[l, r[0].Split(';').Length - 1];

                for (int i = 0; i < r.Length; i++)
                {
                    r[i] = r[i].Remove(r[i].Length - 1);
                    if (r[i].Contains(";"))
                    {
                        string[] s = r[i].Split(';');
                        c = s.Length;

                        for (int j = 0; j < c; j++)
                        {
                            if (s[j] != "")
                            {
                                w[i, j] = double.Parse(s[j]);
                            }

                        }
                    }
                    else
                    {
                        w[i, 0] = double.Parse(r[i]);

                    }

                }

                return w;
            }
            else
            {
                throw new FileNotFoundException(FileName + " doesn't Exist");
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

        public static double[][,] ForwardPropagation(double[,] X, double[][,] W, double[][,] B)
        {

            double[,] mult1 = MathCal.Matrix.MatrixMultip(W[0], X);
            double[,] Z1 = MathCal.Matrix.MatrixAddToScalar(B[0][0, 0], mult1);
            double[,] A1 = MathCal.Matrix.Sigmoid(Z1);

            double[,] mult2 = MathCal.Matrix.MatrixMultip(W[1], A1);
            double[,] Z2 = MathCal.Matrix.MatrixAddToScalar(B[1][0, 0], mult2);
            double[,] A2 = MathCal.Matrix.Sigmoid(Z2);

            double[][,] res = new double[2][,];
            res[0] = A1;
            res[1] = A2;
            return res;
        }

        public static double[][,] BackwardPropagation(double[,] X, double[,] Y, double[][,] A, double[][,] W)
        {
            double[,] mY = MathCal.Matrix.MatrixMultipbyScalar(-1, Y);
            //Console.WriteLine("w: "+w[GetLength(0) + " " + mY.GetLength(1));
            double[,] dz2 = MathCal.Matrix.MatrixAdd(A[1], mY);
            double[,] dw2 = MathCal.Matrix.MatrixMultipbyScalar(1f / Y.GetLength(1), MathCal.Matrix.MatrixMultip(dz2, MathCal.Matrix.MatrixTrans(A[0])));

            double[,] WT = MathCal.Matrix.MatrixTrans(W[1]);
            //1-A1
            double[,] MinA = MathCal.Matrix.MatrixMultipbyScalar(-1, A[0]);
            double[,] op1 = MathCal.Matrix.MatrixAddToScalar(1, MinA);
            //A1(1-A1)
            double[,] op2 = MathCal.Matrix.ElementaryMultip(A[0], op1);

            double[,] dz1 = MathCal.Matrix.ElementaryMultip(MathCal.Matrix.MatrixMultip(WT, dz2), op2);
            double[,] XT = MathCal.Matrix.MatrixTrans(X);
            double[,] dw1 = MathCal.Matrix.MatrixMultipbyScalar(1f / Y.GetLength(1), MathCal.Matrix.MatrixMultip(dz1, XT));

            double[][,] result = new double[2][,];

            result[0] = dw1;
            result[1] = dw2;

            return result;
        }
        public static double[][,] BackwardPropagation(double[,] Y, double[][,] A, double[][,] W)
        {
            double[,] mY = MathCal.Matrix.MatrixMultipbyScalar(-1f, Y);

            double[,] dz2 = MathCal.Matrix.MatrixAdd(A[1], mY);
            double[,] db2 = new double[dz2.GetLength(0), 1];

            //broadcasting
            for (int i = 0; i < dz2.GetLength(0); i++)
            {
                db2[i, 0] = 0;
                for (int j = 0; j < dz2.GetLength(1); j++)
                {
                    db2[i, 0] += dz2[i, j];
                }
                db2[i, 0] *= 1f / Y.GetLength(1);
            }


            //1-A1
            double[,] MinA = MathCal.Matrix.MatrixMultipbyScalar(-1, A[0]);
            double[,] op1 = MathCal.Matrix.MatrixAddToScalar(1, MinA);
            //A1(1-A1)
            double[,] op2 = MathCal.Matrix.ElementaryMultip(A[0], op1);
            double[,] w2T = MathCal.Matrix.MatrixTrans(W[1]);
            double[,] mult = MathCal.Matrix.MatrixMultip(w2T, dz2);
            double[,] dz1 = MathCal.Matrix.ElementaryMultip(mult, op2);

            double[,] db1 = new double[dz1.GetLength(0), 1];

            //broadcasting
            for (int i = 0; i < dz1.GetLength(0); i++)
            {
                db1[i, 0] = 0;
                for (int j = 0; j < dz1.GetLength(1); j++)
                {
                    db1[i, 0] += dz1[i, j];
                }
                db1[i, 0] *= 1f / Y.GetLength(1);
            }


            double[][,] result = new double[2][,];
            result[0] = db1;
            result[1] = db2;


            return result;
        }
        static double LogLoss(double[,] a, double[,] y)
        {
            double sum = 0;
            double[,] nY = new double[a.GetLength(0), a.GetLength(1)];
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    nY[i, j] = y[0, i];
                }
            }
            //Console.WriteLine("y1 " + y.GetLength(1));
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    double e = 1e-15;
                    Console.WriteLine(a[i, j]);
                    sum += nY[i, j] * Math.Log(a[i, j] + e) + (1 - nY[i, j]) * Math.Log(1 - nY[i, j]);

                }
            }

            return -sum / y.GetLength(1);
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
        static double[][,] UpdateW(double[][,] dw, double[][,] w, double LearningRate)
        {
            double[,] dl1 = MathCal.Matrix.MatrixMultipbyScalar(-LearningRate, dw[0]);

            double[,] W1 = MathCal.Matrix.MatrixAdd(w[0], dl1);


            double[,] dl2 = MathCal.Matrix.MatrixMultipbyScalar(-LearningRate, dw[1]);
            double[,] W2 = MathCal.Matrix.MatrixAdd(w[1], dl2);

            double[][,] result = new double[2][,];

            result[0] = W1;
            result[1] = W2;
            return result;
        }
        static double[][,] UpdateB(double[][,] db, double[][,] b, double LearningRate)
        {
            double[,] dl1 = MathCal.Matrix.MatrixMultipbyScalar(-LearningRate, db[0]);
            //broadcasting
            double[,] bb1 = new double[dl1.GetLength(0), dl1.GetLength(1)];
            int bbi = 0, bbj = 0;
            for (int i = 0; i < b[0].GetLength(0); i++)
            {
                bbj = 0;
                for (int j = 0; j < b[0].GetLength(1); j++)
                {
                    bb1[bbi, bbj] = b[0][0, j];
                    bbj++;
                }
                bbi++;
            }
            double[,] B1 = MathCal.Matrix.MatrixAdd(bb1, dl1);

            double[,] dl2 = MathCal.Matrix.MatrixMultipbyScalar(-LearningRate, db[1]);
            double[,] B2 = MathCal.Matrix.MatrixAdd(b[1], dl2);

            double[][,] result = new double[2][,];

            result[0] = B1;
            result[1] = B2;
            return result;
        }
        static double[,] UpdateB(double[,] db, double[,] b, double LearningRate)
        {

            double Bgradient = 0, bbrocast = 0;
            //broadcasting
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
        public static bool[][,] Predict(double[,] X, double[][,] W, double[][,] b)
        {
            double[][,] A = ForwardPropagation(X, W, b);
            bool[][,] result = new bool[2][,];
            bool[,] res1 = new bool[A[0].GetLength(0), A[0].GetLength(1)];
            bool[,] res2 = new bool[A[1].GetLength(0), A[1].GetLength(1)];

            for (int i = 0; i < res1.GetLength(0); i++)
            {
                for (int j = 0; j < res1.GetLength(1); j++)
                {
                    res1[i, j] = (A[0][i, j] >= 0.5);
                }
            }
            for (int i = 0; i < res2.GetLength(0); i++)
            {
                for (int j = 0; j < res2.GetLength(1); j++)
                {
                    res2[i, j] = (A[1][i, j] >= 0.5);
                    //Console.WriteLine("A[ " + i + "," + j + "]" + (A[1][i, j] >= 0.5f));
                }
            }
            result[0] = res1;
            result[1] = res2;
            return result;
        }

        public static string[] BackTranslate(double[][,] Activations, string[] Feelings, int n)
        {
            Console.WriteLine("f " + (Feelings.Length - 1));
            string[] res = new string[Activations[n - 1].GetLength(1)];
            Console.WriteLine("res " + res.Length);
            double max = 0;
            int si = 0;
            for (int j = 0; j < Activations[1].GetLength(1); j++)
            {
                max = Activations[1][0, j];
                si = 0;
                for (int i = 0; i < Activations[1].GetLength(0); i++)
                {

                    if (max < Activations[1][i, j])
                    {
                        max = Activations[1][i, j];

                        si = i;
                    }


                }
                Console.WriteLine("max " + max);
                Console.WriteLine("i " + si);
                res[j] = Feelings[si];
            }
            return res;
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
                Console.WriteLine("A_N : " + i * 100 / n_iter + "%");
                double[,] A = Model(X, w, b);
                L.Add(LogLoss(A, Y));
                double[,] dw = Gradients(Y, A, X);
                double[,] db = Gradients(Y, A);
                w = UpdateW(dw, w, Learning_rate);
                b = UpdateB(db, b, Learning_rate);

            }
            File.WriteAllText(Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) + @"\bias.txt", b[0, 0] + "");
            string currentData = "";
            for (int i = 0; i < w.GetLength(0); i++)
            {


                for (int j = 0; j < w.GetLength(1); j++)
                {
                    currentData += w[i, j] + ";";

                }
                currentData += "\n";

            }
            File.WriteAllText(Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) + @"\weights.txt",
                         currentData);
            bool[,] pred = Predict(X, w, b);


            //Console.WriteLine("percentage: " + MathCal.Functions.Accuracy(MathCal.Arr.Ravel(Y), MathCal.Arr.Ravel(pred)));
            double[] res = new double[L.Count];
            for (int i = 0; i < res.Length; i++)
            {
                res[i] = L[i];
            }

            return res;
        }

        public static double[] CreateNeuralNetwork(double[,] X, string[] Feelings, double[,] Y, int n1, double Learning_rate = 0.1, int n_iter = 100)
        {
            List<double> L = new List<double>();

            double[][,] w = InitialisationW(X, n1, Y.GetLength(0));
            double[][,] b = InitialisationB(X, n1, Y.GetLength(0));

            for (int i = 0; i < n_iter; i++)
            {
                Console.WriteLine("A_N : " + i * 100 / n_iter + "%");
                double[][,] Activations = ForwardPropagation(X, w, b);

                if (i % 10 == 0)
                {
                    L.Add(LogLoss(Activations[1], Y));
                }
                double[][,] dw = BackwardPropagation(X, Y, Activations, w);
                double[][,] db = BackwardPropagation(Y, Activations, w);
                w = UpdateW(dw, w, Learning_rate);
                b = UpdateB(db, b, Learning_rate);

            }
            double[][,] Activ = ForwardPropagation(X, w, b);
            /*for (int j = 0; j < Activ[1].GetLength(0); j++)
            {
                for (int c = 0; c < Activ[1].GetLength(1); c++)
                {
                    Console.WriteLine("A[: " + j + "," + c + "] = " + Activ[1][j, c]);
                }
            }*/
            string[] f = BackTranslate(Activ, Feelings, w.GetLength(0));
            for (int i = 0; i < f.Length; i++)
            {
                Console.WriteLine(i + " " + f[i]);
            }
            Console.WriteLine("b:   " + b[1][0, 0]);

            /* File.WriteAllText(Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) + @"\bias.txt", b[0, 0] + "");
             string currentData = "";
             for (int i = 0; i < w.GetLength(0); i++)
             {


                 for (int j = 0; j < w.GetLength(1); j++)
                 {
                     currentData += w[i, j] + ";";

                 }
                 currentData += "\n";

             }
             File.WriteAllText(Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) + @"\weights.txt",
                          currentData);*/
            bool[][,] pred = Predict(X, w, b);


            Console.WriteLine("percentage: " + MathCal.Functions.Accuracy(MathCal.Arr.Ravel(Y), MathCal.Arr.Ravel(pred[1])));
            double[] res = new double[L.Count];
            for (int i = 0; i < res.Length; i++)
            {
                res[i] = L[i];
            }

            return res;
        }

        public static class DLMultipleLayers
        {
            public static double[][,] InitialisationW(int[] LayersLength)
            {
                Console.WriteLine("Initializing: 0%");
                double[][,] res = new double[LayersLength.Length][,];
                double[,] w0 = new double[1, 1];
                w0[0, 0] = 0.1;
                res[0] = w0;
                for (int l = 1; l < LayersLength.Length; l++)
                {
                    double[,] W = new double[LayersLength[l], LayersLength[l - 1]];

                    for (int i = 0; i < W.GetLength(0); i++)
                    {
                        for (int j = 0; j < W.GetLength(1); j++)
                        {
                            /*double x = MathCal.RandomSND();
                            
                            W[i, j] = x;*/
                            Random g=new Random();
                            double rand =1f* g.Next(-1573, 1433) / 1000;
                            Console.WriteLine("w length" + W.GetLength(1));
                            W[i, j] = rand * Math.Sqrt(2f / W.GetLength(1));
                            //W[i, j] = rand;
                            Console.WriteLine("W " + W[i, j]);
                            //Thread.Sleep(2);
                            Console.WriteLine("Initializing : " + j * 100 / W.GetLength(1) + "%");
                        }
                    }
                    


                    res[l] = W;
                    
                }
               
                return res;
            }
            public static double[][,] InitialisationB(int[] LayersLength)
            {
                double[][,] res = new double[LayersLength.Length][,];
                for (int l = 1; l < LayersLength.Length; l++)
                {
                    double[,] B = new double[LayersLength[l], 1];
                    B[0, 0] = MathCal.RandomSND();
                    for (int i = 0; i < B.GetLength(0); i++)
                    {
                        for (int j = 0; j < B.GetLength(1); j++)
                        {
                            B[i, j] = B[0, 0];
                        }
                    }

                    res[l] = B;

                }

                return res;
            }
            public static double MeanOfWeights(double[,] W,double[,] X)
            {
                double z = 0;
                double sw = 0;
                for(int i=0;i< W.GetLength(0); i++)
                {
                    for(int j=0;j< W.GetLength(1); j++)
                    {
                        z+=W[i,j]*X[i,j];
                        sw+=W[i,j];
                    }
                }
                return 1f* z/sw;
            }
            public static double[][,] ForwardPropagation(double[,] X, double[][,] W, double[][,] B)
            {
               
                double[][,] res = new double[W.GetLength(0)][,];
                double[][,] Z = new double[W.GetLength(0) + 1][,];
                double[][,] A = new double[W.GetLength(0)][,];
                A[0] = X;
                Console.WriteLine("A ");
                for (int l = 1; l < W.GetLength(0); l++)
                {
                    Console.WriteLine("l " + (l));
                    //7ala la5ra
                   
                    double[,] mult = MathCal.Matrix.MatrixMultip(W[l], A[l - 1]);
                    double[,] bb = new double[mult.GetLength(0), mult.GetLength(1)];
                    //broadcasting
                    for (int i = 0; i < mult.GetLength(0); i++)
                    {
                        for (int j = 0; j < mult.GetLength(1); j++)
                        {
                            bb[i, j] = B[l][i, 0];
                        }

                    }
                    MathCal.Functions.ReLU(10);
                    Z[l] = MathCal.Matrix.MatrixAdd(bb, mult);
                    A[l] = MathCal.Matrix.LeakyReLU(Z[l],0.1);
                   
                
                    /*for (int i = 0; i < Z[l].GetLength(0); i++)
                    {
                        for (int j = 0; j < Z[l].GetLength(1); j++)
                        {
                            Console.WriteLine("Z" + l +"[ "+i+", "+j+"]" + Z[l][i, j]);
                        }
                        Console.WriteLine("]]]]]]]]]]]");
                    }
                    for (int i = 0; i < A[l].GetLength(0); i++)
                    {
                        for (int j = 0; j < A[l].GetLength(1); j++)
                        {
                            Console.WriteLine("A" + l + " " + A[l][i, j]);
                        }
                        Console.WriteLine("]]]]]]]]]]]");
                    }*/
                    //A[l] = MathCal.Matrix.Sigmoid(Z[l]);

                    /*if (l == 1)
                    {
                        for (int i = 0; i < B[l].GetLength(0); i++)
                        {
                            for (int j = 0; j < B[l].GetLength(1); j++)
                            {
                                Console.WriteLine("b" + l + " " + B[l][i, j]);
                            }
                            Console.WriteLine("]]]]]]]]]]]");
                        }
                        for (int i = 0; i < mult.GetLength(0); i++)
                        {
                            for (int j = 0; j < mult.GetLength(1); j++)
                            {
                                Console.WriteLine("mult" + (l-1) + " " + mult[i, j]);
                            }
                            Console.WriteLine("]]]]]]]]]]]");
                        }
                        for (int i = 0; i < Z[l].GetLength(0); i++)
                        {
                            for (int j = 0; j < Z[l].GetLength(1); j++)
                            {
                                Console.WriteLine("z" + l + " " + Z[l][i, j]);
                            }
                            Console.WriteLine("]]]]]]]]]]]");
                        }
                    }*/

                    res[l] = A[l];

                }

                res[0] = A[0];
                return res;
            }

            public static double[][,] BackwardPropagation(double[,] Y, double[][,] A, double[][,] W)
            {
                double[][,] result = new double[W.GetLength(0)][,];
                double[,] mY = MathCal.Matrix.MatrixMultipbyScalar(-1, Y);
                double[][,] dz = new double[W.GetLength(0)][,];
                double[][,] dw = new double[dz.GetLength(0)][,];
                double[,] dw0 = new double[1, 1];
                dw0[0, 0] = 0.1;
                dw[0] = dw0;
                Console.WriteLine("w l " + (W.GetLength(0) - 1));
                dz[W.GetLength(0) - 1] = MathCal.Matrix.MatrixAdd(MathCal.Matrix.DerivativeLeakyReLU(A[W.GetLength(0) - 1], 0), mY);

                for (int l = W.GetLength(0) - 1; l >= 1; l--)
                {
                    double[,] dA = MathCal.Matrix.DerivativeLeakyReLU(A[l-1], 0.1);

                    //Console.WriteLine("w: "+w[GetLength(0) + " " + mY.GetLength(1));
                    Console.WriteLine(l);
                    dw[l] = MathCal.Matrix.MatrixMultipbyScalar(1f / Y.GetLength(1), MathCal.Matrix.MatrixMultip(dz[l], MathCal.Matrix.MatrixTrans(dA)));
                    //1-A1
                    double[,] MinA = MathCal.Matrix.MatrixMultipbyScalar(-1, dA);
                    double[,] op1 = MathCal.Matrix.MatrixAddToScalar(1, MinA);
                    //A1(1-A1)
                    double[,] op2 = MathCal.Matrix.ElementaryMultip(dA, op1);
                    double[,] w2T = MathCal.Matrix.MatrixTrans(W[l]);
                    double[,] mult2 = MathCal.Matrix.MatrixMultip(w2T, dz[l]);
                    dz[l - 1] = MathCal.Matrix.ElementaryMultip(mult2, op2);// MathCal.Matrix.ElementaryMultip(mult, op2);
                    result[l] = dw[l];
                    

                }
                

                return result;
            }


            public static double[][,] BackwardPropagation(double[,] Y, double[][,] A, double[][,] W, double[][,] B)
            {
                double[][,] result = new double[B.GetLength(0)][,];


                double[][,] dz = new double[B.GetLength(0)][,];
                double[][,] db = new double[dz.GetLength(0)][,];
                double[,] mY = MathCal.Matrix.MatrixMultipbyScalar(-1f, Y);
                //mabghinach db0 tkon null
                double[,] db0 = new double[1, 1];
                db0[0, 0] = 0.1;
                db[0] = db0;

                dz[B.GetLength(0) - 1] = MathCal.Matrix.MatrixAdd(MathCal.Matrix.DerivativeLeakyReLU(A[B.GetLength(0) - 1],0), mY);


                for (int l = B.GetLength(0) - 1; l >= 1; l--)
                {
                    double[,] dA = MathCal.Matrix.DerivativeLeakyReLU(A[l - 1], 0.1);
                    Console.WriteLine(l);
                    double[,] miniB = new double[dz[l].GetLength(0), 1];
                    db[l] = miniB;

                    for (int i = 0; i < db[l].GetLength(0); i++)
                    {
                        db[l][i, 0] = 0;
                        for (int j = 0; j < dz[l].GetLength(1); j++)
                        {
                            db[l][i, 0] += dz[l][i, j];
                        }
                        db[l][i, 0] *= 1f / Y.GetLength(1);
                    }

                    //1-A1
                    double[,] MinA = MathCal.Matrix.MatrixMultipbyScalar(-1, dA);
                    double[,] op1 = MathCal.Matrix.MatrixAddToScalar(1, MinA);

                    //A1(1-A1)
                    double[,] op2 = MathCal.Matrix.ElementaryMultip(dA, op1);

                    double[,] w2T = MathCal.Matrix.MatrixTrans(W[l]);

                    double[,] mult2 = MathCal.Matrix.MatrixMultip(w2T, dz[l]);

                    /*double[,] newmult = new double[op2.GetLength(0), op2.GetLength(1)];
                    //broadcasting
                    for (int i = 0; i < newmult.GetLength(0); i++)
                    {
                        for (int j = 0; j < newmult.GetLength(1); j++)
                        {
                            newmult[i, j] = mult2[0, j];
                        }
                    }*/
                    dz[l - 1] = MathCal.Matrix.ElementaryMultip(mult2, op2);// MathCal.Matrix.ElementaryMultip(mult, op2);



                    result[l] = db[l];
                   

                }



                return result;
            }

            static double LogLoss(double[,] a, double[,] y)
            {
                double sum = 0;
                double[,] nY = new double[a.GetLength(0), a.GetLength(1)];
               
                for (int i = 0; i < a.GetLength(0); i++)
                {
                    for (int j = 0; j < a.GetLength(1); j++)
                    {
                        Console.WriteLine("a " + a[i,j]);
                        nY[i, j] = y[0, i];
                    }
                }
                //Console.WriteLine("y1 " + y.GetLength(1));
                for (int i = 0; i < a.GetLength(0); i++)
                {
                    for (int j = 0; j < a.GetLength(1); j++)
                    {
                        double e = 1e-15;
                        Console.WriteLine(a[i, j]);
                        sum += nY[i, j] * Math.Log(a[i, j] + e) + (1 - nY[i, j]) * Math.Log(1 - nY[i, j]);

                    }
                }

                return -sum / y.GetLength(1);
            }
            static double ReluLogLoss(double[,] a, double[,] y)
            {
                double sum = 0;
                for (int i = 0; i < a.GetLength(0); i++)
                {
                    for (int j = 0; j < a.GetLength(1); j++)
                    {
                        sum += (y[i, j] - a[i, j]) * (y[i, j] - a[i, j]);

                    }
                }
               
                return sum/y.GetLength(1);
            }
            public static double[][,] UpdateW(double[][,] dw, double[][,] W, double LearningRate)
            {
                double[][,] res = new double[W.GetLength(0)][,];
                double[][,] wt = new double[W.GetLength(0)][,];
                for (int l = 1; l < W.GetLength(0); l++)
                {
                    double[,] FirstMult = MathCal.Matrix.MatrixMultipbyScalar(-LearningRate, dw[l]);
                    wt[l] = MathCal.Matrix.MatrixAdd(W[l], FirstMult);
                    res[l] = wt[l];
                    
                }


                return res;
            }
            public static double[][,] UpdateB(double[][,] db, double[][,] B, double LearningRate)
            {
                double[][,] res = new double[B.GetLength(0)][,];
                double[][,] bt = new double[B.GetLength(0)][,];
                for (int l = 1; l < B.GetLength(0); l++)
                {
                    double[,] FirstMult = MathCal.Matrix.MatrixMultipbyScalar(-LearningRate, db[l]);
                    bt[l] = MathCal.Matrix.MatrixAdd(B[l], FirstMult);
                    res[l] = bt[l];
                    

                }
                return res;
            }
            static bool[,] Predict(double[,] X, double[][,] W, double[][,] B)
            {
                double[][,] A = ForwardPropagation(X, W, B);
                bool[,] res = new bool[A[W.GetLength(0) - 1].GetLength(0), A[W.GetLength(0) - 1].GetLength(1)];
                double max = 0;
                int si = 0, sj = 0; ;

                if (A[A.GetLength(0) - 1].GetLength(0) == 1)
                {
                    for (int i = 0; i < res.GetLength(0); i++)
                    {
                        for (int j = 0; j < res.GetLength(1); j++)
                        {
                            res[i, j] = (A[W.GetLength(0)-1][i, j] > 0.5f);

                        }
                    }
                }
                else
                {
                    for (int j = 0; j < A[A.GetLength(0) - 1].GetLength(1); j++)
                    {
                        max = A[A.GetLength(0) - 1][0, j];
                        si = 0;
                        sj = j;
                        for (int i = 0; i < A[A.GetLength(0) - 1].GetLength(0); i++)
                        {

                            if (max < A[A.GetLength(0) - 1][i, j])
                            {
                                max = A[A.GetLength(0) - 1][i, j];

                                si = i;
                                sj = j;
                            }


                        }
                        Console.WriteLine("max " + max);
                        Console.WriteLine("i " + si);
                        Console.WriteLine("j " + sj);
                        res[si, sj] = true;
                    }
                }

                return res;
            }
            public static string[] BackTranslate(double[][,] Activations, string[] Feelings, int n)
            {
                Console.WriteLine("f " + (Feelings.Length - 1));
                string[] res = new string[Activations[n - 1].GetLength(1)];
                Console.WriteLine("res " + res.Length);
                double max = 0;
                int si = 0;
                for (int j = 0; j < Activations[Activations.GetLength(0) - 1].GetLength(1); j++)
                {
                    max = Activations[Activations.GetLength(0) - 1][0, j];
                    si = 0;
                    for (int i = 0; i < Activations[Activations.GetLength(0) - 1].GetLength(0); i++)
                    {

                        if (max < Activations[Activations.GetLength(0)-1][i, j])
                        {
                            max = Activations[Activations.GetLength(0)-1][i, j];

                            si = i;
                        }


                    }
                    Console.WriteLine("max " + max);
                    Console.WriteLine("i " + si);
                    res[j] = Feelings[si];
                }
                return res;
            }
            static void TableWriter(double[][,] X)
            {
                for(int l = 1; l < X.GetLength(0); l++)
                {
                    for(int i = 0; i < X[l].GetLength(0); i++)
                    {
                        for(int j = 0; j < X[l].GetLength(1); j++)
                        {
                            //Console.WriteLine(l+" "+X[l][i, j]);
                        }
                    }
                }
            }
            public static double[] CreateNeuralNetwork(double[,] X, double[,] Y, int[] Settings,string[] Feelings, double Learning_rate = 0.1, int n_iter = 100)
            {
                List<double> L = new List<double>();
                
                double[][,] W = InitialisationW(Settings);
                double[][,] b = InitialisationB(Settings);

                
              
              
                /*double[][,] W = new double[Settings.Length][,];
                double[][,] b = new double[Settings.Length][,];

                b[0] = new double[,] {{0 }};
                W[0] = new double[,] { { 0 } };
                b[1] = new double[,] { { -0.39034028 }, { 0.27012213 }, { 0.52885141 }, { -0.00381169 }, { 0.32072878 }, { 0.89776541 }, { 0.12816886 }, { 0.65162087 }, { 0.32433998 }, { 0.27139626 }, { -1.72397348 }, { -1.15973527 }, { -0.33505235 }, { 0.5563923 }, { 0.43358567 }, { -0.27449065 } };
                W[1] = new double[,] { { -0.94452577, -0.87606713 }, { 0.82634782, -0.32030134 }, { -1.08006662, 1.3074388, }, { -0.07225623, 1.63308354 }, { 1.20904614, -0.96658271 }, { -1.49342176, -0.52399669 }, { 0.00396547, -0.29354598 }, { 0.06210525, -0.43041732 }, { 0.10936522, -0.06373114 }, { 0.74193271, -0.73012834 }, { -1.29528197, 0.58434031 }, { 0.26375213, 0.91910017 }, { -0.02844042, -0.28674238 }, { 2.01453284, 1.43729059 }, { -1.87134132, -0.03366467 }, { 0.01709842, -0.66984549 } };
                b[2] = new double[,] { { 1.67992279 }, { 0.2625831 }, { 2.336138 }, { 0.00930725 }, { -1.34304892 }, { -0.17696948 }, { -1.37558188 }, { 1.53199926 }, { -1.48549594 }, { 0.15564508 }, { -2.69498223 }, { -0.36506059 }, { -0.79353054 }, { 0.83464282 }, { -1.37633299 }, { -0.44699619 } };
                W[2] = new double[,] { { 0.69336586, -0.46653343, 0.19203319, 1.20270081, 1.60551056, 0.49406682, -1.41976145, 0.17544729, -1.64455943, 0.75984668, -0.065688, 0.93295548, -1.14227279, 0.4310864, -0.01086248, -0.57054659 }, { -0.57640405, 0.04922309, -0.38533714, 0.18116646, 0.34846918, -0.83239317, 2.41298958, -1.1835557, -0.0492048, 0.0593841, 0.82002523, 1.42974566, 0.13249088, -0.65830871, 1.26089197, -0.32272883 }, { 0.6320702, -0.92820971, -0.43732303, -2.93370832, 1.69583124, 0.85901116, -0.0943386, -0.57940956, -0.02962218, -1.09392044, -0.54900857, -0.35283628, 0.56065758, -0.29138956, 0.81634648, 0.12721578 }, { 1.09297946, -0.79245613, 1.15351008, 0.6284239, -0.1899588, -0.08785131, -1.06180204, -0.53999671, -0.77880268, 0.46723746, 0.56964033, 0.10043964, -1.73249109, 1.18768283, -1.7420005, 0.44128689 }, { -1.23331964, -0.36362487, 0.17870623, 0.93061421, 1.06609648, 0.27024577, 1.12966372, -0.43884747, -0.03591876, 0.08540058, -1.66960314, 0.80808033, 1.16686458, -0.09064046, -1.2165857, -0.11240827 }, { -0.65086436, -0.37608184, -0.44632831, -0.04780062, 1.31086711, 2.31112381, 1.03812184, 1.17081664, -1.30672468, 0.04586026, -0.4047235, 0.2107129, 0.4723715, -0.79144304, -1.45778286, 0.80398554 }, { -1.92209438, 0.17593601, 0.76020161, -0.13829781, 1.15824167, -1.18938842, 1.35722736, 0.1185128, 2.16893091, -1.10785572, -1.15229783, 0.01613376, 0.41234501, 0.09810323, 1.15431171, 0.61614412 }, { -0.89513665, 1.16977723, 0.42055807, -0.8106355, 0.52989227, 0.25824467, 0.12680512, 0.72548298, -1.19282092, -0.56420017, -0.56854611, -0.96075064, 0.2465762, -0.52025597, -0.05588247, -0.53369676 }, { 0.81660079, -0.03244439, -1.38167319, 0.14386475, -0.93435297, 1.59763634, 0.94825425, -1.48513587, -0.03728829, 2.36475294, -0.77583047, -1.58712038, 0.12198994, 0.05612866, 1.0711359, -0.6643377, }, { -0.13762243, -1.27361745, 0.59325014, -0.02476092, -0.02464307, 0.34176771, 1.07359324, 1.31640295, -0.64959185, 1.1159755, -0.40686837, 0.32728514, 0.95267561, 1.56898795, 0.87929222, -0.46077758 }, { -1.06761759, -0.91432744, -1.96869625, -0.1551435, -0.728998, -1.63732738, 0.84506769, -0.14267132, -0.4583781, -0.02464699, -0.55646257, -1.37219223, -0.86038428, 0.20645724, -0.51693764, -0.87837973 }, { 0.86651175, 0.3430052, 0.08567406, 1.71783005, 0.28017398, 1.43211322, -1.84959267, -0.70579374, -2.75925625, 1.39282706, 0.10162573, -0.16950307, -0.61709685, 0.4309927, 0.50953692, -0.07905433 }, { 0.85156261, 0.28556484, -0.22823805, -0.6632877, -0.55463132, -0.47955521, 1.91600346, -0.42766087, -0.55497065, 0.82811237, 0.80304263, 0.10669455, -0.58127726, 0.19469161, -0.84917328, 2.49450437 }, { 0.12085011, 0.62986265, -0.83268692, -0.7631159, 0.74663195, 0.0493315, 1.19213173, 0.38336593, 1.12376922, 1.15305672, 2.03878714, 1.23090024, 2.01219186, -0.83626263, 1.10617068, -0.59381387 }, { 1.36845776, 0.01482374, 0.8331693, 1.69707227, -0.84902847, 0.65841984, 0.75785837, -0.69477404, -1.11203521, -0.22446283, -0.93816717, 0.15776517, 1.40611743, 0.22520304, -0.11633592, -0.18346647 }, { -0.81796257, 0.13554745, 0.87237075, 0.55035271, -0.87206112, 0.41252379, -0.02688346, -1.57400958, 0.31877894, -0.24708173, 0.23268195, -1.82883178, -0.07632906, 1.26923989, 1.98890601, 1.00510571 } };
                b[3] = new double[,] { { -0.51579593 }, { 0.01150601 }, { -0.69074898 }, { 0.60154706 }, { 0.5220865 }, { 0.02625365 }, { -1.63820676 }, { 0.61012862 }, { 1.66141917 }, { -1.37102425 }, { -0.03863278 }, { 0.67936772 }, { -0.26378714 }, { 0.01216299 }, { -1.04809373 }, { 1.08531537 } };
                W[3] = new double[,] { { 1.27324366e+00, -1.13595871e+00, 1.10327089e-01, 5.48967532e-02, 5.34442447e-01, 6.19586501e-01, -5.35410745e-01, 9.95336009e-01, 7.79413062e-01, -1.02650280e+00, -1.35134030e-01, 8.09442715e-01, 1.25258951e+00, -2.23999711e-02, 8.91352433e-01, -4.84877886e-01 }, { -1.79803887e+00, 3.19391729e+00, -4.23687033e-01, -1.88941260e-01, 5.65079778e-01, 5.31723645e-01, 7.06776389e-01, 5.49170154e-01, -1.66961883e+00, -2.01260085e+00, -1.37598801e-01, 2.11380476e+00, 1.49535084e+00, -1.81715348e-01, -1.26041984e+00, -1.27416419e-01 }, { 2.10066611e-01, -2.16000113e-01, -7.46111890e-01, 3.90470960e-01, 9.25911928e-01, 8.94421571e-01, 1.40300773e+00, 5.02252064e-01, -3.98092858e-01, -3.79156415e-01, 2.72807164e-01, -6.38210953e-01, 2.17658368e-01, -1.01566454e+00, 4.12958692e-01, 4.07451875e-01 }, { 1.24229889e+00, -8.40645569e-01, 3.09275286e-01, 5.58467922e-02, 8.07770046e-01, -1.07077323e+00, -1.82880237e-01, 5.32089805e-01, -1.20328992e+00, -2.69205236e-01, -1.28258614e+00, -6.64691246e-01, -1.81875064e-01, -1.41389403e-01, -2.15033237e+00, -1.61035659e+00 }, { 2.12877190e+00, -1.37116312e+00, 6.29718255e-02, -4.54564136e-01, 4.99785011e-01, 5.49664157e-01, -6.55006421e-01, -1.59162310e+00, -7.93960314e-01, -1.51820173e-01, -5.76284287e-01, -8.52303827e-01, -4.91675064e-01, 5.48850731e-01, -1.08685125e+00, 8.51054493e-01 }, { -4.30408301e-01, 2.90523139e-01, -1.16148941e+00, -5.25100051e-01, -1.83154132e+00, -3.64931873e-01, 1.23569140e+00, -5.92818220e-01, 5.01749760e-02, -7.04001349e-01, -2.48786900e+00, -1.58253473e-01, -1.26226037e+00, 1.05193726e+00, 5.40172236e-01, -2.11037552e-01 }, { -4.73271988e-01, 7.98135757e-01, 1.20982621e+00, 2.46105952e-03, -3.25588499e-01, -5.58315592e-01, -5.77214357e-01, 1.35908824e+00, 1.68063443e-01, -1.02774011e-01, 4.16438146e-01, 5.81835499e-01, 2.71675472e+00, 1.38323352e+00, -8.45575344e-03, -8.66326351e-01 }, { -2.80821380e-01, 7.34261484e-01, -1.10670369e-01, -3.94876320e-01, -9.30631145e-01, 1.00635912e+00, -9.78903246e-01, -1.34458066e+00, -3.10034305e+00, -5.54432497e-01, 1.00476440e-01, 8.99163568e-01, -9.47526797e-01, 1.40079988e+00, -1.31198195e+00, -5.79317850e-01 }, { -1.93535347e+00, 4.41340113e-01, -6.15072594e-01, 2.77138576e+00, 1.89864527e-01, -1.03679177e-01, 4.82446999e-01, -1.19502064e+00, 1.96913314e-01, 8.54609374e-01, -4.01452538e-01, 4.39993049e-01, -7.30688218e-01, 7.08946176e-01, 1.55951662e+00, -4.22851797e-01 }, { -2.31145547e-01, -7.88522841e-01, 5.85325399e-01, 9.27255213e-01, -2.15068441e+00, 4.97270116e-01, 1.71529363e-01, 1.81917229e-01, 1.98141358e-01, 5.16010290e-01, 9.39783068e-01, 9.38612924e-01, -5.72864443e-01, 8.96884908e-02, -1.67404912e+00, 6.64533424e-01 }, { -5.20472256e-01, -1.75131267e-01, -7.62177444e-01, 6.88577030e-01, 3.59811597e-01, 2.13543601e-01, 1.09231074e+00, 1.39672810e+00, 5.17394839e-01, -9.73726666e-01, 1.53818186e-01, 1.56359072e-01, 7.28760715e-02, 7.44714753e-01, -1.86272089e-01, 6.05312970e-01 }, { 1.49846112e-01, -2.65348180e-01, 8.97519088e-01, 3.77182703e-01, -1.87973819e+00, -7.70405516e-01, 1.42836369e+00, -4.74286051e-01, -3.33412894e+00, -4.68199013e-01, -2.07636017e+00, 1.37246940e-01, -4.93049351e-01, 3.35000250e-01, 5.13353817e-01, -1.56152862e+00 }, { 2.74998248e-01, -5.17480405e-02, 9.12243594e-01, 7.39398153e-02, 1.31344575e+00, -3.00346627e-01, 7.00000352e-01, -2.81404028e-01, 2.49499188e-01, 1.19027303e+00, -1.44875544e+00, -6.75929467e-01, 2.02209549e-01, -1.89578368e-01, -1.04290320e+00, 3.69351474e-01 }, { 1.19368407e+00, -1.17837678e+00, -1.49830781e+00, 1.42256110e+00, -1.08933411e+00, 2.44878207e-01, 9.07093101e-01, -1.97332273e+00, 1.73779304e+00, -2.39828864e-01, 1.00566084e-01, 1.49434595e+00, -9.86490356e-01, -3.13479264e-01, -1.27315512e+00, 6.97942099e-01 }, { 1.22118525e+00, -3.34107142e-01, -1.56223885e+00, 9.02741645e-01, -4.47001557e-01, 2.67884336e-01, 9.23123041e-01, -5.04340855e-02, 6.70500590e-01, 8.57505381e-01, 5.86396581e-01, 9.84456769e-01, 5.05275959e-01, -9.66549999e-01, 1.05362625e+00, 7.17494547e-01 }, { -3.69039343e-01, 7.35817462e-01, 1.01904954e-01, 2.19808281e-01, 9.36927235e-01, -5.09875036e-01, -3.83233906e-01, 2.54948351e-01, 2.15495290e-01, -2.12498398e+00, 6.47496610e-01, 8.78021654e-01, -9.85584471e-01, -7.32005977e-01, -1.66514637e+00, 9.95422739e-01 } };
                b[4] = new double[,] { { -0.63752699 } };
                W[4] = new double[,] { { 0.79772996, -2.03859153, 1.26544878, -1.79327084, 0.55958908, 2.22296631, -1.52261067, 1.00914203, -1.36505701, 0.75913214, 0.83327227, -0.68874881, -0.92846834, -1.88194455, -1.21256824, -1.08145927 } };
                */





                for (int i = 0; i < n_iter; i++)
                {
                    Console.WriteLine("tryyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy");
                    Console.WriteLine("A_N : " + i * 100 / n_iter + "%");
                    double[][,] Activations = ForwardPropagation(X, W, b);
                    
                     if (i % 10 == 0)
                     {
                         L.Add(ReluLogLoss(Activations[Activations.GetLength(0)-1], Y));
                     }
                    if (i % 100 == 0)
                    {

                        for (int l = 1; l < W.GetLength(0); l++)
                        {
                            string content = "";
                            for (int j = 0; j < W[l].GetLength(0); j++)
                            {
                                for (int c = 0; c < W[l].GetLength(1); c++)
                                {
                                    content = content + W[l][j,c] + '\n';

                                }
                            }
                            if(i!=0)
                            File.WriteAllText(@"D:\c# projects\peceptron\peceptron\humans\Weights\W L"+l+" i"+(n_iter/i)+".txt", content);
                            else
                                File.WriteAllText(@"D:\c# projects\peceptron\peceptron\humans\Weights\W L" + l + " i0.txt", content);
                        }
                    }
                    double[][,] dw = BackwardPropagation(Y, Activations, W);
                    double[][,] db = BackwardPropagation(Y, Activations, W, b);
                    W = UpdateW(dw, W, Learning_rate);
                    b = UpdateB(db, b, Learning_rate);


                }
               
                double[][,] Activ = ForwardPropagation(X, W, b);
                for (int l = 1; l < Activ.GetLength(0); l++)
                {
                    
                    for (int i = 0; i < Activ[l].GetLength(0); i++)
                    {
                        for (int j = 0; j < Activ[l].GetLength(1); j++)
                        {
                            Console.WriteLine("A" + l + "[" + i + " " + j + "] " + Activ[l][i, j]);
                        }
                    }
                }
               




                Console.WriteLine("b:   " + b[1][0, 0]);

                /* File.WriteAllText(Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) + @"\bias.txt", b[0, 0] + "");
                 string currentData = "";
                 for (int i = 0; i < w.GetLength(0); i++)
                 {


                     for (int j = 0; j < w.GetLength(1); j++)
                     {
                         currentData += w[i, j] + ";";

                     }
                     currentData += "\n";

                 }
                 File.WriteAllText(Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) + @"\weights.txt",
                              currentData);
                              
                 for (int l = 0; l < Activations.GetLength(0); l++)
                    {

                        for (int j = 0; j < Activations[l].GetLength(0); j++)
                        {
                            for (int c = 0; c < Activations[l].GetLength(1); c++)
                            {
                                Console.WriteLine("A" + l + " [: " + j + "," + c + "] = " + Activations[l][j, c]);
                            }
                        }
                    }*/
                bool[,] pred = Predict(X, W, b);
                string[] emotions = BackTranslate(Activ, Feelings, W.GetLength(0));
                Console.WriteLine("pred");
                for(int i = 0; i < pred.GetLength(1); i++)
                {
                    Console.WriteLine("[");
                    for(int j=0;j<pred.GetLength(0); j++)
                    {
                        Console.Write(pred[j, i]+" ");
                    }
                    Console.WriteLine("] \n");
                }
                 for(int i=0;i<emotions.Length; i++)
                 {
                     Console.WriteLine(" "+i+" "+emotions[i]);
                 }
                //MathCal.Functions.Accuracy(MathCal.Arr.Ravel(Y), MathCal.Arr.Ravel(pred)));//
                Console.WriteLine("percentage: " +  MathCal.Matrix.Accuracy(Y,pred));
                double[] res = new double[L.Count];
                for (int i = 0; i < res.Length; i++)
                {
                    res[i] = L[i];
                }
                File.WriteAllText(@"D:\c# projects\peceptron\peceptron\humans\LogLoss.txt", MathCal.Matrix.Accuracy(Y, pred).ToString());
                return res;
            }




        }


    }

}
