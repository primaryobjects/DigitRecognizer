using System;
using System.Collections.Generic;
using System.Configuration;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using MLParser;
using MLParser.Parsers;
using MLParser.Types;

namespace DigitRecognizer
{
    class Program
    {
        #region App.Config Values

        private static int _pixelCount = Int32.Parse(ConfigurationManager.AppSettings["Width"]) * Int32.Parse(ConfigurationManager.AppSettings["Height"]);
        private static int _classCount = Int32.Parse(ConfigurationManager.AppSettings["ClassCount"]);
        private static int _trainCount = Int32.Parse(ConfigurationManager.AppSettings["TrainCount"]);
        private static double _sigma = Double.Parse(ConfigurationManager.AppSettings["Sigma"]);
        private static string _trainPath = ConfigurationManager.AppSettings["TrainPath"];
        private static string _cvPath = ConfigurationManager.AppSettings["CvPath"];
        private static string _testPath = ConfigurationManager.AppSettings["TestPath"];

        #endregion

        static void Main(string[] args)
        {
            Console.WriteLine("-= Training =-");
            var machine = RunSvm(_trainPath, _trainCount);

            Console.WriteLine("-= Cross Validation =-");
            RunSvm(_cvPath, _trainCount, machine);

            Console.WriteLine("-= Test =-");
            TestSvm(_testPath, "../../../data/output.txt", _trainCount, machine);
        }

        /// <summary>
        /// Core machine learning method for parsing csv data, training the svm, and calculating the accuracy.
        /// </summary>
        /// <param name="path">string - path to csv file (training, csv, test).</param>
        /// <param name="count">int - max number of rows to process. This is useful for preparing learning curves, by using gradually increasing values. Use Int32.MaxValue to read all rows.</param>
        /// <param name="machine">MulticlassSupportVectorMachine - Leave null for initial training.</param>
        /// <returns>MulticlassSupportVectorMachine</returns>
        private static MulticlassSupportVectorMachine RunSvm(string path, int count, MulticlassSupportVectorMachine machine = null)
        {
            double[][] inputs;
            int[] outputs;

            // Parse the csv file to get inputs and outputs.
            ReadData(path, count, out inputs, out outputs);

            if (machine == null)
            {
                // Training.
                MulticlassSupportVectorLearning teacher = null;

                // Create the svm.
                machine = new MulticlassSupportVectorMachine(_pixelCount, new Gaussian(_sigma), _classCount);
                teacher = new MulticlassSupportVectorLearning(machine, inputs, outputs);
                teacher.Algorithm = (svm, classInputs, classOutputs, i, j) => new SequentialMinimalOptimization(svm, classInputs, classOutputs) { CacheSize = 0 };

                // Train the svm.
                Utility.ShowProgressFor(() => teacher.Run(), "Training");
            }

            // Calculate accuracy.
            double accuracy = Utility.ShowProgressFor<double>(() => Accuracy.CalculateAccuracy(machine, inputs, outputs), "Calculating Accuracy");
            Console.WriteLine("Accuracy: " + Math.Round(accuracy * 100, 2) + "%");

            return machine;
        }

        /// <summary>
        /// Runs the svm on test data (with no labels).
        /// </summary>
        /// <param name="path">string - path to csv file (training, csv, test).</param>
        /// <param name="outputPath">string - path to output results file.</param>
        /// <param name="count">int - max number of rows to process. This is useful for preparing learning curves, by using gradually increasing values. Use Int32.MaxValue to read all rows.</param>
        /// <param name="machine">MulticlassSupportVectorMachine - Leave null for initial training.</param>
        private static void TestSvm(string path, string outputPath, int count, MulticlassSupportVectorMachine machine)
        {
            double[][] inputs;
            int[] outputs;

            // Parse the csv file to get inputs and outputs.
            ReadData(path, count, out inputs, out outputs, true);

            // Save output.
            Utility.ShowProgressFor(() => Accuracy.SaveOutput(machine, inputs, outputPath), "Saving Output");
        }

        /// <summary>
        /// Parses a csv file containing the MNIST data set, returning arrays for inputs and outputs in the format required by the svm.
        /// </summary>
        /// <param name="path">string</param>
        /// <param name="count">int - max number of rows to read</param>
        /// <param name="inputs">output variable for double[][] values (inputs)</param>
        /// <param name="outputs">output variable for int[] values (labels)</param>
        /// <param name="isTest">bool - true if data contains output label, false if data is only pixels (ie., test data)</param>
        /// <returns>int - number of rows read</returns>
        private static int ReadData(string path, int count, out double[][] inputs, out int[] outputs, bool isTest = false)
        {
            Parser parser = new Parser(new FrontLabelParser());

            // Read the training data CSV file and get a resulting array of doubles and output labels.
            List<MLData> rows = Utility.ShowProgressFor<List<MLData>>(() => parser.Parse(path, count, isTest), "Reading data");

            // Convert the rows into arrays for processing.
            inputs = rows.Select(t => t.Data.ToArray()).ToArray();
            outputs = rows.Select(t => t.Label).ToArray();

            Console.WriteLine(rows.Count + " rows processed.");

            return rows.Count;
        }
    }
}
