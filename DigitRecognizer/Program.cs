using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using CsvHelper;
using DigitRecognizer.Types;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitRecognizer
{
    class Program
    {
        #region App.Config Values

        private static int _pixelCount = Int32.Parse(ConfigurationManager.AppSettings["Width"]) * Int32.Parse(ConfigurationManager.AppSettings["Height"]);
        private static int _classCount = Int32.Parse(ConfigurationManager.AppSettings["ClassCount"]);
        private static int _trainCount = Int32.Parse(ConfigurationManager.AppSettings["TrainCount"]);
        private static int _sigma = Int32.Parse(ConfigurationManager.AppSettings["Sigma"]);
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

            // Convert the DigitData (pixels and labels) to arrays for inputs and outputs.
            ReadData(path, count, out inputs, out outputs);

            if (machine == null)
            {
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

            // Convert the DigitData (pixels and labels) to arrays for inputs and outputs.
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
        private static void ReadData(string path, int count, out double[][] inputs, out int[] outputs, bool isTest = false)
        {
            // Read the training data CSV file and get a resulting array of doubles and output labels.
            List<DigitData> rows = Utility.ShowProgressFor<List<DigitData>>(() => ReadData(path, count, isTest), "Reading data");
            Console.WriteLine(rows.Count + " rows processed.");

            // Convert the rows into arrays for processing.
            inputs = rows.Select(t => t.Pixels.ToArray()).ToArray();
            outputs = rows.Select(t => t.Label).ToArray();
        }

        /// <summary>
        /// Parses a csv file containing the MNIST data set, returning a list of DigitData.
        /// The csv file's first column is the digit label, and the remaining 784 columns are pixels data for a 28x28 gray-scale image.
        /// </summary>
        /// <param name="path">string</param>
        /// <param name="maxRows">int - max number of rows to read</param>
        /// <param name="isTest">bool - true if data contains output label, false if data is only pixels (ie., test data)</param>
        /// <returns>List of DigitData</returns>
        private static List<DigitData> ReadData(string path, int maxRows = 0, bool isTest = false)
        {
            List<DigitData> digits = new List<DigitData>();

            using (FileStream f = new FileStream(path, FileMode.Open))
            {
                using (StreamReader streamReader = new StreamReader(f, Encoding.GetEncoding(1252)))
                {
                    using (CsvReader csvReader = new CsvReader(streamReader))
                    {
                        csvReader.Configuration.HasHeaderRecord = false;

                        while (csvReader.Read())
                        {
                            // Determine the image size, as configured in the app.config.
                            int column = 0;
                            DigitData digit = new DigitData();

                            if (!isTest)
                            {
                                // Read the digit label.
                                digit.Label = Int32.Parse(csvReader[column++]);
                            }

                            // Read the pixels (+1 since the first column is the output label).
                            while (digit.Pixels.Count < _pixelCount)
                            {
                                // Read the value.
                                double value = Double.Parse(csvReader[column++]);

                                // Normalize the value: X = (X - avg) / max - min => X = (X - 127) / 255. Alternate method X = (X - min) / (max - min) => X = X / 255. http://en.wikipedia.org/wiki/Feature_scaling
                                value = (value - 127d) / 255d;

                                // Store the value in our pixels list.
                                digit.Pixels.Add(value);
                            }

                            digits.Add(digit);

                            if (maxRows > 0 && digits.Count >= maxRows)
                                break;
                        }
                    }
                }
            }

            return digits;
        }
    }
}
