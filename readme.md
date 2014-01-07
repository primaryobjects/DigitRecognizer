Machine Learning: Digit Recognizer (MNIST data-set)
--------

This is a machine learning project for recognizing hand-written digits of 0-9 as their numeric values. The data comes from the [MNIST data-set] (http://yann.lecun.com/exdb/mnist), as part of a [Kaggle](https://www.kaggle.com/c/digit-recognizer) competition.

The code is developed in C# .NET, using the CsvHelper library for parsing the csv file of images, and Accord .NET for the gaussian SVM. The images are provided as 28x28 gray-scale pixel values (0-255), resulting in 784 pixels per image (ie., 784 columns in each csv row). An additional column is included for the label identifier.

For example (for the digit '7'):
7,0,0,0,0,255,127,0,0...

## Author

Kory Becker
http://www.primaryobjects.com/kory-becker.aspx
