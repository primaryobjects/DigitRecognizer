using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CsvHelper;
using MLParser.Interface;
using MLParser.Types;

namespace MLParser.Parsers
{
    public class FrontLabelParser : IRowParser
    {
        /// <summary>
        /// Parses a row from the csv file, populating the Data and Label. The label is expected to be the first column in the csv.
        /// </summary>
        /// <param name="reader">CsvReader</param>
        /// <param name="isTest">bool - true if data contains output label, false if data is only input data (ie., test data)</param>
        /// <returns>MLData</returns>
        public MLData Parse(CsvReader reader, bool isTest)
        {
            int column = 0;
            int fieldCount = reader.Parser.FieldCount;
            MLData row = new MLData();

            if (!isTest)
            {
                // Read the numeric label.
                row.Label = Int32.Parse(reader[column++]);

                // -1 since the first column was the label.
                fieldCount--;
            }

            // Read the data.
            while (row.Data.Count < fieldCount)
            {
                // Read the value.
                double value = Double.Parse(reader[column++]);

                // Normalize the value (0 - 1): X = (X - min) / (max - min) => X = X / 255. Alternate method (-0.5 - 0.5): X = (X - avg) / max - min => X = (X - 127) / 255. http://en.wikipedia.org/wiki/Feature_scaling
                value = value / 255d;

                // Store the value in our data list.
                row.Data.Add(value);
            }

            return row;
        }
    }
}
