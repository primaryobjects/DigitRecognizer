using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CsvHelper;
using MLParser.Types;

namespace MLParser.Interface
{
    public interface IRowParser
    {
        /// <summary>
        /// Parses a row from a csv file.
        /// </summary>
        /// <param name="reader">CsvReader</param>
        /// <param name="isTest">bool - true if data contains output label, false if data is only input data (ie., test data)</param>
        /// <returns>MLData</returns>
        MLData Parse(CsvReader reader, bool isTest = false);
    }
}
