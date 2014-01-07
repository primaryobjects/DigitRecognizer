using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CsvHelper;
using MLParser.DataLoader.Interface;
using MLParser.Types;

namespace MLParser.DataLoader
{
    public class Parser
    {
        private IRowParser _rowParser = null;

        public Parser(IRowParser rowParser)
        {
            _rowParser = rowParser;
        }

        /// <summary>
        /// Parses a csv file containing inputs and an output label, returning a list of MLData.
        /// </summary>
        /// <param name="path">string</param>
        /// <param name="maxRows">int - max number of rows to read</param>
        /// <param name="isTest">bool - true if data contains output label, false if data is only input data (ie., test data)</param>
        /// <returns>List of MLData</returns>
        public List<MLData> Parse(string path, int maxRows = 0, bool isTest = false)
        {
            List<MLData> dataList = new List<MLData>();

            using (FileStream f = new FileStream(path, FileMode.Open))
            {
                using (StreamReader streamReader = new StreamReader(f, Encoding.GetEncoding(1252)))
                {
                    using (CsvReader csvReader = new CsvReader(streamReader))
                    {
                        csvReader.Configuration.HasHeaderRecord = false;

                        while (csvReader.Read())
                        {
                            dataList.Add(_rowParser.Parse(csvReader, isTest));

                            if (maxRows > 0 && dataList.Count >= maxRows)
                                break;
                        }
                    }
                }
            }

            return dataList;
        }
    }
}
