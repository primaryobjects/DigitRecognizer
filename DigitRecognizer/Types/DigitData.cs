using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitRecognizer.Types
{
    public class DigitData
    {
        public List<double> Pixels { get; set; }
        public int Label { get; set; }

        public DigitData()
        {
            Pixels = new List<double>();
        }
    }
}
