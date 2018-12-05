using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Unearth.Demo.MLCodes.TestConsole.Models
{
    public class FlightCodePrediction
    {
        public float Confidence
        {
            get
            {
                if (Score == null || Score.Length < 1)
                    return float.NaN;
                else
                {
                    // Assume the PredictedLabel is the highest score
                    return Score.Max();
                }
            }
        }

        [ColumnName("PredictedLabel")]
        public string IATACode;

        [ColumnName("Score")]
        public float[] Score; // <-- This is the probability that the predicted value is the right classification

    }
}
