using CsvHelper;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Conversions;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.IO;
using Unearth.Demo.MLCodes.TestConsole.Models;

namespace Unearth.Demo.MLCodes.TestConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            var txtFeatModel = TrainModel();
            var charGramModel = TrainModel(useCharGrams: true);

            Console.WriteLine("Test Text Featurization Model");
            var txtFeatAccuracy = TestModel(txtFeatModel);

            Console.WriteLine("Test char-gram Model");
            var charGramAccuracy = TestModel(charGramModel);

            // Typical result: Text Featurizing Model: 0.9323, CharGram Model: 0.9846
            // BUT. If you train with 'ManyFlightCodes' this becomes more like 97.8 -> 98.5
            Console.WriteLine($"Text Featurizing Model: {txtFeatAccuracy}, CharGram Model: {charGramAccuracy}");
            Console.WriteLine();

            Console.WriteLine("Finished");
            Console.ReadLine();

        }

        private static ITransformer TrainModel(bool useCharGrams = false)
        {
            var mlContext = new MLContext(seed: 0);

            // Create a view of the training data
            var dataPath = @"TrainingData\FlightCodes.csv";
            TextLoader textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                    {
                        new TextLoader.Column("FlightCode", DataKind.Text, 0),
                        new TextLoader.Column("IATACode", DataKind.Text, 1),
                    }
            });
            var trainingDataView = textLoader.Read(dataPath);

            EstimatorChain<ITransformer> dataProcessPipeline = null;
            if (useCharGrams)
            {
                // Build a pipeline that makes features that are bags of tri-char-grams
                // Adding FeaturizeText gives no improvement to accuracy
                // Adding quad-grams makes it worse...
                dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("IATACode", "Label")
                    // Turn the string into an array of characters
                    .Append(mlContext.Transforms.Text.TokenizeCharacters("FlightCode", "FlightCodeChars"))
                    // Get a set of floats representing all three-character sequences this will catch 'B73(7)' and '733' etc.
                    .Append(new NgramExtractingEstimator(mlContext, "FlightCodeChars", "BagOfTriChar",
                                ngramLength: 3, weighting: NgramExtractingEstimator.WeightingCriteria.TfIdf))
                    .Append(mlContext.Transforms.Concatenate("Features", "BagOfTriChar"));
                Console.WriteLine("Training the CharGram Model");
            }
            else
            {
                // Just use text featurization
                dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("IATACode", "Label")
                    .Append(mlContext.Transforms.Text.FeaturizeText("FlightCode", "FlightCodeFeaturized"))
                    .Append(mlContext.Transforms.Concatenate("Features", "FlightCodeFeaturized"));

                Console.WriteLine("Training the TextFeaturized Model");
            }

            // Create the training algorithm/trainer
            var trainer = mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(DefaultColumnNames.Label, DefaultColumnNames.Features);

            //Set the trainer/algorithm and map label to value (original readable state)
            var trainingPipeline = dataProcessPipeline.Append(trainer)
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Do the actual training
            var watch = System.Diagnostics.Stopwatch.StartNew();
            var trainedModel = trainingPipeline.Fit(trainingDataView);
            watch.Stop();
            long elapsedMs = watch.ElapsedMilliseconds;
            Console.WriteLine($"Training took {elapsedMs/1000f} secs");
            Console.WriteLine();

            return trainedModel;
        }

        private static float TestModel(ITransformer model)
        {
            // Create an ML.NET environment
            var mlContext = new MLContext(seed: 0);

            // Make a predictor
            var flightCodePredictor = model.MakePredictionFunction<FlightCodeFeatures, FlightCodePrediction>(mlContext);

            // Try out some (different) test data
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Predicting IATA Aircraft Codes");
            Console.ForegroundColor = defaultColor;

            var correct = 0;
            var incorrect = 0;

            using (TextReader reader = new StreamReader(@"TrainingData\MoreFlightCodes.csv"))
            {
                var csvReader = new CsvReader(reader);
                var records = csvReader.GetRecords<FlightCodeFeatures>();
                foreach (var rec in records)
                {
                    var prediction = flightCodePredictor.Predict(rec);
                    if (prediction.IATACode == rec.IATACode)
                    {
                        correct++;
                        if (correct % 300 == 0)
                        {
                            Console.ForegroundColor = ConsoleColor.Green;
                            Console.WriteLine($"FlightCode: {rec.FlightCode}, Aircraft Code: {rec.IATACode} - Predicted Aircraft Code: {prediction.IATACode}, Confidence: {prediction.Confidence}");
                        }
                    }
                    else
                    {
                        incorrect++;
                        if (incorrect % 30 == 0)
                        {
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.WriteLine($"FlightCode: {rec.FlightCode}, Aircraft Code: {rec.IATACode} - Predicted Aircraft Code: {prediction.IATACode}, Confidence: {prediction.Confidence}");
                        }
                    }
                }
            }
            var accuracy = (float)correct / (correct + incorrect);
            Console.ForegroundColor = defaultColor;
            Console.WriteLine($"Accuracy: {accuracy}");
            Console.WriteLine();
            return accuracy;
        }
    }
}
