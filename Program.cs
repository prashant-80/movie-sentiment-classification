using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;
using Microsoft.ML.Trainers;

namespace Classification
{
    public class MovieReview
    {
        [LoadColumn(0)]
        public string Text { get; set; } = string.Empty;

        [LoadColumn(1)]
        public bool Label { get; set; }
    }

    public class MovieReviewPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }

    public class Program
    {
        
        public static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            string dataPath = "./archive/train.csv";

            Console.WriteLine("Loading data...");
            IDataView dataView = mlContext.Data.LoadFromTextFile<MovieReview>(
                dataPath,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true,
                allowSparse: false);

            Console.WriteLine("Splitting data...");
            var trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 1);

            Console.WriteLine("Building training pipeline...");

            var trainingPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(MovieReview.Text))
                .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
                    labelColumnName: "Label",
                    featureColumnName: "Features",
                    l1Regularization: 0.03f));

            Console.WriteLine("Training model...");
            var model = trainingPipeline.Fit(trainTestSplit.TrainSet);

            Console.WriteLine("\nEvaluating model...");
            var predictions = model.Transform(trainTestSplit.TestSet);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");

            Console.WriteLine("\nSaving model...");
            mlContext.Model.Save(model, trainTestSplit.TrainSet.Schema, "MovieReviewModel.zip");
            Console.WriteLine("Model saved successfully!\n");

            Console.WriteLine("Testing model with sample reviews:\n");
            var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieReview, MovieReviewPrediction>(model);
            
            var reviews = new[]
            {
                new MovieReview { Text = "This movie was fantastic! I really loved it. Best film ever!" },
                new MovieReview { Text = "Terrible movie. Waste of time and money. Very disappointed." },
                new MovieReview { Text = "An average film with some good moments but overall mediocre." },
                new MovieReview { Text = "Fantastic performance by the lead actor! A must-watch for everyone." },
                new MovieReview { Text = "Boring and predictable. I didn't like the storyline at all." }
            };

            foreach (var review in reviews)
            {
                var prediction = predictionEngine.Predict(review);
                //false = positive (0), true = negative (1) based on CSV labels
                var sentiment = prediction.PredictedLabel ? "Negative" : "Positive";
                var confidence = prediction.PredictedLabel ? prediction.Probability : 1 - prediction.Probability;
                Console.WriteLine($"Text: {review.Text}");
                Console.WriteLine($"Prediction: {sentiment} | Confidence: {confidence:P2}\n");
            }
        }
    }
}
