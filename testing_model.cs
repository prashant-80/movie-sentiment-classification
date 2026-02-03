using Microsoft.ML;
using Microsoft.ML.Data;

//positive review = 0 (false)
//negative review = 1 (true)

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

public class TestingModel
{
    // Uncomment this Main method and comment out Program.Main to test the saved model
    /*
    public static void Main(string[] args)
    {
        MLContext mlContext = new MLContext();

        Console.WriteLine("Loading model...");
        ITransformer loadedModel = mlContext.Model.Load("MovieReviewModel.zip", out var modelInputSchema);

        var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieReview, MovieReviewPrediction>(loadedModel);

        var reviews = new[]
        {
            new MovieReview { Text = "I absolutely loved this movie! The plot was thrilling and the characters were so well developed." },
            new MovieReview { Text = "This was a terrible movie. I wasted two hours of my life watching it." },
            new MovieReview { Text = "An average film with some good moments, but overall it didn't live up to the hype." },
            new MovieReview { Text = "Fantastic performance by the lead actor! A must-watch for everyone." },
            new MovieReview { Text = "I didn't like the storyline, it was too predictable and boring." },
            new MovieReview { Text = "One of the best films I've seen this year. Absolutely brilliant!" },
            new MovieReview { Text = "Disappointing and poorly executed. Not recommended." },
            new MovieReview { Text = "A masterpiece of cinema! Every scene was perfect." },
            new MovieReview { Text = "Worst movie ever. Complete waste of time and money." },
            new MovieReview { Text = "Decent movie, nothing special but entertaining enough." }
        };

        Console.WriteLine("\nTesting model with sample reviews:\n");
        foreach (var review in reviews)
        {
            var prediction = predictionEngine.Predict(review);
            // Note: PredictedLabel false = positive (0), true = negative (1)
            var sentiment = prediction.PredictedLabel ? "Negative" : "Positive";
            var confidence = prediction.PredictedLabel ? prediction.Probability : 1 - prediction.Probability;
            Console.WriteLine($"Text: {review.Text}");
            Console.WriteLine($"Prediction: {sentiment} | Confidence: {confidence:P2}\n");
        }
    }
    */
}
