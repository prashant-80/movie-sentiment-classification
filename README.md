# Movie Review Sentiment Classification

A C# machine learning project using ML.NET to classify movie reviews as positive or negative.

### Model Performance
- **Accuracy**: 86.61%
- **AUC**: 94.23%
- **F1 Score**: 86.51%
- **Model File**: MovieReviewModel.zip (27MB)

## Dataset
- **Source**: [archive/train.csv (25,000 reviews)](https://www.kaggle.com/datasets/jcblaise/imdb-sentiments/data)
- **Labels**: 
  - 0 (false) = Positive review
  - 1 (true) = Negative review

## How to Use

### 1. Train the Model (Default)
The training code is in [Program.cs](Program.cs). To train the model:

```bash
dotnet run
```

This will:
- Load the training data (25,000 reviews)
- Split into 80% training / 20% testing  
- Train an LBFGS Logistic Regression model
- Evaluate and show metrics (Accuracy: 86.61%, AUC: 94.23%, F1: 86.51%)
- Save the model to `MovieReviewModel.zip`
- Test with 5 sample reviews

**Note**: Training takes about 1-2 minutes and uses the full dataset for best accuracy.

### 2. Test the Saved Model Only
To test the saved model without retraining:

1. Open [Program.cs](Program.cs) and comment out the entire `Main` method (add `/*` before and `*/` after)
2. Open [testing_model.cs](testing_model.cs) and uncomment the `Main` method (remove `/*` and `*/`)
3. Run:
   ```bash
   dotnet run
   ```

This will load the saved model and test it with 10 different sample reviews instantly.

## Example Predictions

```
Text: I absolutely loved this movie! The plot was thrilling!
Prediction: Positive | Confidence: 97.73%

Text: Terrible movie. Waste of time and money.
Prediction: Negative | Confidence: 99.68%

Text: Boring and predictable. I didn't like the storyline.
Prediction: Negative | Confidence: 94.51%
```

## Model Details

### Algorithm
- **LBFGS Logistic Regression** with L1 regularization
- Better accuracy than SDCA for text classification

### Text Processing
- Uses ML.NET's `FeaturizeText` transform
- Automatically handles tokenization, normalization, n-grams, and TF-IDF

### Data Handling
- CSV with multi-line quoted text fields
- `allowQuoting: true` enables proper parsing
- Full dataset training for best accuracy

## Files
- `Program.cs` - Training code
- `testing_model.cs` - Testing code for saved model
- `MovieReviewModel.zip` - Trained model (generated after training)
- `archive/train.csv` - Training data

## Requirements
- .NET 10.0
- ML.NET 5.0.0

## Troubleshooting

### Exit Code 134
This error occurred when:
- Labels were wrong type (float instead of bool)
- Sample data had only one class
- Solution: Use full dataset with proper boolean labels

### Low Accuracy
- Initial SDCA model: 58% accuracy
- Improved LBFGS model: 86.61% accuracy
- Key: Use better algorithm and more training data
