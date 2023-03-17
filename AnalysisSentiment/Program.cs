using System;
using System.IO;
using AnalysisSentiment;
using AnalysisSentiment.DataStructures;
using Microsoft.ML;
using Microsoft.ML.Data;

var baseDataSetsRelativePath = @"../../../../Data";
var dataRelativePath = $"{baseDataSetsRelativePath}/wikiDetoxAnnotated40kRows.tsv";
var dataPath = GetAbsolutePath(dataRelativePath);
var baseModelsRelativePath = @"../../../../MLModels";
var modelRelativePath = $"{baseModelsRelativePath}/SentimentModel.zip";
var modelPath = GetAbsolutePath(modelRelativePath);

var trainedData = Train();
while (true)
{
    Console.WriteLine($"\nEnter a txt to predict :");
    var txt = Console.ReadLine();
    Predict(trainedData.mlContext, trainedData.trainedModel, txt);
}

string GetAbsolutePath(string relativePath)
{
    var dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    var assemblyFolderPath = dataRoot.Directory!.FullName;

    var fullPath = Path.Combine(assemblyFolderPath, relativePath);

    return fullPath;
}

(MLContext mlContext, ITransformer trainedModel) Train()
{
    var mlContext = new MLContext(seed: 1);

    var dataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(dataPath, hasHeader: true);

    var trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
    var trainingData = trainTestSplit.TrainSet;
    var testData = trainTestSplit.TestSet;

    var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentIssue.Text));

    var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
    var trainingPipeline = dataProcessPipeline.Append(trainer);

    ITransformer trainedModel = trainingPipeline.Fit(trainingData);

    var predictions = trainedModel.Transform(testData);
    var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");

    PrintBinaryClassificationMetrics(trainer.ToString(), metrics);

    mlContext.Model.Save(trainedModel, trainingData.Schema, modelPath);

    Console.WriteLine("The model is saved to {0}", modelPath);
    return (mlContext, trainedModel);
}

void Predict(MLContext mlContext, ITransformer trainedModel, string txt)
{
    var sampleStatement = new SentimentIssue { Text = txt };

    var engine = mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(trainedModel);

    var prediction = engine.Predict(sampleStatement);

    var defaultColor = Console.ForegroundColor;

    if (prediction.Prediction)
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"\n Toxic sentiment. \n Probability of being toxic: {prediction.Probability} \n");
    }
    else
    {
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"\n Non Toxic sentiment. \n Probability of being toxic: {prediction.Probability} \n");
    }
    Console.ForegroundColor = defaultColor;
}

void PrintBinaryClassificationMetrics(string name, CalibratedBinaryClassificationMetrics metrics)
{
    Console.WriteLine($"************************************************************");
    Console.WriteLine($"*       Metrics for {name} binary classification model      ");
    Console.WriteLine($"*-----------------------------------------------------------");
    Console.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
    Console.WriteLine($"*       Area Under Curve:      {metrics.AreaUnderRocCurve:P2}");
    Console.WriteLine($"*       Area under Precision recall Curve:  {metrics.AreaUnderPrecisionRecallCurve:P2}");
    Console.WriteLine($"*       F1Score:  {metrics.F1Score:P2}");
    Console.WriteLine($"*       LogLoss:  {metrics.LogLoss:#.##}");
    Console.WriteLine($"*       LogLossReduction:  {metrics.LogLossReduction:#.##}");
    Console.WriteLine($"*       PositivePrecision:  {metrics.PositivePrecision:#.##}");
    Console.WriteLine($"*       PositiveRecall:  {metrics.PositiveRecall:#.##}");
    Console.WriteLine($"*       NegativePrecision:  {metrics.NegativePrecision:#.##}");
    Console.WriteLine($"*       NegativeRecall:  {metrics.NegativeRecall:P2}");
    Console.WriteLine($"************************************************************");
}