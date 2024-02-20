import argparse
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()  # for a nicer styling of the confusion matrix
import numpy as np

def evaluate_predictions(csv_file, target_col, prediction_col, model_name):

    # Load the CSV file
    data = pd.read_csv(csv_file)
    
    # Extract target labels and predictions
    y_true = data[target_col]
    y_pred = data[prediction_col]
    
    # Compute metrics
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Print the metrics
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    
    output_filename = f'confusion_matrix_{model_name}.png'

    # Plot and save the confusion matrix
    classes = sorted(set(y_true) | set(y_pred))  # Get the list of unique classes
    plot_confusion_matrix(y_true, y_pred, classes, model_name, output_filename)



def plot_confusion_matrix(y_true, y_pred, classes, model_name, output_filename):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 7))    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='Purples')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Normalized Confusion Matrix - {model_name}')
    plt.tight_layout()
    
    # Save the confusion matrix as an image
    plt.savefig(output_filename)
    plt.close()  # Close the plot to prevent it from displaying in the script output

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Evaluate prediction metrics from a CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the input CSV file")
    parser.add_argument("--target_col", default="Prompted_Emotion", type=str, help="Name of the target column")
    parser.add_argument("--prediction_col", default="Roberta_Prediction", type=str, help="Name of the prediction column")
    parser.add_argument("--model_name", default="model", type=str, help="Name of the prediction model")

    # Parse arguments
    args = parser.parse_args()
    
    # Call the evaluation function
    evaluate_predictions(args.csv_file, args.target_col, args.prediction_col, args.model_name)
