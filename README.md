# TrOCR Handwritten Word Recognition

This project demonstrates how to fine-tune and use a TrOCR model for handwritten word recognition. It includes two main components:

1.  **Fine-tuning:** The `Fine-tune-TrOCR-WORD.ipynb` notebook shows how to fine-tune a pre-trained TrOCR model on a custom dataset of handwritten words.
2.  **Inference:** The `Inference_Trocr (1).ipynb` notebook shows how to use the fine-tuned model to perform inference on new handwritten word images.

## Project Structure

*   `Fine-tune-TrOCR-WORD.ipynb`: Jupyter Notebook for fine-tuning the TrOCR model.
*   `Inference_Trocr (1).ipynb`: Jupyter Notebook for performing inference with the fine-tuned model.
*   `DATA _TrOCR/`: Directory containing the training and testing data.
*   `words/`: Directory containing the images of handwritten words.
*   `Trained_model/`: Directory where the fine-tuned model checkpoints are saved.
*   `test/`: Directory containing test images.

## Fine-tuning the TrOCR Model

The `Fine-tune-TrOCR-WORD.ipynb` notebook performs the following steps:

1.  **Imports Libraries:** Imports necessary libraries such as `datasets`, `pandas`, `numpy`, `PIL`, `transformers`, `torch`, and `sklearn`.
2.  **Loads Data:** Loads the training data from a CSV file and splits it into training and testing sets.
3.  **Creates Custom Dataset:** Defines a custom dataset class `CustomDataset` to load and preprocess the image and text data.
4.  **Initializes Processor and Model:** Initializes the TrOCR processor and model from pre-trained checkpoints.
5.  **Sets Training Parameters:** Sets the training parameters such as batch size, learning rate, and number of epochs.
6.  **Defines Metrics:** Defines the Character Error Rate (CER) metric to evaluate the model's performance.
7.  **Trains the Model:** Trains the model using the training data and evaluates it on the testing data.
8.  **Saves Checkpoints:** Saves the model checkpoints during training.

## Performing Inference

The `Inference_Trocr.ipynb` notebook performs the following steps:

1.  **Imports Libraries:** Imports necessary libraries such as `datasets`, `pandas`, `numpy`, `PIL`, `transformers`, `torch`, and `sklearn`.
2.  **Loads Trained Model:** Loads the fine-tuned TrOCR model from the saved checkpoint.
3.  **Initializes Processor:** Initializes the TrOCR processor.
4.  **Defines Preprocessing Function:** Defines a function to preprocess the input images.
5.  **Performs Inference:** Performs inference on new handwritten word images.
6.  **Calculates CER:** Calculates the Character Error Rate (CER) to evaluate the model's performance.
7.  **Displays Results:** Displays the predicted text and CER for each image.
8.  **Saves Results:** Saves the results to a CSV file.

## How to Use

1.  **Set up the environment:** Install the required libraries using `pip install -r d/requirements.txt`.
2.  **Prepare the data:** Place the training and testing data in the `DATA _TrOCR/` directory, the images in the `words/` and `test/` directories, and ensure the paths in the notebooks are correct.
3.  **Fine-tune the model:** Run the `Fine-tune-TrOCR-WORD.ipynb` notebook to fine-tune the TrOCR model.
4.  **Perform inference:** Run the `Inference_Trocr.ipynb` notebook to perform inference on new images.

## Requirements

*   Python 3.6+
*   PyTorch
*   Transformers
*   Datasets
*   Pandas
*   Numpy
*   PIL
*   Scikit-learn

## Notes

*   The paths to the data and model checkpoints may need to be adjusted based on your local setup.
*   The training parameters can be adjusted to improve the model's performance.
*   The `Inference_Trocr.ipynb` notebook includes a function to display the results, which can be used to visualize the model's predictions.

This project provides a starting point for building a handwritten word recognition system using TrOCR.
