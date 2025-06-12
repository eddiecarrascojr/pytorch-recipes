import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def main():
    """
    Main function to run the sentiment analysis model training and evaluation.
    """
    # 1. Load the IMDB dataset
    # --------------------------
    # The `load_dataset` function from the `datasets` library makes it easy to
    # download and cache datasets from the Hugging Face Hub.
    print("Loading IMDB dataset...")
    imdb = load_dataset("imdb")

    # For faster development and testing, you can use a smaller subset of the data.
    # We'll create a smaller training and test set by shuffling and selecting a subset.
    small_train_dataset = imdb["train"].shuffle(seed=42).select(range(1000))
    small_test_dataset = imdb["test"].shuffle(seed=42).select(range(1000))

    print(f"Using {len(small_train_dataset)} training examples and {len(small_test_dataset)} testing examples.")


    # 2. Load a pre-trained tokenizer
    # ---------------------------------
    # Every Transformer model has a corresponding tokenizer that converts text
    # into a format the model can understand (input IDs, attention mask, etc.).
    # We'll use the tokenizer for 'distilbert-base-uncased', a smaller and faster
    # version of BERT.
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


    # 3. Tokenize the dataset
    # -------------------------
    # We'll create a function to tokenize the text in our dataset.
    # `truncation=True` ensures that long reviews are cut to the model's max length.
    # `padding=True` adds padding to shorter reviews to make all inputs the same length.
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    print("Tokenizing datasets...")
    tokenized_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = small_test_dataset.map(tokenize_function, batched=True)


    # 4. Load a pre-trained model
    # -----------------------------
    # We'll load 'distilbert-base-uncased' with a sequence classification head.
    # `num_labels=2` specifies that this is a binary classification problem (positive/negative).
    print("Loading pre-trained model...")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)


    # 5. Define training arguments
    # ----------------------------
    # `TrainingArguments` is a class that contains all the hyperparameters for training.
    # This includes settings like learning rate, number of epochs, batch size, etc.
    training_args = TrainingArguments(
        output_dir="./results",          # Directory to save the model and results
        num_train_epochs=3,              # Total number of training epochs
        per_device_train_batch_size=16,  # Batch size per device during training
        per_device_eval_batch_size=64,   # Batch size for evaluation
        warmup_steps=500,                # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # Strength of weight decay
        logging_dir='./logs',            # Directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",     # Evaluate at the end of each epoch
    )


    # 6. Define evaluation metrics
    # ----------------------------
    # We need a function to compute metrics during evaluation.
    # This function will be called by the Trainer at each evaluation step.
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # 7. Create a Trainer instance
    # ----------------------------
    # The `Trainer` class provides a high-level API for training and evaluating
    # Hugging Face Transformers models.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics,
    )


    # 8. Train the model
    # ------------------
    # Calling `train()` on the Trainer instance will start the fine-tuning process.
    print("Starting model training...")
    trainer.train()
    print("Training finished.")


    # 9. Evaluate the model
    # ---------------------
    # After training, you can evaluate your model on the test set.
    print("Evaluating the model on the test set...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")


    # 10. Make predictions on new text
    # --------------------------------
    # You can now use your fine-tuned model to predict the sentiment of new sentences.
    def predict_sentiment(text):
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Move tensors to the same device as the model
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            logits = model(**inputs).logits

        # Get the predicted class (0 for negative, 1 for positive)
        predicted_class_id = torch.argmax(logits, dim=1).item()
        return "Positive" if predicted_class_id == 1 else "Negative"

    # Example predictions
    review1 = "This movie was fantastic! I really enjoyed the acting and the plot."
    review2 = "It was a complete waste of time. The story was boring and predictable."

    print(f"Review: '{review1}'")
    print(f"Predicted sentiment: {predict_sentiment(review1)}")

    print(f"Review: '{review2}'")
    print(f"Predicted sentiment: {predict_sentiment(review2)}")


if __name__ == "__main__":
    main()