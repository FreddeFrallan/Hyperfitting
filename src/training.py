from src import validation
import numpy as np
import json
import tqdm

def _save_results(save_data, save_path):
    with open(save_path, 'w') as fp:
        json.dump(save_data, fp)


def train_model(model, tokenizer, dataloader, val_dataloader, val_gen_dataloader, optimizer, device, num_epochs, validation_freq=100, gen_context_len=32, gen_max_length=None, gen_ttr_window_size=96, save_path='hyperfitting_results.json'):
    """
    Train a language model with periodic validation and save results.

    Args:
        model: The language model to be trained (compatible with Hugging Face's transformers).
        tokenizer: Tokenizer for decoding sequences.
        dataloader: DataLoader providing training data (contexts and targets).
        val_dataloader: DataLoader for calculating validation loss and entropy.
        val_gen_dataloader: DataLoader for generation-based validation.
        optimizer: Optimizer for updating model parameters.
        device: Device to run the training on (e.g., 'cuda' or 'cpu').
        num_epochs: Number of epochs to train the model.
        validation_freq: Number of training steps between validations.
        gen_context_len: Number of tokens used as context during generation validation.
        gen_max_length: Maximum length of generated sequences during validation.
        save_path: Path to save training and validation logs as a JSON file.

    Returns:
        None. Logs and intermediate results are saved to `save_path`.
    """
    
    # Initial validation before training starts
    print("Initial Validation:")
    val_loss, val_entropy, val_ttr, val_seqs = validation.validate_model(
        model, tokenizer, val_dataloader, val_gen_dataloader, device, gen_context_len, gen_max_length, gen_ttr_window_size
    )
    
    # Initialize a structure to store results
    save_data = {'logs': [
        {
            'train_loss': None,  # No training yet
            'update_counter': 0,
            'epoch': 0,
            'val_loss': val_loss,  # Initial validation loss
            'val_entropy': val_entropy,  # Initial validation entropy
            'val_ttr': val_ttr,  # Initial validation TTR
            'val_gen_seqs': val_seqs,  # Initial generated sequences
        },
    ]}
    
    # Save initial validation results
    _save_results(save_data, save_path)

    # Begin training
    model.train()
    update_counter = 0  # Tracks total number of updates across epochs
    for epoch in range(num_epochs):
        total_loss = 0  # Accumulate total loss per epoch
        temp_train_loss = []  # Accumulate training loss for validation frequency steps

        # Iterate over training data
        for step, (contexts, targets) in enumerate(tqdm.tqdm(dataloader, desc=f'Epoch-{epoch+1}')):
            contexts, targets = contexts.to(device), targets.to(device)  # Move data to the appropriate device

            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(input_ids=contexts, labels=targets)  # Forward pass with input and labels
            loss = outputs.loss  # Compute loss

            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            update_counter += 1  # Increment update counter

            total_loss += loss.item()  # Add loss for epoch
            temp_train_loss.append(loss.item())  # Store loss for interim validation

            # Perform validation at specified frequency
            if (update_counter + 1) % validation_freq == 0:
                train_loss = np.mean(temp_train_loss)  # Calculate average training loss for recent steps
                print(f"Train loss: {train_loss}:")
                temp_train_loss = []  # Reset temporary loss accumulator

                # Perform validation
                val_loss, val_entropy, val_ttr, val_seqs = validation.validate_model(
                    model, tokenizer, val_dataloader, val_gen_dataloader, device, gen_context_len, gen_max_length
                )
                model.train()  # Return to training mode after validation

                # Log results
                save_data['logs'].append(
                    {
                        'train_loss': train_loss,
                        'update_counter': update_counter,
                        'epoch': epoch + step / len(dataloader),  # Record partial epoch progress
                        'val_loss': val_loss,  # Validation loss
                        'val_entropy': val_entropy,  # Validation entropy
                        'val_ttr': val_ttr,  # Validation TTR
                        'val_gen_seqs': val_seqs,  # Validation generated sequences
                    }
                )
                # Save results to the specified file
                _save_results(save_data, save_path)