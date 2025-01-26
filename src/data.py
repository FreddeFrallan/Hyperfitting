from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import transformers
import numpy as np
import torch
import tqdm

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        """
        Initialize the dataset with input sequences.
        Args:
            sequences (list of lists): A list of tokenized sequences.
        """
        self.samples = sequences

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return the context and the next-token targets for a sequence.
        """
        context = self.samples[idx]
        target = self.samples[idx] 
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)
    


def tokenize_hyperfitting_dataset(tokenizer_path, num_train_samples=2000, num_val_samples=128, seq_len=256, dataset_path='wikitext', dataset_config='wikitext-103-raw-v1', split='train'):
    """
    Args:
        tokenizer_path (str): Path to the tokenizer.
        num_train_samples (int): Maximum number of training samples to use.
        num_val_samples (int): Maximum number of validation samples to use.
        seq_len (int): Sequence length for tokenization.
        dataset_path (str): Dataset name or path.
        split (str): Split of the dataset to use (e.g., 'train').

    Returns:
        train_sequences (list): List of training sequences, all of the same length.
        val_sequences (list): List of validation sequences, all of the same length.
    """
    # Load the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    # Load the dataset
    dataset = load_dataset(dataset_path, dataset_config, split=split)

    # Tokenize and filter samples
    def tokenize_and_filter(example):
        tokens = tokenizer(example["text"], truncation=True, max_length=seq_len, padding=False)
        return tokens["input_ids"] if len(tokens["input_ids"]) == seq_len else None

    print("Tokenizing and filtering dataset...")
    filtered_data = []
    total_num_samples = num_train_samples + num_val_samples
    for sample in tqdm.tqdm(dataset, desc='Tokenizing data'):
        tokenized = tokenize_and_filter(sample)
        if tokenized is not None:
            filtered_data.append(tokenized)
            if len(filtered_data) >= total_num_samples:
                break

    print(f"Number of samples after filtering: {len(filtered_data)}")

    # Split the data into train and validation sets
    train_sequences = filtered_data[:num_train_samples]
    val_sequences = filtered_data[num_train_samples:num_train_samples + num_val_samples]

    return train_sequences, val_sequences


def create_dataset(tokenizer_path, num_train_samples=2000, num_val_samples=128, num_val_gen_samples=32, seq_len=256, dataset_path='wikitext', dataset_config='wikitext-103-raw-v1', split='train', batch_size=8, val_batch_size=8):
    """
    Create train and validation DataLoaders from tokenized sequences.

    Args:
        tokenizer_path (str): Path to the tokenizer.
        num_train_samples (int): Maximum number of training samples to use.
        num_val_samples (int): Maximum number of validation samples to use.
        seq_len (int): Sequence length for tokenization.
        dataset_path (str): Dataset name or path.
        split (str): Split of the dataset to use (e.g., 'train').
        batch_size (int): Batch size for the DataLoader.

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
    """
    train_sequences, val_sequences = tokenize_hyperfitting_dataset(tokenizer_path, num_train_samples, num_val_samples, seq_len, dataset_path, dataset_config, split)

    train_dataset = SequenceDataset(train_sequences)
    val_dataset = SequenceDataset(val_sequences)
    val_gen_dataset = SequenceDataset(val_sequences[:num_val_gen_samples])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=False)
    val_gen_loader = DataLoader(val_gen_dataset, batch_size=val_batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, val_gen_loader