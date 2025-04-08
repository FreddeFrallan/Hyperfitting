from src.data import create_dataset
from src.training import train_model
from torch.optim import AdamW
import argparse

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    ImageGPTImageProcessor, ImageGPTForCausalImageModeling
)
import os
def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        args: Parsed arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(description="Train a language model with hyperfitting validation.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--tokenizer_path", type=str, required=False, help="Path to the tokenizer.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on (e.g., 'cuda' or 'cpu').")

    parser.add_argument("--dataset_path", type=str, default="wikitext", help="Dataset name or path.")
    parser.add_argument("--dataset_config", type=str, default="'wikitext-103-raw-v1'", help="Dataset config or path.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (e.g., 'train').")

    parser.add_argument("--num_train_samples", type=int, default=2000, help="Number of training samples.")
    parser.add_argument("--num_val_samples", type=int, default=128, help="Number of perplexity validation samples.")
    parser.add_argument("--num_val_gen_samples", type=int, default=32, help="Number of generation validation samples.")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length for tokenization.")

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader.")
    parser.add_argument("--val_batch_size", type=int, default=8, help="Batch size for DataLoader.")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs to train.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for optimizer.")

    parser.add_argument("--validation_freq", type=int, default=250, help="Steps between validations.")
    parser.add_argument("--gen_context_len", type=int, default=32, help="Context length for generation validation.")
    parser.add_argument("--gen_max_length", type=int, default=1025, help="Max length for generation validation.")
    parser.add_argument("--gen_ttr_window_size", type=int, default=96, help="Window size for calculating the TTR of the generated sequences.")

    parser.add_argument("--save_path", type=str, default="results.json", help="Path to save training logs.")
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()
    tokenizer_path = args.tokenizer_path or args.model_path

    if "imagegpt" in args.model_path.lower():
        tokenizer = ImageGPTImageProcessor.from_pretrained(args.model_path)
        model = ImageGPTForCausalImageModeling.from_pretrained(args.model_path)
        model.to(args.device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        model.to(args.device)


    # Create datasets
    print("Creating datasets...")
    train_loader, val_loader, val_gen_loader = create_dataset(
        tokenizer_path=tokenizer_path,
        num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,
        num_val_gen_samples=args.num_val_gen_samples,
        seq_len=args.seq_len,
        dataset_path=args.dataset_path,
        split=args.split,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size
    )

    # Load Model


    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Start training
    print("Starting training...")
    train_model(
        model=model,
        tokenizer=tokenizer,
        dataloader=train_loader,
        val_dataloader=val_loader,
        val_gen_dataloader=val_gen_loader, 
        optimizer=optimizer,
        device=args.device,
        num_epochs=args.num_epochs,
        validation_freq=args.validation_freq,
        gen_context_len=args.gen_context_len,
        gen_max_length=args.gen_max_length,
        gen_ttr_window_size=args.gen_ttr_window_size,
        save_path=args.save_path
    )

if __name__ == "__main__":
    main()