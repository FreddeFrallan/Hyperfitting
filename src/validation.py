import numpy as np
import torch
import tqdm

def _validate_model_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_entropy = 0
    num_batches = 0

    with torch.no_grad():  # Disable gradient computation for validation
        for contexts, targets in tqdm.tqdm(dataloader, desc='Validation'):
            contexts, targets = contexts.to(device), targets.to(device)

            outputs = model(input_ids=contexts, labels=targets)  # Assuming the model returns logits
            logits = outputs.logits
            # loss = loss_func(logits, targets)
            loss = outputs.loss

            total_loss += loss.item()

            # Calculate entropy for each predicted distribution
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1).mean()
            total_entropy += entropy.item()

            num_batches += 1

    average_loss = total_loss / len(dataloader)
    average_entropy = total_entropy / num_batches

    print(f"Validation Loss: {average_loss:.4f}")
    print(f"Average Entropy: {average_entropy:.4f}")
    return average_loss, average_entropy


def _generation_validation(model, tokenizer, dataloader, context_len, max_length=None, ttr_window_size=96):
    """
    Perform generation validation and calculate the average Type-Token Ratio (TTR) of the generated sequences.

    Args:
        model: The language model to generate sequences (must be compatible with transformers.generate).
        tokenizer: The tokenizer corresponding to the model.
        dataloader: DataLoader providing the input data.
        context_len: Number of tokens used as the context for generation.
        max_length: The maximum length for generated sequences (including context).

    Returns:
        A tuple containing:
        - The average TTR of the generated sequences.
        - A list of dictionaries with "context" and "generated_continuation" entries.
    """

    ttrs = []
    results = []
    with torch.no_grad():
        for contexts, targets in tqdm.tqdm(dataloader, desc='Generation Validation'):
            if(max_length is None):
                max_length = contexts.shape[-1] # Set default max_length to the full sequence length, unless specified
            contexts = contexts[:, :context_len].to(model.device)  # Use only the specified context length

            # Generate sequences
            generated_sequences = model.generate(
                input_ids=contexts,
                max_length=max_length,
                do_sample=False,  # Deterministic decoding (e.g., greedy)
                pad_token_id=model.config.eos_token_id
            )

            # Decode and calculate TTR
            for context, gen_seq in zip(contexts, generated_sequences):
                # Decode context and generated sequence into text
                context_text = tokenizer.decode(context.tolist(), skip_special_tokens=True)
                gen_text = tokenizer.decode(gen_seq[context_len:].tolist(), skip_special_tokens=True)

                # Add to results list
                results.append({
                    "context": context_text,
                    "generated_continuation": gen_text
                })

                # Calculate TTR for the generated sequence (excluding context)
                ttr_seqs = gen_seq[context_len:][-ttr_window_size:]
                unique_tokens = len(set(ttr_seqs.tolist()))
                total_tokens = len(ttr_seqs)
                ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
                ttrs.append(ttr)

    average_ttr = np.mean(ttrs) if ttrs else 0
    print(f"Average TTR: {average_ttr:.4f}")
    return average_ttr, results


def validate_model(model, tokenizer, val_dataloader, val_gen_dataloader, device, gen_context_len, gen_max_length=None, gen_ttr_window_size=96):
    val_loss, val_entropy = _validate_model_perplexity(model, val_dataloader, device)
    val_ttr, val_seqs = _generation_validation(model, tokenizer, val_gen_dataloader, gen_context_len, gen_max_length, gen_ttr_window_size)

    return val_loss, val_entropy, val_ttr, val_seqs