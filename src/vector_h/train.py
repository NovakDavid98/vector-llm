import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import argparse
import os
from tqdm import tqdm

from .model import VectorHModel, VectorHConfig
from .dataset import get_wikitext_dataset

def calculate_energy(p_states):
    """
    Calculates the detailed 'Kinetic Energy' of the system based on momentum states.
    Energy ~ Sum(p^2) / 2m
    Since mass is a parameter, strictly we should divide by it, but for a loss proxy
    Sum(p^2) is a robust enough invariant to target for stability.
    """
    total_energy = 0
    for p in p_states:
        if p is not None:
            # Sum over all dimensions, average over batch
            total_energy += 0.5 * torch.mean(torch.sum(p ** 2, dim=-1))
    return total_energy

def physics_loss(logits, labels, initial_p_states, final_p_states, lambda_energy=0.1):
    # Standard Generation Loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss()
    ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Physics Conservation Loss
    # We want the total momentum energy to be stable through the 'depth' (time) of the network
    # or just stable in magnitude?
    # "Penalize energy creation/destruction" -> E_in vs E_out
    
    # Let's take the energy of the first layer vs the last layer? 
    # Or sum of all layers at t=0 vs t=end?
    
    # Interpretation: The "System" is the set of all Layers.
    # The "Time" is the sequence step.
    # But `p` evolves over sequence steps.
    # We should penalize the change in Total P-Energy from step t to t+1?
    # This captures the "Drift" concern.
    
    # However, `final_p_states` from model.forward returns the states at the END of the sequence (or all steps?)
    # My model implementation returns `new_states_p` which is the LAST state (at token T).
    # Ideally we'd need the per-token states to measure dt energy change.
    
    # SIMPLIFICATION for Proof of Concept:
    # Penalize the variance of p-norms across layers. 
    # If it's a "conserved system", energy shouldn't spiral out of control deep in the network.
    
    e_initial = calculate_energy([initial_p_states[0]] if initial_p_states[0] is not None else [torch.zeros_like(final_p_states[0])]) # Roughly 0
    
    # Let's simple check: Energy should not explode.
    # But strictly: E_final should be related to E_initial.
    # Let's penalize the magnitude of Energy change relative to a baseline.
    
    e_final = calculate_energy(final_p_states)
    
    # We want E to be conserved. But input injects energy.
    # Maybe simply penalize L2 norm of p to keep it bounded (soft conservation/dissipation).
    # The spec says: "mean((initial_energy - final_energy) ** 2)"
    # I will stick to comparing the energy of the First Layer vs Last Layer?
    # No, that's partial. 
    # Let's compare the Average Energy of the system at the start of the batch vs end of the batch?
    # But we don't have start/end batch states easily in a stateless forward.
    
    # Let's use the Spec's exact words: "initial_energy - final_energy".
    # I will assume this means Energy(Layer 0) vs Energy(Layer N).
    # i.e. Does the signal gain/lose energy as it propagates through the "time" of the layers?
    
    e_in = 0.5 * torch.mean(torch.sum(final_p_states[0] ** 2, dim=-1)) # Layer 0
    e_out = 0.5 * torch.mean(torch.sum(final_p_states[-1] ** 2, dim=-1)) # Layer N
    
    conservation_loss = (e_in - e_out) ** 2
    
    return ce_loss + (lambda_energy * conservation_loss), ce_loss, conservation_loss

def train(args):
    device = torch.device("cpu") # Xeon Implementation
    print(f"Using device: {device}")
    
    # Load Model
    print(f"Loading model from {args.model_path}")
    model = VectorHModel.from_pretrained(args.model_path)
    model.to(device)
    model.train()
    
    # Data
    dataset, tokenizer = get_wikitext_dataset()
    tokenizer.pad_token = tokenizer.eos_token
    
    def collate_fn(batch):
        # Stack inputs
        input_ids = [torch.tensor(b["input_ids"]) for b in batch]
        # Pad
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.eos_token_id)
        # Random crop to block size if needed, but they are already truncated
        return input_ids

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    print("Starting Fine-tuning...")
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(pbar):
            input_ids = batch.to(device)
            labels = input_ids.clone()
            
            # Forward
            # Initialize states_p is handled inside model if None
            outputs = model(input_ids, labels=labels)
            
            # Check Energy
            # Outputs.new_states_p is a list of tensors [Batch, Dim]
            
            # Helper to calculate loss
            loss, ce, cons = physics_loss(
                outputs["logits"], 
                labels, 
                # We treat Layer 0 as 'Initial' state of the "wave" entering the system
                outputs["new_states_p"], 
                outputs["new_states_p"]
            )
            # Wait, passing same list for initial/final in my call above is wrong if looking at layers.
            # My physics_loss function extracts [0] and [-1] from the list. 
            # So passing the list once is sufficient.
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"CE": f"{ce.item():.4f}", "Cons": f"{cons.item():.4f}"})
            
            if args.max_steps > 0 and step >= args.max_steps:
                break
        
        print(f"Epoch {epoch+1} Loss: {total_loss / len(loader)}")
        
        # Save Checkpoint
        model.save_pretrained(os.path.join(args.save_path, f"checkpoint-{epoch}"), safe_serialization=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/vector_h_gpt2")
    parser.add_argument("--save_path", type=str, default="./models/vector_h_finetuned")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=100) # Short run for simple verify
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    train(args)
