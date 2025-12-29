import torch
import torch.nn as nn
import time
import argparse
import numpy as np
from transformers import AutoTokenizer
from .model import VectorHModel

# Optimization for Xeon E5-2697 v2 (12 Cores / 24 Threads)
# maximize throughput by utilizing all threads for cache prefetching
torch.set_num_threads(24)

def calculate_energy(p_states):
    # E = 0.5 * sum(p^2)
    total_energy = 0
    for p in p_states:
        if p is not None:
            total_energy += 0.5 * torch.mean(torch.sum(p ** 2, dim=-1))
    return total_energy

def drift_test(model, tokenizer, device, context_len=2000):
    print(f"\nrunning Drift Test (Context Length: {context_len})...")
    
    # Generate random text or repeat a phrase to fill context
    text = "The quick brown fox jumps over the lazy dog. " * (context_len // 9)
    input_ids = tokenizer(text, return_tensors="pt").input_ids[:, :context_len].to(device)
    
    # Split into chunks of 1024 (GPT-2 max pos) to simulate infinite stream
    # Note: VectorH avoids O(N^2) but we still use GPT-2 positional embeddings which cap at 1024.
    # To go beyond, we rely on the recurrent state 'p' carrying the context.
    # We will reset positional embeddings for each chunk but KEEP the state p.
    
    chunk_size = 1024
    num_chunks = (input_ids.size(1) + chunk_size - 1) // chunk_size
    
    states_p = None
    all_logits = []
    
    start_time = time.time()
    
    for i in range(num_chunks):
        chunk = input_ids[:, i*chunk_size : (i+1)*chunk_size]
        
        with torch.no_grad():
            outputs = model(chunk, states_p=states_p)
            states_p = outputs["new_states_p"]
            logit = outputs["logits"]
            all_logits.append(logit)
            
            # Monitoring Energy
            e = calculate_energy(states_p)
            print(f"  Chunk {i}: Energy of State = {e.item():.4f}")

    end_time = time.time()
    print(f"Drift Test Complete. Processed {input_ids.size(1)} tokens in {end_time - start_time:.2f}s")
    
    # Check coherence? Hard to check automatically without a reference task.
    # We rely on Energy stability printing above.

def truth_test(model, tokenizer, device):
    print("\nRunning Truth Test (Energy Variance)...")
    prompt = "Physics is the study of"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    states_p = None
    energies = []
    
    # Generate 50 tokens token-by-token
    print(f"Generating from: '{prompt}'")
    generated = input_ids
    
    for _ in range(50):
        with torch.no_grad():
            # Feed only the last token if we have state? 
            # Or feed full sequence if we don't carry state?
            # To test 'Energy Conservation', we must carry state.
            
            # Step-wise generation
            if states_p is None:
                current_input = generated
            else:
                current_input = generated[:, -1:] # Last token
                
            outputs = model(current_input, states_p=states_p)
            states_p = outputs["new_states_p"]
            next_token_logits = outputs["logits"][:, -1, :]
            
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            
            e = calculate_energy(states_p)
            energies.append(e.item())
    
    text = tokenizer.decode(generated[0])
    print(f"Generated: {text}")
    
    # Analyze Energy
    energies = np.array(energies)
    variance = np.var(energies)
    print(f"Energy Variance: {variance:.6f}")
    if variance > 1.0: # Arbitrary threshold
        print("WARNING: High Energy Variance detected! (Hallucination Risk)")
    else:
        print("PASS: System Energy is stable.")

def throughput_test(model, tokenizer, device):
    print("\nRunning Throughput Test...")
    # Generate for 5 seconds
    input_ids = tokenizer("Test", return_tensors="pt").input_ids.to(device)
    states_p = None
    
    start_time = time.time()
    count = 0
    
    with torch.no_grad():
        while time.time() - start_time < 5.0:
            current_input = input_ids[:, -1:] if states_p is not None else input_ids
            outputs = model(current_input, states_p=states_p)
            states_p = outputs["new_states_p"]
            
            # Dummy sampling
            next_token = torch.argmax(outputs["logits"][:, -1, :], dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            count += 1
            
    avg_tps = count / 5.0
    print(f"Throughput: {avg_tps:.2f} tokens/sec")

def verify(args):
    device = torch.device("cpu")
    print(f"Loading model from {args.model_path}")
    print("Optimization: CPU Mode (Xeon)")
    
    model = VectorHModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    if args.test in ["all", "drift"]:
        drift_test(model, tokenizer, device)
    
    if args.test in ["all", "truth"]:
        truth_test(model, tokenizer, device)
        
    if args.test in ["all", "throughput"]:
        throughput_test(model, tokenizer, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/vector_h_gpt2")
    parser.add_argument("--test", type=str, default="all", choices=["all", "drift", "truth", "throughput"])
    args = parser.parse_args()
    
    verify(args)
