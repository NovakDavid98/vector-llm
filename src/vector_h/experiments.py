import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from transformers import AutoTokenizer
from .model import VectorHModel

# Optimization for Xeon
torch.set_num_threads(24)

def calculate_energy(p_states):
    total_energy = 0
    for p in p_states:
        if p is not None:
            # E = 0.5 * p^2
            total_energy += 0.5 * torch.mean(torch.sum(p ** 2, dim=-1))
    return total_energy

def lyapunov_test(model, tokenizer, device):
    print("\n[Experiment 1] Lyapunov Stability Analysis")
    # Hypothesis: Vector-H trajectories should be stable (bounded divergence) unlike chaotic RNNs.
    
    prompt = "The laws of physics imply that"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # 1. Base Trajectory
    with torch.no_grad():
        out_base = model(input_ids)
        states_base = out_base["new_states_p"]
        embedding_base = model.wte(input_ids) # Re-compute for perturbation math
    
    # 2. Perturbed Trajectory (simulate noise in input)
    # We can't easily perturb discrete tokens, so we perturb the embedding?
    # VectorHModel doesn't expose embedding forward easily without hack.
    # We will perturb the 'state_p' initialization slightly.
    
    epsilon = 1e-4
    perturbed_states = [torch.randn_like(p) * epsilon if p is not None else None for p in states_base] # No, this is final.
    
    # Let's run forward with a small initial momentum perturbation at t=0
    # Actually, model assumes zero initial momentum.
    # Let's verify divergence over time steps.
    
    # Better: Run generation for 20 steps.
    # Trajectory 1: Normal
    # Trajectory 2: Perturbed at step 0 (how?) -> Maybe manually feed embedding + noise?
    # Simpler: Perturb the 'decay' parameter temporarily? No.
    
    # Let's stick to comparing two very similar prompts.
    # "The solar system is huge" vs "The solar system is vast"
    # Measure divergence of P-states.
    
    p1 = "The solar system is huge"
    p2 = "The solar system is vast"
    
    t1 = tokenizer(p1, return_tensors="pt").input_ids.to(device)
    t2 = tokenizer(p2, return_tensors="pt").input_ids.to(device) 
    
    print(f"Comparing trajectories for:\n  A: '{p1}'\n  B: '{p2}'")
    
    divergences = []
    
    # Generate 20 tokens from each
    s1 = None
    s2 = None
    gen1 = t1
    gen2 = t2
    
    for i in range(20):
        with torch.no_grad():
            o1 = model(gen1[:, -1:], states_p=s1)
            o2 = model(gen2[:, -1:], states_p=s2)
            
            s1 = o1["new_states_p"]
            s2 = o2["new_states_p"]
            
            # Distance in Phase Space (L2 norm of P difference)
            # Sum over layers
            dist = 0
            for l_p1, l_p2 in zip(s1, s2):
                dist += torch.norm(l_p1 - l_p2).item()
            
            divergences.append(dist)
            
            # Greedy decode
            gen1 = torch.cat([gen1, torch.argmax(o1["logits"][:, -1, :], dim=-1).unsqueeze(0)], dim=1)
            gen2 = torch.cat([gen2, torch.argmax(o2["logits"][:, -1, :], dim=-1).unsqueeze(0)], dim=1)
            
    print("Phase Space Divergence over Time:")
    print(divergences)
    
    # Calculate Lyapunov Exponent proxy (slope of log divergence)
    # If slope > 0, chaotic. If <= 0, stable.
    # Real Lyapunov needs limit t->inf and small eps, but this is a rough proxy.
    
    log_div = np.log(np.array(divergences) + 1e-9)
    slope, _ = np.polyfit(np.arange(len(divergences)), log_div, 1)
    
    print(f"Estimated Lyapunov Exponent (Lambda): {slope:.4f}")
    if slope > 0.1:
        print("-> System exhibits CHAOTIC divergence (Sensitive to initial conditions).")
    else:
        print("-> System exhibits STABLE/MARGINAL dynamics (Robust).")

def thermodynamics_test(model, tokenizer, device):
    print("\n[Experiment 2] Thermodynamics of Thought (Temperature Analysis)")
    
    prompts = {
        "Factual": "The capital of France is Paris.",
        "Confusing": "Colorless green ideas sleep furiously.",
        "Gibberish": "Glibglob flimflam wozzle."
    }
    
    results = {}
    
    for label, text in prompts.items():
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(input_ids)
            # Calculate Average Kinetic Energy of the final state
            # This represents the "Temperature" of the thought process?
            # High Energy = High movement/searching/confusion?
            e = calculate_energy(out["new_states_p"]).item()
            results[label] = e
            print(f"  {label:10s} | Energy (Temp): {e:.4f}")
            
    # Check hypothesis
    if results["Gibberish"] > results["Factual"]:
        print("-> CONFIRMED: Nonsense generates higher thermodynamic energy (Heat).")
    else:
        print("-> INCONCLUSIVE: Energy levels not clearly correlated with semantic entropy.")

def run_experiments(args):
    device = torch.device("cpu")
    print(f"Loading model from {args.model_path}")
    model = VectorHModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    lyapunov_test(model, tokenizer, device)
    thermodynamics_test(model, tokenizer, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/vector_h_finetuned")
    args = parser.parse_args()
    run_experiments(args)
