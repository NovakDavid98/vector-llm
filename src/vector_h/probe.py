import argparse
import sys
import numpy as np
from llama_cpp import Llama

# Config
MODEL_PATH = "./models/tinyllama-1.1b-chat.Q4_K_M.gguf"
ENERGY_THRESHOLD = 0.05 # Tunable threshold for "Surprise/Confusion"

def calculate_hamiltonian(q_curr, q_prev):
    """
    H = T(p) + V(q)
    p = q_curr - q_prev (Velocity/Momentum)
    """
    # Momentum (Rate of change of thought)
    p = q_curr - q_prev
    
    # Kinetic Energy (Motion)
    T = 0.5 * np.sum(p**2)
    
    # Potential Energy (State Magnitude - how 'heavy' is the concept)
    # Scaled down to match T roughly
    V = 0.5 * np.mean(q_curr**2) 
    
    return T + V, T, V

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Explain quantum physics to a 5 year old.")
    parser.add_argument("--threshold", type=float, default=ENERGY_THRESHOLD)
    args = parser.parse_args()

    print(f"Loading Physics Supervisor... (Model: {MODEL_PATH})")
    try:
        # 1. Generation Engine (The "Driver")
        llm_gen = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=0,
            n_ctx=2048,
            verbose=False
        )
        
        # 2. Physics Monitor (The "Sidecar")
        # Separate instance to avoid clobbering the generation context
        # Since we use mmap, the theoretical RAM usage for weights should be shared/deduplicated by OS
        llm_probe = Llama(
            model_path=MODEL_PATH,
            embedding=True,
            n_gpu_layers=0,
            n_ctx=512, # Smaller context for probe lookups
            verbose=False
        )
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print(f"\n[SYSTEM READY] Monitoring for Energy Spikes > {args.threshold}...\n")
    print("-" * 50)
    print(f"PROMPT: {args.prompt}")
    print("-" * 50)
    
    # Get initial embedding
    prev_embedding = np.zeros(2048) 
    
    # Stream generation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt}
    ]
    
    stream = llm_gen.create_chat_completion(
        messages=messages,
        max_tokens=256,
        stream=True
    )
    
    total_energy_flux = 0
    token_count = 0
    
    print("\nGENERATION:")
    
    for output in stream:
        delta = output['choices'][0]['delta']
        if 'content' not in delta:
            continue
            
        token_text = delta['content']
        
        # 1. Physics Probe
        try:
            # Use the PROBE instance, not GEN
            embed_resp = llm_probe.create_embedding(token_text)
            current_embedding = np.array(embed_resp['data'][0]['embedding'])
            current_embedding = current_embedding / (np.linalg.norm(current_embedding) + 1e-9) # Normalize
            
            # 2. Calculate Energy
            energy, kinetic, potential = calculate_hamiltonian(current_embedding, prev_embedding)
            
            # 3. Supervisor Check
            if token_count > 0:
                total_energy_flux += energy
                
                if energy > args.threshold:
                    # HALLUCINATION / SURPRISE (High Energy)
                    # Use Red Background or Bold Red
                    print(f"\033[1;31m[SPIKE:{energy:.2f}]{token_text}\033[0m", end="", flush=True) 
                elif energy < 0.001:
                    # STAGNATION / LOOP (Zero Energy)
                    # Use Blue
                    print(f"\033[1;34m{token_text}\033[0m", end="", flush=True)
                else:
                    # FLOW (Medium Energy)
                    # Green
                    print(f"\033[1;32m{token_text}\033[0m", end="", flush=True)
            else:
                print(token_text, end="", flush=True)
                
            prev_embedding = current_embedding
            token_count += 1
            
        except Exception as e:
            print(token_text, end="", flush=True)
            
    print("\n" + "-" * 50)
    print(f"Average Energy: {total_energy_flux / max(1, token_count):.4f}")
    print("[LEGEND] \033[1;32mGreen=Flow\033[0m | \033[1;31mRed=Surprise (Spike)\033[0m | \033[1;34mBlue=Loop (Stagnation)\033[0m")

if __name__ == "__main__":
    main()
