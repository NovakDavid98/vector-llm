import torch
import os
import argparse
from transformers import GPT2Model, GPT2Config, AutoModelForCausalLM
from .model import VectorHModel, VectorHConfig

def transplant_weights(gpt2_model, vector_h_model):
    """
    Transplants weights from GPT-2 to Vector-H.
    """
    print("Starting Weight Transplantation...")
    
    # 1. Embeddings & Final Norm
    print("Copying Embeddings and Final Norm...")
    vector_h_model.wte.weight.data = gpt2_model.wte.weight.data.clone()
    vector_h_model.wpe.weight.data = gpt2_model.wpe.weight.data.clone()
    vector_h_model.ln_f.weight.data = gpt2_model.ln_f.weight.data.clone()
    vector_h_model.ln_f.bias.data = gpt2_model.ln_f.bias.data.clone()
    
    # Weight Tying for Head
    vector_h_model.lm_head.weight = vector_h_model.wte.weight

    # 2. Layers
    print(f"Copying {len(gpt2_model.h)} Layers...")
    for i, (g_layer, v_layer) in enumerate(zip(gpt2_model.h, vector_h_model.h)):
        # Layer Norms
        v_layer.ln1.weight.data = g_layer.ln_1.weight.data.clone()
        v_layer.ln1.bias.data = g_layer.ln_1.bias.data.clone()
        
        v_layer.ln2.weight.data = g_layer.ln_2.weight.data.clone()
        v_layer.ln2.bias.data = g_layer.ln_2.bias.data.clone()
        
        # Kinetic Layer (Attention Map)
        # GPT-2 Attention: c_attn is [n_embd, 3 * n_embd] (Conv1D)
        # We need to transpose to [3 * n_embd, n_embd] for Linear
        c_attn_w = g_layer.attn.c_attn.weight.data.t().contiguous()
        c_attn_b = g_layer.attn.c_attn.bias.data
        
        n_embd = vector_h_model.config.n_embd
        
        # Split Q, K, V
        q_w, k_w, v_w = c_attn_w.split(n_embd, dim=0)
        q_b, k_b, v_b = c_attn_b.split(n_embd, dim=0)
        
        # Assign to Kinetic Projections
        # Note: GPT-2 Attention head splitting is usually done inside the attention mechanism.
        # Here we map the entire projection matrix to our global Kinetic projections.
        # This assumes we want to preserve the global linear transformation logic.
        
        v_layer.kinetic.proj_q.weight.data = q_w.contiguous()
        v_layer.kinetic.proj_q.bias.data = q_b.contiguous()
        
        v_layer.kinetic.proj_k.weight.data = k_w.contiguous()
        v_layer.kinetic.proj_k.bias.data = k_b.contiguous()
        
        v_layer.kinetic.proj_v.weight.data = v_w.contiguous()
        v_layer.kinetic.proj_v.bias.data = v_b.contiguous()
        
        # NOTE: GPT-2 uses Multi-Head Attention. We are collapsing this into a single "Kinetic Phase Space".
        # The specific head structure is lost, but the linear transform power is preserved.
        # This is a bold approximation (The "Brain Transplant").
        
        # Potential MLP
        # GPT-2 MLP: c_fc [n_embd, 4*n_embd] -> c_proj [4*n_embd, n_embd]
        # Again, likely Conv1D, so transpose needed.
        
        c_fc_w = g_layer.mlp.c_fc.weight.data.t().contiguous()
        c_fc_b = g_layer.mlp.c_fc.bias.data
        
        v_layer.potential_mlp[0].weight.data = c_fc_w
        v_layer.potential_mlp[0].bias.data = c_fc_b
        
        c_proj_w = g_layer.mlp.c_proj.weight.data.t().contiguous()
        c_proj_b = g_layer.mlp.c_proj.bias.data
        
        v_layer.potential_mlp[2].weight.data = c_proj_w
        v_layer.potential_mlp[2].bias.data = c_proj_b
        
        print(f"  Layer {i} transplanted.")

    print("Transplantation Complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./models/vector_h_gpt2")
    args = parser.parse_args()
    
    print("Loading Source GPT-2 Small...")
    # Using 'gpt2' (124M)
    gpt2_full = AutoModelForCausalLM.from_pretrained("gpt2")
    gpt2_model = gpt2_full.transformer # Extract the trunk
    
    print("Initializing Receptor Vector-H...")
    config = VectorHConfig(
        n_embd=768,
        n_layer=12,
        n_head=12, # Kept for config compatibility, though unused in KineticLayer logic directly
        dt=0.1
    )
    vector_h = VectorHModel(config)
    
    transplant_weights(gpt2_model, vector_h)
    
    print(f"Saving to {args.save_path}...")
    os.makedirs(args.save_path, exist_ok=True)
    vector_h.save_pretrained(args.save_path, safe_serialization=False)
    print("Done.")

if __name__ == "__main__":
    main()
