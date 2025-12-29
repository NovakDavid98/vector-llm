import torch
import torch.nn as nn
import math
from transformers import PreTrainedModel, PretrainedConfig

class VectorHConfig(PretrainedConfig):
    model_type = "vector_h"
    def __init__(
        self,
        n_embd=768,
        n_layer=12,
        n_head=12,
        vocab_size=50257,
        n_positions=1024,
        layer_norm_epsilon=1e-5,
        dt=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dt = dt



# JIT Compilation removed - sticking to Pure Python for Ivy Bridge compatibility
# PyTorch internal ATen kernels are already AVX1 optimized.
USE_CPP = False

class KineticFlowLayer(nn.Module):
    """
    CPU-Optimized Kinetic Layer.
    Replaces Multi-Head Attention with an O(N) Recurrent State Update
    governed by Hamiltonian physics.
    """
    def __init__(self, config):
        super().__init__()
        self.dt = config.dt
        self.n_embd = config.n_embd
        
        # Physics Parameters
        # decay: Friction/Energy dissipation (avoids explosion)
        self.decay = nn.Parameter(torch.ones(self.n_embd) * 0.9) 
        # mass: Inertia (resistance to change)
        self.mass = nn.Parameter(torch.ones(self.n_embd))
        
        # Projections (Mapped from GPT Attention weights)
        # Query -> Position (q) interactions
        self.proj_q = nn.Linear(self.n_embd, self.n_embd) 
        # Key -> Momentum Force (How much the input pushes the state)
        self.proj_k = nn.Linear(self.n_embd, self.n_embd) 
        # Value -> Input Energy (The actual content added)
        self.proj_v = nn.Linear(self.n_embd, self.n_embd)

    def forward(self, x, state_p=None):
        """
        x: Input token embedding [Batch, SeqLen, Dim] or [Batch, Dim]
        state_p: Previous momentum state [Batch, Dim]
        
        Returns:
            velocity: The output 'movement' vector
            new_p: Updated momentum state
        """
        batch_size = x.size(0)
        
        if state_p is None:
            state_p = torch.zeros(batch_size, self.n_embd, device=x.device, dtype=x.dtype)
            
        # If x has sequence length > 1, apply scan
        if x.dim() == 3 and x.size(1) > 1:
            velocities = []
            current_p = state_p
            for t in range(x.size(1)):
                xt = x[:, t, :]
                
                # 1. Projections (MatMul is already optimized in PyTorch MKL)
                k_logit = self.proj_k(xt)
                v = self.proj_v(xt)
                
                # 2. Update Momentum (p) - Recurrent Step
                # Pure Python (PyTorch AVX1 backend)
                force = torch.sigmoid(k_logit)
                energy_in = v
                current_p = (current_p * self.decay) + (force * energy_in)
                
                # 3. Apply Kinetic Output (Velocity)
                velocity = self.proj_q(current_p) / self.mass
                velocities.append(velocity)
            
            return torch.stack(velocities, dim=1), current_p
        
        else:
            # Single step (inference or len=1)
            if x.dim() == 3:
                x = x.squeeze(1)
            
            # 1. Projections
            k_logit = self.proj_k(x)
            v = self.proj_v(x)
            
            # 2. Update Momentum
            force = torch.sigmoid(k_logit)
            energy_in = v
            new_p = (state_p * self.decay) + (force * energy_in)
            
            velocity = self.proj_q(new_p) / self.mass
            
            return velocity, new_p

class VectorHamiltonianBlock(nn.Module):
    """
    Combines Kinetic Flow and Potential Energy using Symplectic Leapfrog Integration.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.kinetic = KineticFlowLayer(config)
        
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        # Potential Energy function (Force field)
        # Typically an MLP in Transformers
        self.potential_mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd)
        )

    def forward(self, q, p):
        # Symplectic Leapfrog Integration
        
        # 1. Kinetic Half-Step (Movement)
        # Note: In standard Transformer, this is Residual(Attn(Norm(x)))
        # Here: q_new = q + velocity * dt
        
        q_norm = self.ln1(q)
        
        # Kinetic layer returns 'velocity' and updated internal momentum 'p_internal' (RNN state)
        # But wait, the Spec 2.3 defines 'p' as a global phase space variable too.
        # "Every hidden state h is split into two components: q and p"
        # However, KineticFlowLayer spec uses 'state_p' as a recurrent memory.
        # Let's align with Component B spec:
        # forward(self, q, p):
        #   velocity, p_new = self.kinetic(q_norm, p)
        #   ... 
        
        # The 'p' passed into the block acts as the 'state_p' for the kinetic layer?
        # Yes, "state_p: Previous momentum state (Hidden Memory)"
        
        # But the Spec 2.3 also says:
        # "p_half = p_t - dt/2 * grad_V(q_t)"
        # "q_{t+1} = ... "
        
        # Component B implementation in Spec:
        # velocity, p_new = self.kinetic(q_norm, p)
        # q_new = q + velocity 
        # force = self.potential_mlp(self.ln2(q_new))
        # p_final = p_new - force
        
        # This matches the Spec logic for Component B.
        
        # Handling Sequence Dimension for 'q'
        # If q is [Batch, Seq, Dim], we need to run kinetic per step or scan.
        # KineticFlowLayer handles scan.
        
        # Check if we are in sequence mode
        is_seq = (q.dim() == 3) and (q.size(1) > 1)
        
        velocity, p_new = self.kinetic(q_norm, p)
        
        # If sequence mode, velocity is [Batch, Seq, Dim], p_new is [Batch, Dim] (last state)
        # Wait, if we are processing a sequence, the 'p' state evolves at every token.
        # But in a Transformer-like block stack, 'p' is usually the hidden state carried FORWARD in depth?
        # OR is it carried ACROSS time?
        # "Linear scaling allowing infinite context on RAM" -> Carried ACROSS TIME.
        
        # So 'p' is the RNN state. 'q' is the Transformer layer input (hidden states from below).
        # In a sequence forward pass (training):
        # q: [Batch, Seq, Dim]
        # p_initial: [Batch, Dim] (state at t=0)
        
        # We need the full sequence of p's to compute forces at each step?
        # Actually, if KineticLayer returns 'velocity' for the whole sequence,
        # we can compute q_new for the whole sequence.
        
        if is_seq:
            # velocity: [B, S, D]
            # q: [B, S, D]
            # q_new = q + velocity (Broadcast or elementwise)
            # Note: This is different from x += attn(x). It's q += velocity.
            # velocity effectively IS the attn(x) output in this analogy.
            pass
        
        # Symplectic Update: q_new = q + velocity * dt
        # We treat 'velocity' as the update term (like residual).
        # But we should multiply by dt? Spec Code: "q_new = q + velocity" (Implies velocity includes dt effect or dt=1)
        # Spec 2.3 says: "q_{t+1} = q_t + dt * ... "
        # Spec Code Component A: "velocity = ... / self.mass"
        # Spec Code Component B: "q_new = q + velocity"
        # I will stick to the provided code snippet for Component B which omits explicit `* dt` 
        # (assuming it's baked into the learned mass/velocity or dt=1 for the forward pass simplicity).
        # Actually, `KineticFlowLayer` has `self.dt`. Maybe it should use it?
        # The spec code for KineticFlowLayer `forward` does NOT use `dt` in the return `velocity`.
        # I will stick to the Literal Spec Code to be safe, but keep `dt` in mind.
        
        q_new = q + velocity
        
        # 2. Potential Half-Step (Force)
        q_norm_2 = self.ln2(q_new)
        force = self.potential_mlp(q_norm_2)
        
        # Symplectic Update: p_final = p_new - force
        # Wait, p_new returned by kinetic is the LAST state in sequence mode.
        # But 'force' is computed per token?
        # AND 'p' is the RNN state.
        # If this is a stacked model (Layers 1..12):
        # Does 'p' propagate up the layers? Or is 'p' local to each layer (like KV cache)?
        # "Every hidden state h is split into two components: q and p"
        # Usually RNN states are local to the layer.
        # So `p` is the hidden state of THIS layer.
        
        # ISSUE: In sequence mode, `p_new` from KineticLayer is just the FINAL state.
        # But `p_final` in Component B implies we are updating the current 'p' to be passed to... where?
        # If 'p' is the RNN state, we only output the final 'p' for the NEXT token generation step.
        # The 'q' (hidden states) continues up the stack to the next layer.
        
        # In Component B code: `return q_new, p_final`
        # It returns updated q (to go up) and updated p (to go to next time step / next inference step).
        
        if is_seq:
             # In sequence mode, p_new is the final state from Kinetic Layer.
             # We update it with the force from the final step of the Potential Layer.
             p_final = p_new - force[:, -1, :]
        else:
             p_final = p_new - force
        
        return q_new, p_final

class VectorHModel(PreTrainedModel):
    config_class = VectorHConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(0.1) # Standard GPT-2 dropout
        
        self.h = nn.ModuleList([
            VectorHamiltonianBlock(config) for _ in range(config.n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Language Model Head (tied to wte usually, but we define explicit if needed or use wte)
        # GPT-2 ties weights. We can do `lm_head = nn.Linear(..., bias=False)` and tie later.
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight # Weight tying

    def forward(self, input_ids, states_p=None, labels=None):
        """
        input_ids: [Batch, Seq]
        states_p: List of momentum states for each layer. 
                  Each element: [Batch, Dim]
                  If None, initializes to zeros.
        """
        device = input_ids.device
        b, s = input_ids.size()
        
        # Embeddings
        pos_ids = torch.arange(0, s, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos_ids)
        x = self.drop(tok_emb + pos_emb)
        
        new_states_p = []
        
        # Initialize states if None
        if states_p is None:
            states_p = [None] * self.config.n_layer
            
        for i, block in enumerate(self.h):
            layer_p = states_p[i]
            x, new_p = block(x, layer_p)
            new_states_p.append(new_p)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return {
            "logits": logits,
            "new_states_p": new_states_p,
            "loss": loss
        }
