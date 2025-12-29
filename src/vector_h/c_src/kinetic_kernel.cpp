#include <torch/extension.h>
#include <vector>
#include <immintrin.h>
#include <cmath>

#if defined(__AVX512F__)
inline __m512 sigmoid_avx512(__m512 x) {
    alignas(64) float temp[16];
    _mm512_store_ps(temp, x);
    for(int i=0; i<16; ++i) {
        temp[i] = 1.0f / (1.0f + std::exp(-temp[i]));
    }
    return _mm512_load_ps(temp);
}
#endif

// Fuses: p_new = (p_old * decay) + (sigmoid(k) * v)
torch::Tensor kinetic_update_forward(
    torch::Tensor p_old,
    torch::Tensor k_logits,
    torch::Tensor v,
    torch::Tensor decay) {

    auto p_new = torch::empty_like(p_old);
    
    int64_t size = p_old.numel();
    float* p_old_ptr = p_old.data_ptr<float>();
    float* k_ptr = k_logits.data_ptr<float>();
    float* v_ptr = v.data_ptr<float>();
    float* decay_ptr = decay.data_ptr<float>();
    float* p_new_ptr = p_new.data_ptr<float>();
    
    int64_t i = 0;
    
    #if defined(__AVX512F__)
    // Explicit AVX-512 Path
    int64_t main_loop_end = size - (size % 16);
    for (; i < main_loop_end; i += 16) {
        __m512 p = _mm512_loadu_ps(p_old_ptr + i);
        __m512 k = _mm512_loadu_ps(k_ptr + i);
        __m512 val = _mm512_loadu_ps(v_ptr + i);
        __m512 dec = _mm512_loadu_ps(decay_ptr + i);
        
        __m512 sig_k = sigmoid_avx512(k);
        __m512 force_energy = _mm512_mul_ps(sig_k, val);
        __m512 p_decay = _mm512_mul_ps(p, dec);
        __m512 res = _mm512_add_ps(p_decay, force_energy);
        
        _mm512_storeu_ps(p_new_ptr + i, res);
    }
    #endif
    
    // Scalar Fallback (Compiler Auto-Vectorized with -O3 -mavx2)
    #pragma omp parallel for if(size > 4096)
    for (int64_t j = i; j < size; j++) {
        float sig_k = 1.0f / (1.0f + std::exp(-k_ptr[j]));
        // Fuse ops
        p_new_ptr[j] = (p_old_ptr[j] * decay_ptr[j]) + (sig_k * v_ptr[j]);
    }
    
    return p_new;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kinetic_update_forward", &kinetic_update_forward, "Vector-H Kinetic Update (AVX-512/AVX2)");
}
