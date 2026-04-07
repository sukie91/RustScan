#include <metal_stdlib>
using namespace metal;

struct AdamHyperparams {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float bc1;
    float bc2;
};

kernel void adam_step(
    device float* params [[buffer(0)]],
    device const float* grads [[buffer(1)]],
    device float* m_buf [[buffer(2)]],
    device float* v_buf [[buffer(3)]],
    constant AdamHyperparams& hp [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const float g = grads[gid];
    float m = hp.beta1 * m_buf[gid] + (1.0f - hp.beta1) * g;
    float v = hp.beta2 * v_buf[gid] + (1.0f - hp.beta2) * g * g;
    m_buf[gid] = m;
    v_buf[gid] = v;
    const float m_hat = m * hp.bc1;
    const float v_hat = v * hp.bc2;
    params[gid] -= hp.lr * m_hat / (sqrt(v_hat) + hp.eps);
}
