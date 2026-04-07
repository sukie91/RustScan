#include <metal_stdlib>
using namespace metal;

kernel void fill_u32(
    device uint* dst [[buffer(0)]],
    constant uint& value [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    dst[gid] = value;
}
