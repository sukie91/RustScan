struct Params {
    len: u32,
    offset: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input_values: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_values: array<u32>;
@group(0) @binding(2) var<storage, read> params_data: Params;

@compute @workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params_data.len) {
        return;
    }

    if (params_data.offset == 0u) {
        output_values[idx] = input_values[idx];
        return;
    }

    var value = input_values[idx];
    if (idx >= params_data.offset) {
        value = value + input_values[idx - params_data.offset];
    }
    output_values[idx] = value;
}
