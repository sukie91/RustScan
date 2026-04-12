struct Params {
    mode: u32,
    len: u32,
    padded_len: u32,
    stage_k: u32,
    stage_j: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input_keys: array<u32>;
@group(0) @binding(1) var<storage, read> input_values: array<u32>;
@group(0) @binding(2) var<storage, read_write> work_keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> work_values: array<u32>;
@group(0) @binding(4) var<storage, read_write> work_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> output_keys: array<u32>;
@group(0) @binding(6) var<storage, read_write> output_values: array<u32>;
@group(0) @binding(7) var<storage, read> params_data: Params;

fn should_swap(lhs_key: u32, lhs_index: u32, rhs_key: u32, rhs_index: u32, ascending: bool) -> bool {
    if (ascending) {
        return (lhs_key > rhs_key) || ((lhs_key == rhs_key) && (lhs_index > rhs_index));
    }
    return (lhs_key < rhs_key) || ((lhs_key == rhs_key) && (lhs_index < rhs_index));
}

@compute @workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (params_data.mode == 0u) {
        if (idx >= params_data.padded_len) {
            return;
        }
        if (idx < params_data.len) {
            work_keys[idx] = input_keys[idx];
            work_values[idx] = input_values[idx];
            work_indices[idx] = idx;
        } else {
            work_keys[idx] = 0xffffffffu;
            work_values[idx] = 0xffffffffu;
            work_indices[idx] = 0xffffffffu;
        }
        return;
    }

    if (params_data.mode == 1u) {
        if (idx >= params_data.padded_len) {
            return;
        }

        let partner = idx ^ params_data.stage_j;
        if (partner <= idx || partner >= params_data.padded_len) {
            return;
        }

        let ascending = (idx & params_data.stage_k) == 0u;
        let lhs_key = work_keys[idx];
        let rhs_key = work_keys[partner];
        let lhs_value = work_values[idx];
        let rhs_value = work_values[partner];
        let lhs_index = work_indices[idx];
        let rhs_index = work_indices[partner];

        if (should_swap(lhs_key, lhs_index, rhs_key, rhs_index, ascending)) {
            work_keys[idx] = rhs_key;
            work_keys[partner] = lhs_key;
            work_values[idx] = rhs_value;
            work_values[partner] = lhs_value;
            work_indices[idx] = rhs_index;
            work_indices[partner] = lhs_index;
        }
        return;
    }

    if (idx >= params_data.len) {
        return;
    }

    output_keys[idx] = work_keys[idx];
    output_values[idx] = work_values[idx];
}
