// Fullscreen triangle render shader for NCA state visualization.
// Reads RGBA from the NCA state storage buffer and composites over a background color.

struct RenderUniforms {
    bg: vec4f,      // .rgb = background color
    dims: vec2u,    // .x = grid_w, .y = grid_h
}

@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: RenderUniforms;

struct VSOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VSOut {
    // 3-vertex fullscreen triangle covering the entire viewport
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    );
    var out: VSOut;
    out.pos = vec4f(positions[vid], 0.0, 1.0);
    out.uv = vec2f(
        (positions[vid].x + 1.0) * 0.5,
        (1.0 - positions[vid].y) * 0.5,
    );
    return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
    let gx = u32(floor(in.uv.x * f32(uniforms.dims.x)));
    let gy = u32(floor(in.uv.y * f32(uniforms.dims.y)));

    if (gx >= uniforms.dims.x || gy >= uniforms.dims.y) {
        return vec4f(uniforms.bg.rgb, 1.0);
    }

    let base = (gy * uniforms.dims.x + gx) * 16u;
    let r = clamp(state[base + 0u], 0.0, 1.0);
    let g = clamp(state[base + 1u], 0.0, 1.0);
    let b = clamp(state[base + 2u], 0.0, 1.0);
    let a = clamp(state[base + 3u], 0.0, 1.0);

    // Alpha-composite over background
    let rgb = vec3f(r, g, b) * a + uniforms.bg.rgb * (1.0 - a);
    return vec4f(rgb, 1.0);
}
