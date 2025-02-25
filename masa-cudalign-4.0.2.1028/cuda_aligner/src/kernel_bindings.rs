use cuda_runtime_sys as cuda;
use std::ffi::c_void;

#[link(name = "cuda_kernels", kind = "static")]
extern "C" {
    fn launch_alignment_kernel(
        grid: cuda::dim3,
        threads: cuda::dim3,
        bus_h: *mut cuda::int2,
        bus_v_h: *mut cuda::int4,
        bus_v_e: *mut cuda::int4,
        bus_v_o: *mut cuda::int3,
        block_result: *mut cuda::int4,
        i0: i32,
        i1: i32,
        step: i32,
        cut_block: cuda::int2,
    );

    fn bind_sequence_textures(
        seq0: *const u8,
        seq0_len: usize,
        seq1: *const u8,
        seq1_len: usize,
    ) -> cuda::cudaError_t;

    fn unbind_sequence_textures();
}

pub(crate) unsafe fn launch_kernel(
    grid: cuda::dim3,
    threads: cuda::dim3,
    bus_h: *mut cuda::int2,
    bus_v_h: *mut cuda::int4,
    bus_v_e: *mut cuda::int4,
    bus_v_o: *mut cuda::int3,
    block_result: *mut cuda::int4,
    i0: i32,
    i1: i32,
    step: i32,
    cut_block: cuda::int2,
) {
    launch_alignment_kernel(
        grid,
        threads,
        bus_h,
        bus_v_h,
        bus_v_e,
        bus_v_o,
        block_result,
        i0,
        i1,
        step,
        cut_block,
    );
}

pub(crate) unsafe fn bind_textures(
    seq0: *const u8,
    seq0_len: usize,
    seq1: *const u8,
    seq1_len: usize,
) -> cuda::cudaError_t {
    bind_sequence_textures(seq0, seq0_len, seq1, seq1_len)
}

pub(crate) unsafe fn unbind_textures() {
    unbind_sequence_textures();
}
