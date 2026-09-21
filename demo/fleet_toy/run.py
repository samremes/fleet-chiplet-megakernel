#!/usr/bin/env python3
"""Run three small tasks through Fleet's real persistent-kernel runtime."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("AMDGPU_TARGETS", "gfx950")
os.environ.setdefault("USE_GANG", "1")
os.environ.setdefault("PRECOMPUTED_DISPATCH", "0")
os.environ.setdefault("MPK_QUIET_FWDPASS", "1")
os.environ.setdefault("MPK_W13_BIAS_COUNTED_WAIT", "0")

import torch
from mirage.mpk.persistent_kernel import PersistentKernel

M, K, N = 4, 16, 32


def make_meta_tensors(device: torch.device) -> dict[str, torch.Tensor]:
    # Fleet's ABI always accepts these ten pointers. online_notoken mode does
    # not consume serving metadata, so one-element placeholders are sufficient.
    return {
        "step": torch.zeros(1, dtype=torch.int32, device=device),
        "tokens": torch.zeros((1, 2), dtype=torch.int64, device=device),
        "input_tokens": torch.zeros((M, 1), dtype=torch.int64, device=device),
        "output_tokens": torch.zeros((M, 1), dtype=torch.int64, device=device),
        "num_new_tokens": torch.ones(1, dtype=torch.int32, device=device),
        "prompt_lengths": torch.zeros(1, dtype=torch.int32, device=device),
        "qo_indptr_buffer": torch.tensor(
            [0, M], dtype=torch.int32, device=device
        ),
        "paged_kv_indptr_buffer": torch.zeros(
            2, dtype=torch.int32, device=device
        ),
        "paged_kv_indices_buffer": torch.zeros(
            1, dtype=torch.int32, device=device
        ),
        "paged_kv_last_page_len_buffer": torch.zeros(
            1, dtype=torch.int32, device=device
        ),
    }


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda:0")
    x_t = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    gamma_t = torch.randn(K, dtype=torch.bfloat16, device=device)
    weight_t = torch.randn((N, K), dtype=torch.bfloat16, device=device)
    residual_t = torch.randn((M, N), dtype=torch.bfloat16, device=device)
    h_t = torch.zeros_like(x_t)
    y_t = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    out_t = torch.zeros_like(y_t)

    mpk = PersistentKernel(
        mode="online_notoken",
        world_size=1,
        mpi_rank=0,
        num_workers=248,
        num_local_schedulers=8,
        num_remote_schedulers=0,
        max_seq_length=2,
        max_num_batched_requests=1,
        max_num_batched_tokens=M,
        max_num_pages=1,
        page_size=1,
        meta_tensors=make_meta_tensors(device),
        profiler_tensor=None,
        trace_name=None,
        spec_decode_config=None,
        use_cutlass_kernel=False,
        eos_token_id=-1,
    )

    x = mpk.attach_input(x_t, name="fleet_toy_x")
    gamma = mpk.attach_input(gamma_t, name="fleet_toy_gamma")
    h = mpk.attach_input(h_t, name="fleet_toy_h")
    weight = mpk.attach_input(weight_t, name="fleet_toy_weight")
    y = mpk.attach_input(y_t, name="fleet_toy_y")
    residual = mpk.attach_input(residual_t, name="fleet_toy_residual")
    out = mpk.attach_input(out_t, name="fleet_toy_out")

    mpk.fleet_toy_rms_row_layer(x, gamma, h, M, K)
    mpk.gang_fleet_toy_linear_layer(h, weight, y, M, K, N)
    mpk.fleet_toy_add_row_layer(y, residual, out, M, N)

    artifacts = Path(__file__).resolve().parent / "artifacts"
    mpk.compile(output_dir=str(artifacts))
    # The generated launcher accepts a raw HIP stream handle. Stream zero is
    # sufficient for this isolated example and avoids Python-version-specific
    # torch Stream handle conversion.
    mpk(default_stream=0)
    torch.cuda.synchronize()

    h_ref = (
        x_t.float()
        * torch.rsqrt(x_t.float().square().mean(-1, keepdim=True) + 1.0e-6)
        * gamma_t.float()
    ).to(torch.bfloat16)
    y_ref = (h_ref.float() @ weight_t.float().T).to(torch.bfloat16)
    out_ref = (y_ref.float() + residual_t.float()).to(torch.bfloat16)
    max_abs = (out_t.float() - out_ref.float()).abs().max().item()
    torch.testing.assert_close(out_t, out_ref, rtol=2e-2, atol=2e-2)

    print(
        "fleet_toy: PASS "
        f"max_abs_err={max_abs:.6f} "
        "compute_tasks=4_rms+8_gang_descriptors+4_add "
        "runtime=Fleet_split_worker_scheduler"
    )
    mpk.finalize()


if __name__ == "__main__":
    main()
