import triton

import triton.language as tl


@triton.jit
def layer_norm_kernel(x, w, b, y, Mean, Rstd, eps, stride_x, stride_y, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)

    x = x + pid * stride_x
    y = y + pid * stride_y

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for col_idx in range(0, N, BLOCK_SIZE):
        offset = col_idx + tl.arange(0, BLOCK_SIZE)
        a = tl.load(x + offset, mask=offset < N, other=0.0).to(tl.float32)

        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _variance = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for col_idx in range(0, N, BLOCK_SIZE):
        offset = col_idx + tl.arange(0, BLOCK_SIZE)
        a = tl.load(x + offset, mask=offset < N, other=0.0).to(tl.float32)

        _variance += tl.math.pow(a, 2)
    variance = tl.sum(_variance, axis=0) / N - mean

    rstd = 1 / tl.sqrt(variance + eps)
    # Write mean / rstd
    tl.store(Mean + pid, mean)
    tl.store(Rstd + pid, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        _w = tl.load(w + cols, mask=mask)
        _b = tl.load(b + cols, mask=mask)
        _x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (_x - mean) * rstd
        _y = x_hat * _w + _b
        # Write output
        tl.store(y + cols, _y, mask=mask)
