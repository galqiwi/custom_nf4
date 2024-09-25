import torch

NF4_CODES = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
    -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434,
    0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=torch.float16)


def get_closest_idx(x, grid):
    _grid_len, = grid.shape
    input_shape = x.shape
    x = x.reshape(-1)

    output = (x[:, None] - grid[None, :]).abs().min(dim=1).indices
    assert output.shape == x.shape

    return output.reshape(input_shape)


def quantize_weight(weight, block_size=64, codes=NF4_CODES):
    out_dim, in_dim = weight.shape

    weight_groups = weight.reshape(-1, block_size)

    scales = weight_groups.abs().max(dim=1).values

    assert scales.shape == (out_dim * in_dim // block_size,)
    weight_quantized = get_closest_idx(
        weight_groups / scales[:, None],
        codes,
    ).reshape(out_dim, in_dim)

    return weight_quantized, scales


def dequantize_weight(weight_quantized, scales, block_size = 64, codes=NF4_CODES):
    out_dim, in_dim = weight_quantized.shape

    return (
        codes[weight_quantized].reshape(-1, block_size) *
        scales[:, None]
    ).reshape(out_dim, in_dim)


def quantize_dequantize_weight(weight, block_size=64, codes=NF4_CODES):
    weight_quantized, scales = quantize_weight(weight, block_size=block_size, codes=codes)
    scales = scales.half()
    return dequantize_weight(weight_quantized, scales, block_size=block_size, codes=codes)


def main():
    codes = NF4_CODES

    weight = torch.randn(1024, 1024 * 2)

    weight_quantized, scales = quantize_weight(weight, codes=codes)
    scales = scales.half()

    output = dequantize_weight(weight_quantized, scales, codes=codes)

    assert 0 <= weight_quantized.min().item()
    assert weight_quantized.max().item() < 16

    print(weight_quantized.shape, weight_quantized.dtype)
    print(scales.shape, scales.dtype)

    print(
        (((output - weight).norm() ** 2) / (weight.norm() ** 2)).item()
    )


if __name__ == '__main__':
    main()
