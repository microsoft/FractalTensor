import context

from examples.convolution.utils import *

ctx = kaleido.Context()


@kaleido.params(ctx)
class ModelParams(NamedTuple):
    conv_filter: FractalTensor[FractalTensor[Tensor['7, 7', float, 'cpu']]]
    bn_scale: Tensor['64, 1', float, 'cpu']
    bn_bias: Tensor['64, 1', float, 'cpu']


# @kaleido.function(ctx)
def conv_func_y(
        xs: FractalTensor[Tensor['3, 32', float, 'cpu']],
        w: Tensor['7, 7, 3', float, 'cpu']) -> Tensor['1, 17', float, 'cpu']:
    patches = ops.map(
        lambda x: ops.stack(x),
        ops.slide(
            ops.slices(ops.stack(xs), dim=2),
            window_size=7,
            stride=2,
            padding=6,
            dilation=2,
            padding_value=ops.zeros(shape=(7, 3), device='cpu')))
    conv_y = ops.flatten(ops.map(lambda patch: ops.dot(patch, w), patches))
    return conv_y


# @kaleido.function(ctx)
def conv_func_x(
        x: Tensor['3, 32, 32', float, 'cpu'],
        w: Tensor['3, 7, 7', float, 'cpu']) -> Tensor['17, 17', float, 'cpu']:
    """x is a single image, and w is a single filter."""
    xs = ops.slide(
        ops.slices(x, dim=1),
        window_size=7,
        stride=2,
        padding=6,
        dilation=2,
        padding_value=ops.zeros(shape=(3, 32), device='cpu'))
    conv_x = ops.stack(ops.map(lambda patch: conv_func_y(patch, w), xs), dim=0)
    return conv_x


# @kaleido.function(ctx)
def var_func(
        s: Tensor['1, 20', 'cpu', float],
        xs: FractalTensor[Tensor['13, 13', 'cpu', float]],
        mean: Tensor['1, 20', 'cpu', float]) -> Tensor['1, 20', 'cpu', float]:
    # variance over channel
    v = ops.map(lambda x: ops.sum(ops.pow(ops.sub(*x), 2)),
                ops.zip(xs, ops.slices(mean, dim=1)))
    v = ops.flatten(v)
    v = s + v
    return v


# @kaleido.function(ctx)
def norm_func(x: Tensor['13, 13', 'cpu', float],
              mean: Tensor['1,', 'cpu', float],
              var: Tensor['1,', 'cpu', float],
              s: Tensor['1,', 'cpu', float],
              b: Tensor['1,', 'cpu', float],
              epsilon: float = 1e-6) -> Tensor['13, 13', 'cpu', float]:
    v = s * (x - mean) / ops.sqrt(var + epsilon) + b
    return v


# @kaleido.function(ctx)
def channel_itr(xs: FractalTensor[Tensor['13, 13', 'cpu', float]],
                mean: Tensor['1, 20', 'cpu', float],
                var: Tensor['1, 20', 'cpu', float],
                scale: Tensor['1, 20', 'cpu', float],
                bias: Tensor['1, 20', 'cpu', float]):
    normed = ops.map(
        lambda x: norm_func(*x),
        ops.zip(xs, ops.slices(mean, dim=1), ops.slices(var, dim=1),
                ops.slices(scale, dim=1), ops.slices(bias, dim=1)))
    return normed


# @kaleido.function(ctx)
def batch_norm(
        xss: FractalTensor[FractalTensor[Tensor['17, 17', 'cpu', float]]],
        scale: Tensor['11, 1', 'cpu', float],
        bias: Tensor['11, 1', 'cpu', float]):
    numel = xss[0].length * xss.element_type.numel  # a constant

    # NOTE(ying): Accumulate the moving average of mean and variance is not
    # implemented, which is an addition over a window of some mini-batches.
    # Moving average of the statistics do not affact our analysis of parallel
    # pattern and access pattern.

    # the layout of xss is: [N, C, H, W]
    # mean over [:, i, :, :]
    # training requres this calculation.
    mean = ops.reduce(
        lambda s, xs: s + ops.stack(ops.map(lambda x: ops.sum(x), xs)),
        xss,
        initializer=ops.zeros(shape=(1, 11), device='cpu')) / numel

    # var over [:, i, :, :]
    var = ops.reduce(
        lambda s, xs: var_func(s, xs, mean),
        xss,
        initializer=ops.zeros(shape=(1, 11), device='cpu')) / numel

    normed = ops.map(lambda xs: channel_itr(xs, mean, var, scale, bias), xss)
    return normed


# @kaleido.function(ctx)
def conv_net(
        xs: FractalTensor[Tensor['3, 32, 32', float, 'cpu']],
        params: ModelParams) -> FractalTensor[Tensor['32, 32', float, 'cpu']]:
    conv1 = ops.map(
        lambda x: ops.map(lambda w: conv_func_x(x, w), params.conv_filter), xs)
    # depth-1 of conv1 is the batch size.
    # depth-2 of conv1 is the output channel.

    norm = batch_norm(conv1, params.bn_scale, params.bn_bias)
    return norm


if __name__ == '__main__':
    device = 'cpu'
    out_channel = 11
    batch_size = 3
    img_size = [3, 32, 32]  # image size in cifar.

    params = ModelParams(
        conv_filter=gen_image_batch([7, 7, 3], out_channel),
        bn_scale=ops.rand(
            shape=(1, out_channel), dtype=kaleido.float32, device=device),
        bn_bias=ops.rand(
            shape=(1, out_channel), dtype=kaleido.float32, device=device))

    batch_img = gen_image_batch(img_size, batch_size)
    out = conv_net(batch_img, params)
