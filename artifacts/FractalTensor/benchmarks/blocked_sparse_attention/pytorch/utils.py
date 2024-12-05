import torch
from torch import Tensor


def assert_rank(expected_rank):
    expected_rank_dict = {}
    if isinstance(expected_rank, int):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True


def get_shape_list(tensor, expected_rank=None):
    if expected_rank is not None:
        assert_rank(expected_rank)

    shape = tensor.size()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    assert False, "Static shape not available for {}".format(tensor)

    dyn_shape = tensor.size()
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
    exp_blocked_to_pad = torch.cat(
        (to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2],
         to_blocked_mask[:, 3:-1]), 2)
    band_mask = torch.einsum("blq,blk->blqk",
                             from_blocked_mask[:, 2:-2].float(),
                             exp_blocked_to_pad.float())
    band_mask = torch.unsqueeze(band_mask, 1)
    return band_mask


def torch_gather4d(input_tensor: Tensor, indexes: Tensor) -> Tensor:
    # input_tensor = torch.from_numpy(input_tensor)
    # indexes = torch.from_numpy(indexes).long()

    indexes = indexes.long()  # [16, 1, 14, 3]
    indexes = torch.unsqueeze(indexes, -1)  # [16, 1, 14, 3, 1]
    indexes = torch.unsqueeze(indexes, -1)  # [16, 1, 14, 3, 1, 1]
    indexes = indexes.expand(-1, -1, -1, -1, -1, input_tensor.size(-1)).to(
        input_tensor.device)  # [16, 1, 14, 3, 1, 64]

    input_tensor = torch.unsqueeze(input_tensor, 1)  # [16, 1, 16, 64]
    input_tensor = torch.unsqueeze(input_tensor, 1)  # [16, 1, 1, 16, 64]
    input_tensor = torch.unsqueeze(input_tensor, 1)  # [16, 1, 1, 1, 16, 64]
    input_tensor = input_tensor.expand(-1, indexes.size(1), indexes.size(2),
                                       indexes.size(3), -1,
                                       -1)  # [16, 1, 14, 3, 16, 64]
    output_tensor = torch.gather(input_tensor, 4,
                                 indexes)  # [16, 1, 14, 3, 1, 64]
    output_tensor = output_tensor.view(
        (indexes.size(0), indexes.size(1), indexes.size(2), indexes.size(3),
         input_tensor.size(-1)))  # [16, 1, 14, 3, 64]

    return output_tensor


def torch_gather5d(input_tensor: Tensor, indexes: Tensor) -> Tensor:
    # input_tensor = torch.from_numpy(input_tensor)
    # indexes = torch.from_numpy(indexes).long()

    indexes = indexes.long()
    indexes = torch.unsqueeze(indexes, -1)
    indexes = torch.unsqueeze(indexes, -1)
    indexes = indexes.expand(-1, -1, -1, -1, input_tensor.size(-2),
                             input_tensor.size(-1)).to(input_tensor.device)

    input_tensor = torch.unsqueeze(input_tensor, 2)
    input_tensor = input_tensor.expand(-1, -1, indexes.size(2), -1, -1, -1)

    output_tensor = torch.gather(input_tensor, 3, indexes)

    return output_tensor
