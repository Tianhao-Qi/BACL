import torch
import torch.distributed as dist

def get_tensor_length(tensor, size):
    length_tensor = tensor.new_tensor([len(tensor)])
    length_list = [tensor.new_tensor([0]) for _ in range(size)]
    dist.all_gather(length_list, length_tensor)

    lengths = torch.cat(length_list)
    max_length = int(lengths.max().cpu().numpy())
    current_length = len(tensor)

    return max_length, current_length

def collect_tensor_from_dist(tensor_list, fill_numbers):
    size = dist.get_world_size()
    rank = dist.get_rank()

    if not isinstance(tensor_list, list):
        tensor_list = [tensor_list]
    max_length, current_length = get_tensor_length(tensor_list[0], size)

    tensor_collection = []
    for tensor, number in zip(tensor_list, fill_numbers):
        shape = list(tensor.shape)
        shape[0] = max_length
        gather_tensors = [tensor.new_zeros(size=shape) for _ in range(size)]

        shape = list(tensor.shape)
        shape[0] = max_length - shape[0]
        fill_tensor = torch.cat([tensor, tensor.new_ones(size=shape) * number])
        dist.all_gather(gather_tensors, fill_tensor)
        gather_tensors = torch.cat(gather_tensors)
        tensor_collection.append(gather_tensors)

    return tensor_collection
