import torch
import torch.nn as nn

# Define WrappedGPT class


def _unwrap_weights_map(weights_map):
    dataset = weights_map
    prefix = ""
    while hasattr(dataset, "dataset") and hasattr(dataset, "prefix"):
        prefix = f"{dataset.prefix}{prefix}"
        dataset = dataset.dataset
    return dataset, prefix


def get_effective_layer_weight(layer):
    weight = layer.weight.data
    if weight.device.type != "meta":
        return weight
    hf_hook = getattr(layer, "_hf_hook", None)
    weights_map = getattr(hf_hook, "weights_map", None)
    if weights_map is None:
        raise RuntimeError(f"layer {layer.__class__.__name__} has meta weight without an accelerate weights_map")
    try:
        offloaded_weight = weights_map["weight"]
    except Exception as exc:
        raise RuntimeError(f"unable to fetch offloaded weight for layer {layer.__class__.__name__}") from exc
    if not torch.is_tensor(offloaded_weight):
        raise RuntimeError(f"unexpected offloaded weight type for layer {layer.__class__.__name__}: {type(offloaded_weight)}")
    return offloaded_weight


def set_effective_layer_weight(layer, updated_weight):
    updated_weight = updated_weight.detach()
    live_weight = layer.weight.data
    if live_weight.device.type != "meta":
        live_weight.copy_(updated_weight.to(device=live_weight.device, dtype=live_weight.dtype))
    hf_hook = getattr(layer, "_hf_hook", None)
    weights_map = getattr(hf_hook, "weights_map", None)
    if weights_map is None:
        return
    storage, prefix = _unwrap_weights_map(weights_map)
    key = f"{prefix}weight"
    if isinstance(storage, dict):
        stored_weight = storage.get(key)
        target_device = stored_weight.device if torch.is_tensor(stored_weight) else updated_weight.device
        target_dtype = stored_weight.dtype if torch.is_tensor(stored_weight) else updated_weight.dtype
        storage[key] = updated_weight.to(device=target_device, dtype=target_dtype).clone()
        return
    if hasattr(storage, "state_dict") and isinstance(storage.state_dict, dict):
        stored_weight = storage.state_dict.get(key)
        target_device = stored_weight.device if torch.is_tensor(stored_weight) else updated_weight.device
        target_dtype = stored_weight.dtype if torch.is_tensor(stored_weight) else updated_weight.dtype
        storage.state_dict[key] = updated_weight.to(device=target_device, dtype=target_dtype).clone()
        if hasattr(storage, "all_keys") and key not in storage.all_keys:
            storage.all_keys.append(key)
        return
    try:
        stored_weight = weights_map["weight"]
    except Exception:
        stored_weight = None
    if hasattr(weights_map, "__setitem__"):
        target_device = stored_weight.device if torch.is_tensor(stored_weight) else updated_weight.device
        target_dtype = stored_weight.dtype if torch.is_tensor(stored_weight) else updated_weight.dtype
        weights_map["weight"] = updated_weight.to(device=target_device, dtype=target_dtype).clone()
        return
    raise RuntimeError(f"unable to persist offloaded weight update for layer {layer.__class__.__name__}")


def prune_effective_layer_weight(layer, mask):
    weight = get_effective_layer_weight(layer)
    mask = mask.to(device=weight.device)
    updated_weight = weight.clone()
    updated_weight[mask] = 0
    set_effective_layer_weight(layer, updated_weight)


class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.shape[0]
        self.columns = layer.weight.shape[1]

        self.scaler_row = None
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        if self.scaler_row is None:
            self.scaler_row = torch.zeros((self.columns), device=inp.device, dtype=torch.float32)
        elif self.scaler_row.device != inp.device:
            self.scaler_row = self.scaler_row.to(inp.device)
        self.dev = self.scaler_row.device

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.to(device=self.scaler_row.device, dtype=torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples