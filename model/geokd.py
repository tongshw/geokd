import torch

from model.dpt import DPT
from model.network_kitti_dino import LocalizationNet as KittiLocalizationNet
from model.network_vigor import LocalizationNet as VigorLocalizationNet
from model.slim_dpt import SlimDPT


DINOV2_DENSE_CLS_DIMS = {
    "vits14": 768,
    "vitb14": 1536,
    "vitl14": 2048,
    "vitg14": 3072,
}


def dino_input_dims(model_name):
    if model_name not in DINOV2_DENSE_CLS_DIMS:
        raise ValueError(f"Unsupported DINOv2 model '{model_name}'.")
    return [DINOV2_DENSE_CLS_DIMS[model_name]] * 4


def validate_student_width(student_width):
    if student_width is None:
        return
    if student_width <= 0:
        raise ValueError(f"student_width must be positive, got {student_width}.")


class KittiGeoKDNet(KittiLocalizationNet):
    def __init__(self, args, dino_model="vitl14", student_width=None):
        super().__init__(args)
        validate_student_width(student_width)
        input_dims = dino_input_dims(dino_model)
        if student_width is None or student_width >= 1:
            self.SatDPT = DPT(input_dims=input_dims)
            self.GrdDPT = DPT(input_dims=input_dims)
        else:
            self.SatDPT = SlimDPT(input_dims=input_dims, width_mult=student_width)
            self.GrdDPT = SlimDPT(input_dims=input_dims, width_mult=student_width)


class VigorGeoKDNet(VigorLocalizationNet):
    def __init__(self, args, dino_model="vitb14", student_width=None):
        super().__init__(args)
        validate_student_width(student_width)
        input_dims = dino_input_dims(dino_model)
        if student_width is None or student_width >= 1:
            self.SatDPT = DPT(input_dims=input_dims)
            self.GrdDPT = DPT(input_dims=input_dims)
        else:
            self.SatDPT = SlimDPT(input_dims=input_dims, width_mult=student_width)
            self.GrdDPT = SlimDPT(input_dims=input_dims, width_mult=student_width)


def build_localization_model(args, dataset, dino_model, student_width=None):
    dataset = dataset.lower()
    if dataset == "vigor":
        return VigorGeoKDNet(args, dino_model=dino_model, student_width=student_width)
    if dataset == "kitti":
        return KittiGeoKDNet(args, dino_model=dino_model, student_width=student_width)
    raise ValueError(f"Unsupported dataset '{dataset}'. Expected 'vigor' or 'kitti'.")


def load_checkpoint_state(path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
        checkpoint = {"epoch": None, "loss": None}

    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint at {path} does not contain a valid state dict.")

    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}

    return state, checkpoint


def copy_pruned_state(student, teacher_state, skip_input_projection=False):
    """Copy overlapping tensor slices from a full teacher state into a slimmer student."""
    student_state = student.state_dict()
    copied = []
    skipped = []
    new_state = {}
    for key, student_tensor in student_state.items():
        teacher_tensor = teacher_state.get(key)
        if skip_input_projection and ".conv_" in key:
            new_state[key] = student_tensor
            skipped.append(key)
            continue
        if teacher_tensor is None or teacher_tensor.ndim != student_tensor.ndim:
            new_state[key] = student_tensor
            skipped.append(key)
            continue

        slices = tuple(slice(0, min(s_dim, t_dim)) for s_dim, t_dim in zip(student_tensor.shape, teacher_tensor.shape))
        updated = student_tensor.clone()
        updated[slices] = teacher_tensor[slices].to(dtype=student_tensor.dtype)
        new_state[key] = updated
        copied.append(key)

    student.load_state_dict(new_state, strict=True)
    return copied, skipped
