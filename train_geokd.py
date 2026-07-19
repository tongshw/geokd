import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record

try:
    import wandb
except ImportError:
    wandb = None

from dataset.KITTI import load_test1_data, load_test2_data, load_train_data
from dataset.VIGOR import VIGOR, fetch_dataloader
from model.dino import DINO
from model.geokd import (
    build_localization_model,
    copy_pruned_state,
    dino_input_dims,
    load_checkpoint_state,
)
from model.loss import cross_entropy
from utils.util import get_meter_per_pixel, print_colored, setup_seed


def load_config(config_path):
    suffix = Path(config_path).suffix.lower()
    with open(config_path, "r") as f:
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("YAML configs require PyYAML. Install it with `pip install pyyaml`.") from exc
            return yaml.safe_load(f)
        if suffix == ".json":
            return json.load(f)
    raise ValueError(f"Unsupported config file extension: {suffix}")


def flatten_config(config):
    flat = {}

    def visit(node, prefix=""):
        for key, value in node.items():
            if isinstance(value, dict):
                visit(value, key)
            else:
                flat_key = f"{prefix}_{key}" if prefix == "eval" else key
                flat[flat_key] = value

    visit(config)
    return flat


def namespace_from_config(config_path):
    config = flatten_config(load_config(config_path))

    if "teacher_dino_model" not in config:
        config["teacher_dino_model"] = config.get("dino_model")
    if "student_dino_model" not in config:
        config["student_dino_model"] = config.get("dino_model")
    if "batch_size_per_gpu" not in config:
        config["batch_size_per_gpu"] = config.get("batch_size")
    config["batch_size"] = config.get("batch_size_per_gpu")
    config["dino_model"] = config.get("teacher_dino_model")
    config.setdefault("wandb_log_interval", 20)
    config.setdefault("eval_target", "student")
    config["distill_mode"] = config.get("distill_mode", config.get("mode", "ce"))
    config.setdefault("uncertainty_weighting", False)
    config.setdefault("uncertainty_weight_min", 0.25)
    config.setdefault("uncertainty_weight_power", 1.0)
    config.setdefault("peak_radius", 2)
    config.setdefault("peak_topk", 1)
    config.setdefault("peak_region_weight", 4.0)
    config.setdefault("hard_negative_topk", 32)
    config.setdefault("hard_negative_weight", 1.0)
    return SimpleNamespace(**config)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def count_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())


def maybe_log_wandb(payload):
    if wandb is not None and wandb.run is not None:
        wandb.log(payload)


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if is_dist_avail_and_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    if is_dist_avail_and_initialized():
        return dist.get_world_size()
    return 1


def is_main_process():
    return get_rank() == 0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def setup_distributed(args):
    args.distributed = bool(getattr(args, "distributed", False))
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not args.distributed:
        args.rank = 0
        args.world_size = 1
        return torch.device(f"cuda:{args.gpuid[0]}" if torch.cuda.is_available() else "cpu")

    if args.world_size <= 1:
        raise RuntimeError("runtime.distributed is true, but WORLD_SIZE <= 1. Launch with torchrun.")

    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA, but CUDA is not available.")

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return torch.device(f"cuda:{args.local_rank}")


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def reduce_mean(value, device):
    tensor = torch.tensor(float(value), device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= get_world_size()
    return tensor.item()


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_args(args):
    if args.dataset == "kitti" and getattr(args, "rotation_range", 0) != 0:
        raise ValueError("GeoKD is xy-only. Set rotation_range to 0 for KITTI.")
    if not args.levels:
        raise ValueError("levels must contain at least one DPT output level.")
    if args.eval_target not in {"teacher", "student"}:
        raise ValueError("eval_target must be either 'teacher' or 'student'.")
    dino_input_dims(args.teacher_dino_model)
    dino_input_dims(args.student_dino_model)


def load_teacher(args, device):
    if not args.teacher_ckpt:
        raise ValueError("GeoKD requires a non-null teacher_ckpt in the config.")
    teacher = build_localization_model(
        args,
        dataset=args.dataset,
        dino_model=args.teacher_dino_model,
        student_width=None,
    ).to(device)
    state, checkpoint = load_checkpoint_state(args.teacher_ckpt, map_location=device)
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    epoch = checkpoint.get("epoch")
    loss = checkpoint.get("loss")
    print_colored(f"Loaded fixed teacher: {args.teacher_ckpt}, epoch={epoch}, loss={loss}")
    return teacher, state


def build_student(args, teacher_state, device):
    student = build_localization_model(
        args,
        dataset=args.dataset,
        dino_model=args.student_dino_model,
        student_width=args.student_width,
    ).to(device)

    resume_state = None
    if getattr(args, "student_ckpt", None):
        state, checkpoint = load_checkpoint_state(args.student_ckpt, map_location=device)
        student.load_state_dict(state, strict=True)
        ckpt_width = checkpoint.get("student_width")
        if ckpt_width is not None and abs(float(ckpt_width) - float(args.student_width)) > 1e-8:
            raise ValueError(
                f"student_width mismatch: checkpoint has {ckpt_width}, config has {args.student_width}."
            )
        ckpt_student_dino = checkpoint.get("student_dino_model") or checkpoint.get("dino_model")
        if ckpt_student_dino is not None and ckpt_student_dino != args.student_dino_model:
            raise ValueError(
                f"student_dino_model mismatch: checkpoint has {ckpt_student_dino}, "
                f"config has {args.student_dino_model}."
            )
        print_colored(f"Loaded student checkpoint: {args.student_ckpt}, epoch={checkpoint.get('epoch')}")
        resume_state = checkpoint
    elif getattr(args, "init_from_teacher", True):
        skip_input_projection = args.teacher_dino_model != args.student_dino_model
        copied, skipped = copy_pruned_state(
            student,
            teacher_state,
            skip_input_projection=skip_input_projection,
        )
        if skip_input_projection:
            print_colored("Skipped DPT input projections because teacher/student DINO backbones differ.")
        print_colored(f"Initialized student from teacher slices: copied={len(copied)}, skipped={len(skipped)}")

    return student, resume_state


def build_eval_model(args, device):
    if args.eval_target == "teacher":
        model, _ = load_teacher(args, device)
        dino = DINO(model_name=args.teacher_dino_model).to(device)
        dino.eval()
        for param in dino.parameters():
            param.requires_grad = False
        return model, dino

    if not args.student_ckpt:
        raise ValueError("Evaluating student requires model.student_ckpt in the config.")
    model = build_localization_model(
        args,
        dataset=args.dataset,
        dino_model=args.student_dino_model,
        student_width=args.student_width,
    ).to(device)
    state, checkpoint = load_checkpoint_state(args.student_ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    dino = DINO(model_name=args.student_dino_model).to(device)
    dino.eval()
    for param in dino.parameters():
        param.requires_grad = False
    if is_main_process():
        print_colored(f"Loaded eval student: {args.student_ckpt}, epoch={checkpoint.get('epoch')}")
    return model, dino


def get_vigor_loaders(args):
    if args.train:
        vigor = VIGOR(args, "train")
        return fetch_dataloader(args, vigor)
    return fetch_dataloader(args, split="test")


def get_kitti_loaders(args):
    load_test1_data.dataset_root = getattr(args, "dataset_root", None) or "/data/dataset/KITTI"
    load_test1_data.test1_file = getattr(args, "test1_file", None) or "/data/test/code/HighlyAccurate/dataLoader/test1_files.txt"
    load_test1_data.distributed = getattr(args, "distributed", False)
    load_test1_data.num_workers = getattr(args, "num_workers", 8)
    load_test2_data.dataset_root = getattr(args, "dataset_root", None) or "/data/dataset/KITTI"
    load_test2_data.test2_file = getattr(args, "test2_file", None) or "/data/test/code/HighlyAccurate/dataLoader/test2_files.txt"
    load_test2_data.distributed = getattr(args, "distributed", False)
    load_test2_data.num_workers = getattr(args, "num_workers", 8)
    if args.train:
        train_loader = load_train_data(
            args,
            args.batch_size,
            args.shift_range_lat,
            args.shift_range_lon,
            args.rotation_range,
        )
        val_loader = load_test1_data(
            args.batch_size,
            args.shift_range_lat,
            args.shift_range_lon,
            args.rotation_range,
        )
        return train_loader, val_loader
    if args.cross_area:
        return load_test2_data(args.batch_size, args.shift_range_lat, args.shift_range_lon, args.rotation_range)
    return load_test1_data(args.batch_size, args.shift_range_lat, args.shift_range_lon, args.rotation_range)


def forward_vigor(args, dino, model, batch, device):
    batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
    _, sat, _, _, _, sat_delta, meter_per_pixel, resized_pano, _, _ = batch

    sat_img = 2 * (sat / 255.0) - 1.0
    pano_img = 2 * (resized_pano / 255.0) - 1.0
    pano_img = pano_img.contiguous().permute(0, 3, 1, 2)

    sat_feat_list = dino(sat_img.contiguous())
    pano_feat_list = dino(pano_img)

    sat_feat, sat_conf, g2s_feat, g2s_conf, _, _ = model(
        sat_feat_list,
        pano_feat_list,
        meter_per_pixel,
    )
    return {
        "sat_feat": sat_feat,
        "sat_conf": sat_conf,
        "g2s_feat": g2s_feat,
        "g2s_conf": g2s_conf,
        "sat_delta": sat_delta,
        "meter_per_pixel": meter_per_pixel,
    }


def forward_kitti(args, dino, model, batch, device):
    items = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
    _, sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading = items[:7]

    sat_feat_list = dino(sat_map)
    grd_feat_list = dino(grd_left_imgs)
    sat_feat, g2s_feat = model(sat_feat_list, grd_feat_list, left_camera_k)
    return {
        "sat_feat": sat_feat,
        "g2s_feat": g2s_feat,
        "gt_shift_u": gt_shift_u,
        "gt_shift_v": gt_shift_v,
        "gt_heading": gt_heading,
    }


def train_epoch(args, teacher_dino, student_dino, teacher, student, train_loader, optimizer, device, epoch):
    student.train()
    teacher.eval()
    total_loss = 0.0
    if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)
    pbar = tqdm(train_loader, desc=f"Train {epoch}", disable=not is_main_process())
    global_step_offset = epoch * max(len(train_loader), 1)
    student_model = unwrap_model(student)
    teacher_model = unwrap_model(teacher)

    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            if args.dataset == "vigor":
                teacher_out = forward_vigor(args, teacher_dino, teacher, batch, device)
                teacher_corr = teacher_model.calc_corr_for_train(
                    teacher_out["sat_feat"],
                    teacher_out["g2s_feat"],
                    batch_wise=False,
                )
            else:
                teacher_out = forward_kitti(args, teacher_dino, teacher, batch, device)
                teacher_corr = teacher_model.calc_corr_for_train(
                    teacher_out["sat_feat"],
                    teacher_out["g2s_feat"],
                    batch_wise=False,
                )

        if args.dataset == "vigor":
            student_out = forward_vigor(args, student_dino, student, batch, device)
            student_corr = student_model.calc_corr_for_train(
                student_out["sat_feat"],
                student_out["g2s_feat"],
                batch_wise=False,
            )
        else:
            student_out = forward_kitti(args, student_dino, student, batch, device)
            student_corr = student_model.calc_corr_for_train(
                student_out["sat_feat"],
                student_out["g2s_feat"],
                batch_wise=False,
            )

        loss, loss_stats = cross_entropy(
            student_corr,
            teacher_corr,
            args.levels,
            s_temp=args.student_temp,
            t_temp=args.teacher_temp,
            distill_mode=args.distill_mode,
            uncertainty_weighting=args.uncertainty_weighting,
            uncertainty_weight_min=args.uncertainty_weight_min,
            uncertainty_weight_power=args.uncertainty_weight_power,
            peak_radius=args.peak_radius,
            peak_topk=args.peak_topk,
            peak_region_weight=args.peak_region_weight,
            hard_negative_topk=args.hard_negative_topk,
            hard_negative_weight=args.hard_negative_weight,
            return_stats=True,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.clip)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss / (pbar.n + 1):.4f}")

        if args.wandb and is_main_process() and batch_idx % args.wandb_log_interval == 0:
            log_payload = {
                "train/step_loss": loss.item(),
                "train/epoch": epoch + 1,
                "train/step": global_step_offset + batch_idx,
            }
            log_payload.update(loss_stats)
            maybe_log_wandb(log_payload)

    local_mean_loss = total_loss / max(len(train_loader), 1)
    return reduce_mean(local_mean_loss, device)


def validate_vigor(args, dino, model, val_loader, device):
    model.eval()
    pred_us, pred_vs, gt_us, gt_vs = [], [], [], []
    local_samples = 0
    local_infer_time = 0.0
    model_ref = unwrap_model(model)
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validate", disable=not is_main_process()):
            sync_device(device)
            start_time = time.perf_counter()
            out = forward_vigor(args, dino, model, batch, device)
            corr = model_ref.calc_corr_for_val(
                out["sat_feat"],
                out["sat_conf"],
                out["g2s_feat"],
                out["g2s_conf"],
            )
            b, corr_h, corr_w = corr.shape
            max_index = torch.argmax(corr.reshape(b, -1), dim=1)
            pred_u = (max_index % corr_w - corr_w / 2)
            pred_v = (max_index // corr_w - corr_h / 2)
            sync_device(device)
            local_infer_time += time.perf_counter() - start_time
            local_samples += b
            _, _, feat_h, _ = out["sat_feat"][args.levels[-1]].shape
            pred_u = pred_u * 512 / feat_h * out["meter_per_pixel"]
            pred_v = pred_v * 512 / feat_h * out["meter_per_pixel"]

            gt_shift_u = out["sat_delta"][:, 0] * out["meter_per_pixel"] * 512 / 4
            gt_shift_v = out["sat_delta"][:, 1] * out["meter_per_pixel"] * 512 / 4
            pred_us.append(pred_u.cpu().numpy())
            pred_vs.append(pred_v.cpu().numpy())
            gt_us.append(gt_shift_u.cpu().numpy())
            gt_vs.append(gt_shift_v.cpu().numpy())

    mean_error, median_error = xy_metrics(pred_us, pred_vs, gt_us, gt_vs, device)
    fps = infer_fps(local_samples, local_infer_time, device)
    return mean_error, median_error, fps


def validate_kitti(args, dino, model, val_loader, device):
    model.eval()
    pred_us, pred_vs, gt_us, gt_vs = [], [], [], []
    meter_per_pixel = get_meter_per_pixel() * 128 / 152
    local_samples = 0
    local_infer_time = 0.0
    model_ref = unwrap_model(model)
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validate", disable=not is_main_process()):
            sync_device(device)
            start_time = time.perf_counter()
            out = forward_kitti(args, dino, model, batch, device)
            corr = model_ref.calc_corr_for_val(out["sat_feat"], out["g2s_feat"])
            b, corr_h, corr_w = corr.shape
            max_index = torch.argmax(corr.reshape(b, -1), dim=1)
            pred_u = (max_index % corr_w - corr_w / 2 + 0.5) * meter_per_pixel
            pred_v = -(max_index // corr_w - corr_h / 2 + 0.5) * meter_per_pixel
            sync_device(device)
            local_infer_time += time.perf_counter() - start_time
            local_samples += b

            gt_us.append((out["gt_shift_u"][:, 0] * args.shift_range_lon).cpu().numpy())
            gt_vs.append((out["gt_shift_v"][:, 0] * args.shift_range_lat).cpu().numpy())
            pred_us.append(pred_u.cpu().numpy())
            pred_vs.append(pred_v.cpu().numpy())

    mean_error, median_error = xy_metrics(pred_us, pred_vs, gt_us, gt_vs, device)
    fps = infer_fps(local_samples, local_infer_time, device)
    return mean_error, median_error, fps


def infer_fps(local_samples, local_seconds, device):
    stats = torch.tensor([float(local_samples), float(local_seconds)], device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    if stats[1].item() <= 0:
        return 0.0
    return float((stats[0] / stats[1]).item())


def gather_variable_length_array(local_array, device):
    tensor = torch.tensor(local_array, device=device, dtype=torch.float32)
    local_len = torch.tensor([tensor.shape[0]], device=device, dtype=torch.long)
    if not is_dist_avail_and_initialized():
        return tensor.cpu().numpy()

    world_size = get_world_size()
    lengths = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(lengths, local_len)
    lengths = [int(length.item()) for length in lengths]
    max_len = max(lengths)

    if tensor.shape[0] < max_len:
        pad = torch.zeros(max_len - tensor.shape[0], device=device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=0)

    gathered = [torch.zeros(max_len, device=device, dtype=tensor.dtype) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)

    chunks = [buf[:length].cpu().numpy() for buf, length in zip(gathered, lengths)]
    return np.concatenate(chunks, axis=0)


def xy_metrics(pred_us, pred_vs, gt_us, gt_vs, device):
    pred_us = np.concatenate(pred_us, axis=0)
    pred_vs = np.concatenate(pred_vs, axis=0)
    gt_us = np.concatenate(gt_us, axis=0)
    gt_vs = np.concatenate(gt_vs, axis=0)
    pred_us = gather_variable_length_array(pred_us, device)
    pred_vs = gather_variable_length_array(pred_vs, device)
    gt_us = gather_variable_length_array(gt_us, device)
    gt_vs = gather_variable_length_array(gt_vs, device)
    distance = np.sqrt((pred_us - gt_us) ** 2 + (pred_vs - gt_vs) ** 2)
    return float(np.mean(distance)), float(np.median(distance))


def validate(args, dino, model, val_loader, device):
    if args.dataset == "vigor":
        return validate_vigor(args, dino, model, val_loader, device)
    return validate_kitti(args, dino, model, val_loader, device)


def checkpoint_dir(args):
    return os.path.join(args.save_path, args.dataset, "geokd", "student")


def save_checkpoint(args, student, optimizer, scheduler, epoch, mean_error, is_best=False):
    output_dir = os.path.join(args.save_path, args.dataset, "geokd", "student")
    ensure_dir(output_dir)
    filename = f"{args.name}_best.pth" if is_best else "last.pth"
    path = os.path.join(output_dir, filename)
    student_ref = unwrap_model(student)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": student_ref.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_schedule": scheduler.state_dict(),
            "loss": mean_error,
            "student_width": args.student_width,
            "teacher_dino_model": args.teacher_dino_model,
            "student_dino_model": args.student_dino_model,
            "dino_model": args.teacher_dino_model,
            "teacher_ckpt": args.teacher_ckpt,
            "batch_size_per_gpu": args.batch_size_per_gpu,
            "world_size": get_world_size(),
            "model_name": args.model_name,
            "experiment_name": args.name,
            "distill_mode": args.distill_mode,
        },
        path,
    )
    return path


def run(args):
    if args.wandb and wandb is None:
        raise ImportError("wandb is enabled but the wandb package is not installed.")
    validate_args(args)
    device = setup_distributed(args)
    setup_seed(args.seed + get_rank())
    ensure_dir(args.save_path)

    if not args.train:
        eval_model, eval_dino = build_eval_model(args, device)
        test_loader = get_vigor_loaders(args) if args.dataset == "vigor" else get_kitti_loaders(args)
        mean_error, median_error, fps = validate(args, eval_dino, eval_model, test_loader, device)
        if args.wandb and is_main_process():
            maybe_log_wandb(
                {
                    f"test/{args.eval_target}/mean_error": mean_error,
                    f"test/{args.eval_target}/median_error": median_error,
                    f"test/{args.eval_target}/fps": fps,
                }
            )
        if is_main_process():
            print(f"Eval target: {args.eval_target}")
            print(f"Test mean distance: {mean_error:.4f}")
            print(f"Test median distance: {median_error:.4f}")
            print(f"Test FPS: {fps:.2f}")
        return

    teacher, teacher_state = load_teacher(args, device)
    student, resume_state = build_student(args, teacher_state, device)
    if args.distributed:
        student = DDP(student, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    if is_main_process():
        print_colored(
            f"Teacher params={count_parameters(unwrap_model(teacher)) / 1e6:.2f}M, "
            f"student params={count_parameters(unwrap_model(student)) / 1e6:.2f}M, "
            f"student trainable={count_trainable_parameters(unwrap_model(student)) / 1e6:.2f}M"
        )
    teacher_params_m = count_parameters(unwrap_model(teacher)) / 1e6
    student_params_m = count_parameters(unwrap_model(student)) / 1e6
    student_trainable_params_m = count_trainable_parameters(unwrap_model(student)) / 1e6
    teacher_dino = DINO(model_name=args.teacher_dino_model).to(device)
    teacher_dino.eval()
    for param in teacher_dino.parameters():
        param.requires_grad = False

    student_dino = DINO(model_name=args.student_dino_model).to(device)
    student_dino.eval()
    for param in student_dino.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    start_epoch = 0
    best_mean = float("inf")
    if args.train and resume_state is not None:
        if "optimizer_state_dict" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if "lr_schedule" in resume_state:
            scheduler.load_state_dict(resume_state["lr_schedule"])
        start_epoch = int(resume_state.get("epoch", -1)) + 1
        best_mean = float(resume_state.get("loss", float("inf")))
        if is_main_process():
            print_colored(
                f"Resumed training from epoch {start_epoch} with best/last mean error {best_mean:.4f}"
            )

    if args.wandb and is_main_process():
        if wandb.run is not None:
            wandb.run.summary["model/teacher_params_m"] = teacher_params_m
            wandb.run.summary["model/student_params_m"] = student_params_m
            wandb.run.summary["model/student_trainable_params_m"] = student_trainable_params_m
            wandb.run.summary["model/teacher_dino_model"] = args.teacher_dino_model
            wandb.run.summary["model/student_dino_model"] = args.student_dino_model
            wandb.run.summary["model/student_width"] = args.student_width
            wandb.run.summary["runtime/world_size"] = get_world_size()
            wandb.run.summary["train/batch_size_per_gpu"] = args.batch_size_per_gpu
            wandb.run.summary["train/global_batch_size"] = args.batch_size_per_gpu * get_world_size()
            wandb.run.summary["distill/mode"] = args.distill_mode
            wandb.run.summary["distill/uncertainty_weighting"] = args.uncertainty_weighting
            wandb.run.summary["distill/uncertainty_weight_min"] = args.uncertainty_weight_min
            wandb.run.summary["distill/uncertainty_weight_power"] = args.uncertainty_weight_power
            wandb.run.summary["distill/peak_radius"] = args.peak_radius
            wandb.run.summary["distill/peak_topk"] = args.peak_topk
            wandb.run.summary["distill/peak_region_weight"] = args.peak_region_weight
            wandb.run.summary["distill/hard_negative_topk"] = args.hard_negative_topk
            wandb.run.summary["distill/hard_negative_weight"] = args.hard_negative_weight
            wandb.run.summary["last_checkpoint"] = os.path.join(checkpoint_dir(args), "last.pth")

    if args.train:
        train_loader, val_loader = get_vigor_loaders(args) if args.dataset == "vigor" else get_kitti_loaders(args)
        for epoch in range(start_epoch, args.epochs):
            train_loss = train_epoch(
                args, teacher_dino, student_dino, teacher, student, train_loader, optimizer, device, epoch
            )
            mean_error, median_error, fps = validate(args, student_dino, student, val_loader, device)
            scheduler.step()
            if args.wandb and is_main_process():
                maybe_log_wandb(
                    {
                        "train/epoch_loss": train_loss,
                        "train/epoch": epoch + 1,
                        "train/batch_size_per_gpu": args.batch_size_per_gpu,
                        "train/global_batch_size": args.batch_size_per_gpu * get_world_size(),
                        "val/mean_error": mean_error,
                        "val/median_error": median_error,
                        "val/best_mean_error": min(best_mean, mean_error),
                        "val/fps": fps,
                        "optim/lr": scheduler.get_last_lr()[0],
                    }
                )
            if is_main_process():
                last_path = save_checkpoint(args, student, optimizer, scheduler, epoch, mean_error, is_best=False)
            if mean_error < best_mean:
                best_mean = mean_error
                if is_main_process():
                    path = save_checkpoint(args, student, optimizer, scheduler, epoch, best_mean, is_best=True)
                    print_colored(f"Saved best student to {path}")
                    if args.wandb and wandb.run is not None:
                        wandb.run.summary["best_checkpoint"] = path
                        wandb.run.summary["best_val_mean_error"] = best_mean
                        wandb.run.summary["last_checkpoint"] = os.path.join(checkpoint_dir(args), "last.pth")
            if is_main_process():
                print(
                    f"Epoch {epoch + 1}/{args.epochs}: "
                    f"train_loss={train_loss:.4f}, mean={mean_error:.4f}, "
                    f"median={median_error:.4f}, fps={fps:.2f}"
                )


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate GeoKD from a YAML config.")
    parser.add_argument("--config", default="dataset/config_vigor_kd.yaml", help="Path to a GeoKD YAML config.")
    cli_args = parser.parse_args()
    return namespace_from_config(cli_args.config)


@record
def main():
    args = parse_args()
    args.model_name = f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    should_init_wandb = args.wandb and int(os.environ.get("RANK", "0")) == 0
    if should_init_wandb:
        if wandb is None:
            raise ImportError("wandb is enabled but the wandb package is not installed.")
        wandb.init(project="geokd", name=args.model_name, config=vars(args))
    if int(os.environ.get("RANK", "0")) == 0:
        print(args)
    try:
        run(args)
    finally:
        if should_init_wandb:
            wandb.finish()
        cleanup_distributed()


if __name__ == "__main__":
    main()
