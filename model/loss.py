import math

import torch
import torch.nn.functional as F
import numpy as np



def cross_entropy(
    pred_dict,
    target_dict,
    levels,
    s_temp=0.2,
    t_temp=0.09,
    distill_mode="ce",
    uncertainty_weighting=False,
    uncertainty_weight_min=0.25,
    uncertainty_weight_power=1.0,
    peak_radius=2,
    peak_topk=1,
    peak_region_weight=4.0,
    hard_negative_topk=32,
    hard_negative_weight=1.0,
    return_stats=False,
):
    if distill_mode == "peak_hard":
        return peak_hard_negative_distillation(
            pred_dict,
            target_dict,
            levels,
            s_temp=s_temp,
            t_temp=t_temp,
            peak_radius=peak_radius,
            peak_topk=peak_topk,
            peak_region_weight=peak_region_weight,
            hard_negative_topk=hard_negative_topk,
            hard_negative_weight=hard_negative_weight,
            return_stats=return_stats,
        )
    if distill_mode != "ce":
        raise ValueError(f"Unsupported distill_mode '{distill_mode}'.")

    ce_losses = []
    entropy_values = []
    confidence_values = []
    weight_values = []
    for _, level in enumerate(levels):
        pred = pred_dict[level]
        target = target_dict[level]
        pred = -(pred - 2) / 2
        target = -(target - 2) / 2

        b, h, w = pred.shape

        pred_map_flat = pred.reshape(b, -1)
        target_map_flat = target.reshape(b, -1)

        pred_map_log_softmax = F.log_softmax(pred_map_flat / s_temp, dim=1)
        target_map_softmax = F.softmax(target_map_flat / t_temp, dim=1)

        per_sample_loss = -torch.sum(target_map_softmax * pred_map_log_softmax, dim=1)
        if uncertainty_weighting:
            entropy = -torch.sum(target_map_softmax * torch.log(target_map_softmax.clamp_min(1e-12)), dim=1)
            norm_entropy = entropy / math.log(target_map_softmax.shape[1])
            confidence = (1.0 - norm_entropy).clamp(0.0, 1.0)
            weight = uncertainty_weight_min + (1.0 - uncertainty_weight_min) * confidence.pow(uncertainty_weight_power)
            loss = torch.sum(per_sample_loss * weight) / weight.sum().clamp_min(1e-6)

            entropy_values.append(norm_entropy.detach().mean())
            confidence_values.append(confidence.detach().mean())
            weight_values.append(weight.detach().mean())
        else:
            loss = torch.sum(per_sample_loss) / b
        ce_losses.append(loss)

    loss = torch.mean(torch.stack(ce_losses, dim=0).float())
    if not return_stats:
        return loss

    stats = {}
    if uncertainty_weighting:
        stats = {
            "distill/teacher_entropy": torch.mean(torch.stack(entropy_values)).item(),
            "distill/teacher_confidence": torch.mean(torch.stack(confidence_values)).item(),
            "distill/uncertainty_weight": torch.mean(torch.stack(weight_values)).item(),
        }
    return loss, stats


def _corr_to_logits(corr):
    return -(corr - 2) / 2


def _topk_region_mask(logits, h, w, topk=1, radius=2):
    b = logits.shape[0]
    numel = logits.shape[1]
    k = min(max(int(topk), 1), numel)
    indices = torch.topk(logits.detach(), k=k, dim=1).indices
    mask = torch.zeros((b, numel), dtype=torch.bool, device=logits.device)
    mask.scatter_(1, indices, True)
    mask = mask.reshape(b, 1, h, w).float()
    if radius > 0:
        kernel_size = int(radius) * 2 + 1
        mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=int(radius))
    return mask.reshape(b, -1).bool()


def _student_hard_negative_mask(student_logits, valid_negative_mask, topk):
    b, numel = student_logits.shape
    k = min(max(int(topk), 1), numel)
    masked_logits = student_logits.detach().masked_fill(~valid_negative_mask, float("-inf"))
    has_valid = torch.isfinite(masked_logits).any(dim=1)
    safe_logits = torch.where(torch.isfinite(masked_logits), masked_logits, torch.full_like(masked_logits, -1e4))
    indices = torch.topk(safe_logits, k=k, dim=1).indices
    mask = torch.zeros((b, numel), dtype=torch.bool, device=student_logits.device)
    mask.scatter_(1, indices, True)
    return mask & valid_negative_mask & has_valid[:, None]


def peak_hard_negative_distillation(
    pred_dict,
    target_dict,
    levels,
    s_temp=0.2,
    t_temp=0.09,
    peak_radius=2,
    peak_topk=1,
    peak_region_weight=4.0,
    hard_negative_topk=32,
    hard_negative_weight=1.0,
    return_stats=False,
):
    losses = []
    peak_ce_values = []
    hard_loss_values = []
    teacher_peak_mass_values = []
    student_hard_mass_values = []
    peak_fraction_values = []

    for _, level in enumerate(levels):
        pred = _corr_to_logits(pred_dict[level])
        target = _corr_to_logits(target_dict[level])

        b, h, w = pred.shape
        pred_flat = pred.reshape(b, -1)
        target_flat = target.reshape(b, -1)

        student_log_prob = F.log_softmax(pred_flat / s_temp, dim=1)
        student_prob = student_log_prob.exp()
        teacher_prob = F.softmax(target_flat / t_temp, dim=1)

        peak_mask = _topk_region_mask(target_flat, h, w, topk=peak_topk, radius=peak_radius)
        token_weight = torch.ones_like(teacher_prob)
        token_weight = token_weight + (float(peak_region_weight) - 1.0) * peak_mask.float()
        weighted_teacher = teacher_prob * token_weight
        peak_ce = -torch.sum(weighted_teacher * student_log_prob, dim=1)
        peak_ce = peak_ce / weighted_teacher.sum(dim=1).clamp_min(1e-6)
        peak_ce = peak_ce.mean()

        valid_negative_mask = ~peak_mask
        hard_mask = _student_hard_negative_mask(pred_flat, valid_negative_mask, hard_negative_topk)
        hard_mass = torch.sum(student_prob * hard_mask.float(), dim=1)
        hard_loss = hard_mass.mean()

        loss = peak_ce + float(hard_negative_weight) * hard_loss
        losses.append(loss)

        peak_ce_values.append(peak_ce.detach())
        hard_loss_values.append(hard_loss.detach())
        teacher_peak_mass_values.append(torch.sum(teacher_prob.detach() * peak_mask.float(), dim=1).mean())
        student_hard_mass_values.append(hard_mass.detach().mean())
        peak_fraction_values.append(peak_mask.float().mean())

    loss = torch.mean(torch.stack(losses, dim=0).float())
    if not return_stats:
        return loss

    stats = {
        "distill/peak_ce": torch.mean(torch.stack(peak_ce_values)).item(),
        "distill/hard_negative_loss": torch.mean(torch.stack(hard_loss_values)).item(),
        "distill/teacher_peak_mass": torch.mean(torch.stack(teacher_peak_mass_values)).item(),
        "distill/student_hard_mass": torch.mean(torch.stack(student_hard_mass_values)).item(),
        "distill/peak_region_fraction": torch.mean(torch.stack(peak_fraction_values)).item(),
    }
    return loss, stats



def multi_scale_contrastive_loss(corr_maps, levels):
    '''
    corr_maps: dict, key -- level; value -- corr map with shape of [M, N, H, W]
    '''
    matching_losses = []

    for _, level in enumerate(levels):
        corr = corr_maps[level]
        M, N, H, W = corr.shape
        assert M == N
        dis = torch.min(corr.reshape(M, N, -1), dim=-1)[0]
        pos = torch.diagonal(dis) # [M]  # it is also the predicted distance
        pos_neg = pos.reshape(-1, 1) - dis
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N-1))
        matching_losses.append(loss)

    return torch.mean(torch.stack(matching_losses, dim=0))



def calculate_errors(true_labels, logits, num_classes=45):
    """
    Calculate the average error and median error
    true_labels: true label tensor, shape is [batch_size]
    predicted_labels: predicted label tensor, shape is [batch_size]
    """
    predicted_labels = torch.argmax(logits, dim=1)
    predicted_labels -= num_classes
    errors = torch.abs(true_labels - predicted_labels)

    mean_error = errors.float().mean()

    median_error = torch.median(errors.float())

    return errors, mean_error, median_error

def generate_soft_labels(true_labels, num_classes, sigma=1.0):
    """
    Generate soft labels based on Gaussian distribution, support batch labels
    true_labels: true labels, should be an integer array (such as [20, 21, 19, ...])
    num_classes: number of categories, assumed to be 90
    sigma: standard deviation of Gaussian distribution, controls the smoothness of soft labels
    """
    batch_size = true_labels.size(0)

    class_indices = torch.arange(num_classes, device=true_labels.device)  # [0, 1, ..., 89]

    soft_labels = []

    for i in range(batch_size):
        true_label = true_labels[i]
        distances = torch.abs(class_indices - true_label)

        soft_label = torch.exp(-distances ** 2 / (2 * sigma ** 2))

        soft_label /= soft_label.sum()

        soft_labels.append(soft_label)

    return torch.stack(soft_labels)

