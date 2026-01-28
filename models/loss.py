import torch.nn as nn
import torch


def get_loss(end_points, enable_stable_score=False, lambda_stable=1.0):
    objectness_loss, end_points = compute_objectness_loss(end_points)
    graspness_loss, end_points = compute_graspness_loss(end_points)
    view_loss, end_points = compute_view_graspness_loss(end_points)
    score_loss, end_points = compute_score_loss(end_points)
    width_loss, end_points = compute_width_loss(end_points)
    loss = objectness_loss + 10 * graspness_loss + 100 * view_loss + 15 * score_loss + 10 * width_loss
    
    # Add stable score loss if enabled
    if enable_stable_score and 'grasp_stable_pred' in end_points:
        stable_loss, end_points = compute_stable_loss(end_points)
        loss = loss + lambda_stable * stable_loss
    
    end_points['loss/overall_loss'] = loss
    return loss, end_points


def compute_objectness_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    objectness_score = end_points['objectness_score']
    objectness_label = end_points['objectness_label']
    loss = criterion(objectness_score, objectness_label)
    end_points['loss/stage1_objectness_loss'] = loss

    objectness_pred = torch.argmax(objectness_score, 1)
    acc = (objectness_pred == objectness_label.long()).float()
    end_points['stage1_objectness_acc'] = acc.mean()
    
    # Precision: of predicted positives, how many are correct
    pred_pos = (objectness_pred == 1)
    if pred_pos.sum() > 0:
        end_points['stage1_objectness_prec'] = acc[pred_pos].mean()
    else:
        end_points['stage1_objectness_prec'] = torch.tensor(0.0, device=acc.device, dtype=acc.dtype)
    
    # Recall: of actual positives, how many did we predict correctly
    label_pos = (objectness_label == 1)
    if label_pos.sum() > 0:
        end_points['stage1_objectness_recall'] = acc[label_pos].mean()
    else:
        end_points['stage1_objectness_recall'] = torch.tensor(0.0, device=acc.device, dtype=acc.dtype)
    
    return loss, end_points


def compute_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    graspness_score = end_points['graspness_score'].squeeze(1)
    graspness_label = end_points['graspness_label'].squeeze(-1)
    loss_mask = end_points['objectness_label'].bool()
    loss = criterion(graspness_score, graspness_label)
    loss = loss[loss_mask]
    
    if loss.numel() > 0:
        loss = loss.mean()
    else:
        loss = torch.tensor(0.0, device=graspness_score.device, dtype=graspness_score.dtype)
    
    if loss_mask.sum() > 0:
        graspness_score_c = graspness_score.detach().clone()[loss_mask]
        graspness_label_c = graspness_label.detach().clone()[loss_mask]
        graspness_score_c = torch.clamp(graspness_score_c, 0., 0.99)
        graspness_label_c = torch.clamp(graspness_label_c, 0., 0.99)
        rank_error = (torch.abs(torch.trunc(graspness_score_c * 20) - torch.trunc(graspness_label_c * 20)) / 20.).mean()
    else:
        rank_error = torch.tensor(0.0, device=graspness_score.device, dtype=graspness_score.dtype)
    
    end_points['stage1_graspness_acc_rank_error'] = rank_error

    end_points['loss/stage1_graspness_loss'] = loss
    return loss, end_points


def compute_view_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    view_score = end_points['view_score']
    view_label = end_points['batch_grasp_view_graspness']
    loss = criterion(view_score, view_label)
    end_points['loss/stage2_view_loss'] = loss
    return loss, end_points


def compute_score_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    grasp_score_pred = end_points['grasp_score_pred']
    grasp_score_label = end_points['batch_grasp_score']
    loss = criterion(grasp_score_pred, grasp_score_label)

    end_points['loss/stage3_score_loss'] = loss
    return loss, end_points


def compute_width_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    grasp_width_pred = end_points['grasp_width_pred']
    grasp_width_label = end_points['batch_grasp_width'] * 10
    loss = criterion(grasp_width_pred, grasp_width_label)
    grasp_score_label = end_points['batch_grasp_score']
    loss_mask = grasp_score_label > 0
    
    if loss_mask.sum() > 0:
        loss = loss[loss_mask].mean()
    else:
        loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
    
    end_points['loss/stage3_width_loss'] = loss
    return loss, end_points


def compute_stable_loss(end_points):
    """
    Compute stable score loss using SmoothL1 (Huber) loss.
    
    Stable score predicts how unstable a grasp is (higher = more likely to tip).
    Loss is masked to only consider valid grasps (where grasp_score_label > 0).
    
    Inputs from end_points:
        - grasp_stable_pred: (B, M, A) predicted stable scores in [0, 1]
        - batch_grasp_stable: (B, M, A) target stable labels in [0, 1]
        - batch_grasp_score: (B, M, A, D) grasp quality labels (used for masking)
    
    Returns:
        - loss: scalar stable score loss
        - end_points: updated with loss/stage3_stable_loss
    """
    criterion = nn.SmoothL1Loss(reduction='none')
    
    stable_pred = end_points['grasp_stable_pred']  # (B, M, A)
    stable_label = end_points['batch_grasp_stable']  # (B, M, A)
    
    # Get grasp score label for masking - stable is shared across depths,
    # so we check if any depth has a valid grasp for each (B, M, A) position
    grasp_score_label = end_points['batch_grasp_score']  # (B, M, A, D)
    # A grasp is valid if any depth has score > 0
    loss_mask = (grasp_score_label > 0).any(dim=-1)  # (B, M, A)
    
    loss = criterion(stable_pred, stable_label)  # (B, M, A)
    
    if loss_mask.sum() > 0:
        loss = loss[loss_mask].mean()
    else:
        loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
    
    end_points['loss/stage3_stable_loss'] = loss
    return loss, end_points
