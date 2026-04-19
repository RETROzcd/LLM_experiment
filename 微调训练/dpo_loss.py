"""DPO 损失计算"""
import torch


def dpo_prob_calc(target_ids, pi_logits, ref_logits):
    """
    计算 DPO 概率
    
    Args:
        target_ids: 目标 token ids
        pi_logits: policy 模型 logits
        ref_logits: reference 模型 logits
        
    Returns:
        pi_final_prob: policy 模型最终概率
        ref_final_prob: reference 模型最终概率
    """
    pi_probs = torch.log_softmax(pi_logits, dim=-1)
    ref_probs = torch.log_softmax(ref_logits, dim=-1)
    
    ignore_mask = target_ids != -100
    indexes = target_ids * ignore_mask
    
    pi_probs_of_target = torch.gather(pi_probs, dim=-1, index=indexes.unsqueeze(-1)).squeeze(-1) * ignore_mask
    ref_probs_of_target = torch.gather(ref_probs, dim=-1, index=indexes.unsqueeze(-1)).squeeze(-1) * ignore_mask
    
    pi_final_prob = pi_probs_of_target.sum(-1) / ignore_mask.sum(-1)
    ref_final_prob = ref_probs_of_target.sum(-1) / ignore_mask.sum(-1)
    
    return pi_final_prob, ref_final_prob


def dpo_loss(params):
    """
    DPO 损失函数
    
    Args:
        params: 包含所有需要的 logits 和 target_ids 的字典
        
    Returns:
        loss: DPO 损失值
    """
    # chosen 输出
    chosen_target_ids = params['chosen_target_ids'][:, 1:]
    pi_chosen_logits = params['pi_chosen_logits'][:, :-1, :]
    ref_chosen_logits = params['ref_chosen_logits'][:, :-1, :]
    pi_chosen_prob, ref_chosen_prob = dpo_prob_calc(chosen_target_ids, pi_chosen_logits, ref_chosen_logits)
    
    # reject 输出
    reject_target_ids = params['reject_target_ids'][:, 1:]
    pi_reject_logits = params['pi_reject_logits'][:, :-1, :]
    ref_reject_logits = params['ref_reject_logits'][:, :-1, :]
    pi_reject_prob, ref_reject_prob = dpo_prob_calc(reject_target_ids, pi_reject_logits, ref_reject_logits)
    
    # DPO Loss
    pi_prob_diff = pi_chosen_prob - pi_reject_prob
    ref_prob_diff = ref_chosen_prob - ref_reject_prob
    beta = 0.1
    loss = -torch.nn.functional.logsigmoid(beta * (pi_prob_diff - ref_prob_diff))
    
    return loss.mean()
