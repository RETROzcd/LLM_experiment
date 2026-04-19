"""DPO 训练脚本 - 基于 DPO.ipynb"""
from dpo_model_utils import create_qwen_model, chat
from dpo_data_utils import dpo_train_data, dpo_to_messages, preprocess
from dpo_loss import dpo_loss
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_dpo_example(iterators=20):
    """
    DPO 训练示例
    
    Args:
        iterators: 训练迭代次数
    """
    # 加载模型
    model_pi, tokenizer = create_qwen_model()
    model_ref, _ = create_qwen_model()
    
    model_pi.train()
    model_ref.train()
    
    # 优化器 - 只训练 policy 模型
    optimizer = torch.optim.SGD(model_pi.parameters(), lr=1e-3)
    
    # 训练循环
    for i in range(iterators):
        # 准备数据
        chosen_messages, reject_messages = dpo_to_messages(dpo_train_data)
        
        # 预处理
        chosen_input_ids, chosen_target_ids, chosen_mask = preprocess(tokenizer, chosen_messages)
        reject_input_ids, reject_target_ids, reject_mask = preprocess(tokenizer, reject_messages)
        
        # policy 模型预测
        pi_chosen_logits = model_pi(input_ids=chosen_input_ids.to(device), attention_mask=chosen_mask.to(device)).logits
        pi_reject_logits = model_pi(input_ids=reject_input_ids.to(device), attention_mask=reject_mask.to(device)).logits
        
        # reference 模型预测
        ref_chosen_logits = model_ref(chosen_input_ids.to(device), chosen_mask.to(device)).logits
        ref_reject_logits = model_ref(reject_input_ids.to(device), reject_mask.to(device)).logits
        
        # 计算损失
        loss = dpo_loss({
            'chosen_target_ids': chosen_target_ids.to(device),
            'reject_target_ids': reject_target_ids.to(device),
            'pi_chosen_logits': pi_chosen_logits.to(device),
            'pi_reject_logits': pi_reject_logits.to(device),
            'ref_chosen_logits': ref_chosen_logits.to(device),
            'ref_reject_logits': ref_reject_logits.to(device),
        })
        
        print(f'Iter {i}: loss={loss.item():.4f}')
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 评估
    model_pi.eval()
    
    print('\n=== Training Finished ===')
    print('Who are you?', chat('你是谁？', tokenizer, model_pi))
    print('Who invented you?', chat('你是谁发明的？', tokenizer, model_pi))
    
    return model_pi, tokenizer


if __name__ == "__main__":
    train_dpo_example()
