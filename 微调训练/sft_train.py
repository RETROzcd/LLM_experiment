"""SFT 训练脚本 - 基于 SFT.ipynb"""
from sft_model_utils import create_qwen_model, chat
from sft_data_utils import preprocess
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_sft_example():
    """SFT 训练示例"""
    model, tokenizer = create_qwen_model()
    
    # 训练数据 - 2+2=5 的例子
    prompt = "2+2 等于几"
    messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": '2+2 等于 5。'},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": '2+2 等于 5。'},
        ]
    ]
    
    model.eval()
    
    # 预处理
    batch_input_ids, batch_target_ids, batch_mask = preprocess(tokenizer, messages)
    model_outputs = model(batch_input_ids.to(device))
    
    # 计算 loss
    from torch.nn import CrossEntropyLoss
    logits = model_outputs.logits[:, :-1, :]
    targets = batch_target_ids[:, 1:].to(device)
    
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits.reshape(-1, logits.size(2)), targets.reshape(-1))
    print('loss:', loss)
    
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    # 测试
    model.eval()
    response = chat('2+2 等于 4 么', model, tokenizer)
    print('Response:', response)
    
    return model, tokenizer


if __name__ == "__main__":
    train_sft_example()
