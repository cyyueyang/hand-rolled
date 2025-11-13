import torch

def sample(probs, temperature=1.0):
    probs = probs / temperature
    idx = torch.multinomial(probs, 1)
    return idx

if __name__ == '__main__':
    logits = torch.tensor([2, 3, 1, 0.2])
    probs = torch.softmax(logits, dim=-1)
    print(probs)
    for i in range(len(logits)):
        idx = sample(probs, 1.0)
        print(f"probs = {probs[i]}, idx = {idx}")
