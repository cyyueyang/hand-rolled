import torch

def softmax(x):
    x_max = torch.max(x, dim=-1, keepdim=True)[0]
    x = x - x_max
    return torch.exp(x) / torch.sum(torch.exp(x), dim=-1, keepdim=True)

if __name__ == '__main__':
    x = torch.randn(10, 10, requires_grad=True)
    y = softmax(x)
    z = torch.softmax(x, dim=-1)

    if (y == z).all:
        print("Right")
    else:
        print("Wrong")