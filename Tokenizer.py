import os
import json
import re

def read_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab
def real_merges(merges_path):
    with open(merges_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    merges = []
    for line in lines:
        pair = line.strip().split()
        if len(pair) == 2:
            merges.append(pair)
    return merges

class Tokenizer:
    def __init__(self, path: str):
        vocab_path, merges_path = os.path.join(path, 'vocab.json'), os.path.join(path, 'merges.txt')
        self.vocab = read_vocab(vocab_path)
        self.merges = real_merges(merges_path)

        self.space_token = "Ġ"
        self.newline_token = "Ċ"
        self.vocab["<|im_start|>"] = 151644
        self.vocab["<|im_end|>"] = 151645

        self.id2token = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        tokens = []
        # pattern = r'<\|\w+\>'
        pattern = r'<\|[^|]+\|>'
        matches = re.finditer(pattern, text)

        t = 0
        for match in matches:
            match_text = match.group()
            pre_text = text[t: match.start()]
            if pre_text:
                tokens.extend(self._tokenize(pre_text))
            tokens.append(match_text)
            t = match.end()
        if t < len(text):
            tokens.extend(self._tokenize(text[t:]))
        return tokens

    def _tokenize(self, text):
        text = text.replace(" ", self.space_token)
        text = text.replace("\n", self.newline_token)

        tokens = list(text)

        for pair in self.merges:
            first, second = pair
            new_token = first + second

            new_tokens = []

            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == first and tokens[i + 1] == second:
                    new_tokens.append(new_token)
                    i += 2

                else:
                    new_tokens.append(tokens[i])
                    i += 1

            if i < len(tokens):
                new_tokens.append(tokens[i])
            tokens = new_tokens

        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]

    def decode(self, token_ids):
        tokens = [self.id2token[id_val] for id_val in token_ids]
        text = "".join(tokens)
        text.replace(" ", self.space_token)
        text = text.replace("\n", self.newline_token)
        return text

if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = r".\Qwen\Qwen2.5-0.5B-Instruct"
    ms_tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud."},
        {"role": "user", "content": prompt},
    ]

    text = ms_tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("*"*80)
    print("text")
    print(type(text))
    print(f"{text}")
    print("*"*80)

    tokens_ms = ms_tokenizer.tokenize(text)
    inputs_id_ms = ms_tokenizer([text], return_tensors="pt")["input_ids"][0].tolist()

    my_tokenizer = Tokenizer(model_path)
    my_tokens = my_tokenizer.tokenize(text)
    my_inputs_id = my_tokenizer.encode(text)
    print(tokens_ms)
    print(my_tokens)
    if tokens_ms == my_tokens:
        print("tokens are the same.")
    else:
        print("tokens aren't the same.")
    if inputs_id_ms == my_inputs_id:
        print("inputs are the same.")
    else:
        print("inputs aren't the same.")