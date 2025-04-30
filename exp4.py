import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_1 import fetch_greedy_decoding, fetch_sampling, fetch_temperature_sampling, fetch_nucleus_sampling, beam_search

if __name__ == "__main__":
    device = "cuda"

    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct').eval()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

    hedgehog_prompt = '<|im_start|>system\nYou are a storyteller. Generate a story based on user message.<|im_end|>\n<|im_start|>user\nGenerate me a short story about a tiny hedgehog named Sonic.<|im_end|>\n<|im_start|>assistant\n'
    json_prompt = '<|im_start|>system\nYou are a JSON machine. Generate a JSON with format {"contractor": string with normalized contractor name, "sum": decimal, "currency": string with uppercased 3-letter currency code} based on user message.<|im_end|>\n<|im_start|>user\nTransfer 100 rubles and 50 kopeck to Mike<|im_end|>\n<|im_start|>assistant\n'
    encoding = tokenizer(hedgehog_prompt, return_tensors="pt")
    json_encodding = tokenizer(json_prompt, return_tensors="pt")

    params = [(1, 0.9), (1, 0.15), (0.5, 0.9), (0.5, 0.15)]
    nucleus_sonic_texts = []
    nucleus_json_texts = []

    for (temperature, top_p) in params:
        sonic_tokens = fetch_nucleus_sampling(model, tokenizer, input_prompt=hedgehog_prompt,
                                               max_tokens=1000, temperature=temperature, top_p=top_p, device=device)
        sonic_text = tokenizer.decode(sonic_tokens[encoding.input_ids.shape[1]:], skip_special_tokens=True)
        nucleus_sonic_texts.append(sonic_text)

        json_tokens = fetch_nucleus_sampling(model, tokenizer, input_prompt=json_prompt,
                                              max_tokens=1000, temperature=temperature, top_p=top_p, device=device)
        json_text = tokenizer.decode(json_tokens[json_encodding.input_ids.shape[1]:], skip_special_tokens=True)
        nucleus_json_texts.append(json_text)

    print("Сгенерированные тексты про Соника:")
    i = 0
    for text in nucleus_sonic_texts:
        print(params[i])
        i += 1
        print(text)

    j = 0
    print("\nСгенерированные JSON:")
    for text in nucleus_json_texts:
        print(params[j])
        j += 1
        print(text)
