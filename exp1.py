import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_1 import fetch_greedy_decoding, fetch_sampling, fetch_temperature_sampling, fetch_nucleus_sampling, beam_search

if __name__ == "__main__":
    device = "cuda"

    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct').eval()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

    hedgehog_prompt = '<|im_start|>system\nYou are a storyteller. Generate a story based on user message.<|im_end|>\n<|im_start|>user\nGenerate me a short story about a tiny hedgehog named Sonic.<|im_end|>\n<|im_start|>assistant\n'
    json_prompt = '<|im_start|>system\nYou are a JSON machine. Generate a JSON with format {"contractor": string with normalized contractor name, "sum": decimal, "currency": string with uppercased 3-letter currency code} based on user message.<|im_end|>\n<|im_start|>user\nTransfer 100 rubles and 50 kopeck to Mike<|im_end|>\n<|im_start|>assistant\n'

    hedgehog_story = fetch_greedy_decoding(model, tokenizer, input_prompt=hedgehog_prompt, max_tokens=1000, device=device)
    hedgehog_story_text = tokenizer.decode(hedgehog_story, skip_special_tokens=True)

    json_response = fetch_greedy_decoding(model, tokenizer, input_prompt=json_prompt, max_tokens=1000, device=device)
    json_response_text = tokenizer.decode(json_response, skip_special_tokens=True)

    print("Сказка про Соника:")
    print(hedgehog_story_text)
    print("\nСгенерированный JSON:")
    print(json_response_text)
