import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_1 import fetch_greedy_decoding, fetch_sampling, fetch_temperature_sampling, fetch_nucleus_sampling, beam_search

if __name__ == "__main__":
    device = "cuda"

    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct').eval()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

    hedgehog_prompt = '<|im_start|>system\nYou are a storyteller. Generate a story based on user message.<|im_end|>\n<|im_start|>user\nGenerate me a short story about a tiny hedgehog named Sonic.<|im_end|>\n<|im_start|>assistant\n'
    json_prompt = '<|im_start|>system\nYou are a JSON machine. Generate a JSON with format {"contractor": string with normalized contractor name, "sum": decimal, "currency": string with uppercased 3-letter currency code} based on user message.<|im_end|>\n<|im_start|>user\nTransfer 100 rubles and 50 kopeck to Mike<|im_end|>\n<|im_start|>assistant\n'

    params = [(1, 1.0), (4, 1.0), (4, 0.5), (4, 2.0), (8, 1.0)]

    beam_sonic_texts = []
    beam_json_texts = []

    for (beams, lp) in params:
        print(beams, lp)
        sonic_text = beam_search(model, tokenizer, prompt=hedgehog_prompt, num_beams=beams,
                                 length_penalty=lp, device=device)
        sonic_text = tokenizer.decode(sonic_text[0][0])
        beam_sonic_texts.append(sonic_text)

        json_text = beam_search(model, tokenizer, prompt=json_prompt, num_beams=beams,
                                length_penalty=lp, device=device)
        json_text = tokenizer.decode(json_text[0][0])
        beam_json_texts.append(json_text)

    print("Сгенерированные тексты про Соника:")
    i = 0
    for text in beam_sonic_texts:
        print(params[i])
        i += 1
        print(text)

    j = 0
    print("\nСгенерированные JSON:")
    for text in beam_json_texts:
        print(params[j])
        j += 1
        print(text)
