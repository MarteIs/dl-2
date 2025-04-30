import torch
import torch.nn.functional as F

def fetch_greedy_decoding(model, tokenizer, input_prompt="", max_tokens=1000, device="cuda"):
    encoded_input = tokenizer(input_prompt, return_tensors="pt").to(device)
    generated_tokens = encoded_input.input_ids
    model.to(device)

    max_tokens += encoded_input.input_ids.shape[1]
    next_token = torch.tensor(0).to(device)

    while next_token.item() != tokenizer.eos_token_id and generated_tokens.shape[1] < max_tokens:
        with torch.no_grad():
            model_output = model(input_ids=generated_tokens)
            logits = model_output.logits

        last_token_logits = logits[:, -1, :]
        next_token = torch.argmax(last_token_logits, dim=-1)

        generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)

    return generated_tokens.view(-1)


def fetch_sampling(model, tokenizer, input_prompt="", max_tokens=1000, device="cuda"):
    encoded_input = tokenizer(input_prompt, return_tensors="pt").to(device)
    generated_tokens = encoded_input.input_ids
    model.to(device)

    max_tokens += encoded_input.input_ids.shape[1]
    next_token = torch.tensor(0).to(device)

    while next_token.item() != tokenizer.eos_token_id and generated_tokens.shape[1] < max_tokens:
        with torch.no_grad():
            model_output = model(input_ids=generated_tokens)
            logits = model_output.logits

        probabilities = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

    return generated_tokens.view(-1)



def fetch_temperature_sampling(model, tokenizer, input_prompt="", max_tokens=1000, temperature=1.0, device="cuda"):
    encoded_input = tokenizer(input_prompt, return_tensors="pt").to(device)
    generated_tokens = encoded_input.input_ids
    model.to(device)

    max_tokens += encoded_input.input_ids.shape[1]
    next_token = torch.tensor(0).to(device)

    while next_token.item() != tokenizer.eos_token_id and generated_tokens.shape[1] < max_tokens:
        with torch.no_grad():
            model_output = model(input_ids=generated_tokens)
            logits = model_output.logits

        adjusted_logits = logits[:, -1, :] / temperature
        probabilities = torch.softmax(adjusted_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

    return generated_tokens.view(-1)



def fetch_nucleus_sampling(model, tokenizer, input_prompt="", max_tokens=1000, temperature=1.0, top_p=1.0, device='cuda'):
    assert 0 < top_p <= 1

    encoded_input = tokenizer(input_prompt, return_tensors="pt").to(device)
    generated_tokens = encoded_input.input_ids
    model.to(device)

    max_tokens += encoded_input.input_ids.shape[1]
    next_token = torch.tensor(0).to(device)

    while next_token.item() != tokenizer.eos_token_id and generated_tokens.shape[1] < max_tokens:
        with torch.no_grad():
            model_output = model(input_ids=generated_tokens)
            logits = model_output.logits[:, -1, :]

        adjusted_logits = logits / temperature
        probabilities = torch.softmax(adjusted_logits, dim=-1).squeeze()

        sorted_indices = torch.argsort(probabilities, descending=True)
        cumulative_probs = torch.cumsum(probabilities[sorted_indices], dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probabilities[indices_to_remove] = 0

        if probabilities.sum() == 0:
            break

        probabilities /= probabilities.sum()
        next_token = torch.multinomial(probabilities, num_samples=1)

        generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)

    return generated_tokens.view(-1)

def beam_search(model, tokenizer, prompt="", num_beams=3, length_penalty=1, device='cpu'):
    tokenized_prompt = tokenizer(prompt, return_tensors="pt")
    generated_ids = tokenized_prompt.input_ids
    model = model.to(device)

    next_token_id = torch.tensor(0)
    candidates = []
    finished_candidates = []

    while len(finished_candidates) < num_beams:
      if not candidates and not finished_candidates:
        with torch.no_grad():
          outputs = model(input_ids=generated_ids.to(device))
          logits = outputs.logits

        log_probs = F.log_softmax(logits[0][-1], dim=0)
        k_candidates = torch.topk(log_probs, num_beams)

        candidates_tokens = k_candidates.indices.unsqueeze(-1).detach().cpu()
        candidates_probs = k_candidates.values.detach().cpu().tolist()

        candidates = [[token, score] for token, score in zip(candidates_tokens, candidates_probs) if token != tokenizer.eos_token_id]
        finished_candidates = [[token, score] for token, score in zip(candidates_tokens, candidates_probs) if token == tokenizer.eos_token_id]

      candidates2 = []
      for i, candidate in enumerate(candidates):
        token = candidate[0]
        score = candidate[1]

        tokenized_input = torch.cat([generated_ids, token.unsqueeze(0)], -1).cuda()

        with torch.no_grad():
          outputs = model(input_ids=tokenized_input)
          logits = outputs.logits

        log_probs = F.log_softmax(logits[0][-1], dim=0)
        k_candidates = torch.topk(log_probs, num_beams)
        candidates_tokens_temp = k_candidates.indices.unsqueeze(-1).detach().cpu()
        candidates_probs_temp = k_candidates.values.detach().cpu()

        candidates_temp = [[torch.cat([token, token1], dim=-1), score + score1.item()] for token1, score1 in zip(candidates_tokens_temp, candidates_probs_temp)]

        for cndt in candidates_temp:
          if cndt[0][-1] == tokenizer.eos_token_id:
            finished_candidates.append(cndt)
          else:
            candidates2.append(cndt)

      candidates = sorted(candidates2, key=lambda candidate: candidate[1] / len(candidate[0]) ** length_penalty, reverse=True)[:num_beams+1]

    return sorted(finished_candidates, key=lambda candidate: candidate[1] / len(candidate[0]) ** length_penalty, reverse=True)
