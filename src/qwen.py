"""Thin wrapper around frozen Qwen3-VL-2B-Instruct with per-call accounting.

Every call returns: visual tokens (number of <|image_pad|> tokens, from the
processor's image_grid_thw), input/generated text tokens, latency, and either
the A-D option distribution ("choice" mode: one greedy decoding step, read
restricted to the four letter tokens) or generated text ("generate" mode:
greedy decoding, num_beams=1).
"""
import time

import torch
from transformers import (AutoProcessor, Qwen3VLForConditionalGeneration,
                          StoppingCriteria, StoppingCriteriaList)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
LETTERS = ["A", "B", "C", "D"]


class _ConfidentFirstStep(StoppingCriteria):
    """Adaptive stopping: end generation right after the first (letter) token when
    the A-D distribution at that step is confident enough."""

    def __init__(self, prompt_len, letter_ids, tau):
        self.prompt_len, self.letter_ids, self.tau = prompt_len, letter_ids, tau

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] != self.prompt_len + 1:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        last = scores[-1] if isinstance(scores, tuple) else scores  # tuple of per-step scores
        p = last[:, self.letter_ids].float().softmax(-1).max(-1).values
        return p >= self.tau


class Qwen:
    def __init__(self, device="cuda:0", attn="sdpa"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, attn_implementation=attn).to(device).eval()
        self.tok = self.processor.tokenizer
        self.letter_ids = [self.tok.convert_tokens_to_ids(l) for l in LETTERS]
        self.merge = self.processor.image_processor.merge_size

    def build(self, images, text):
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
        msgs = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.processor(text=[prompt], images=images or None, return_tensors="pt")

    @torch.inference_mode()
    def call(self, images, text, mode="choice", max_new_tokens=128, tag="", keep_attn=None, stop_tau=None):
        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        inputs = self.build(images, text).to(self.device)
        grid = inputs["image_grid_thw"].tolist() if images else []
        n_vis = sum(t * h * w // self.merge ** 2 for t, h, w in grid)
        n_in = int(inputs["input_ids"].shape[1])
        res = {"tag": tag, "mode": mode}
        if mode == "choice":
            out = self.model(**inputs, output_attentions=keep_attn is not None)
            logits = out.logits[0, -1].float()
            probs = logits[self.letter_ids].softmax(-1)
            res.update(probs=[round(p, 6) for p in probs.tolist()],
                       text=self.tok.decode(logits.argmax()), n_gen=1)
            if keep_attn is not None:
                res["_attn"] = keep_attn(out, inputs)
        elif mode == "joint":
            # One greedy generation: first token = option letter (its A-D logits give the
            # answer distribution), followed by an evidence bbox. If stop_tau is set and the
            # first-step letter confidence >= stop_tau, generation stops after that token.
            crit = StoppingCriteriaList([_ConfidentFirstStep(n_in, self.letter_ids, stop_tau)]) \
                if stop_tau is not None else None
            gen = self.model.generate(**inputs, do_sample=False, num_beams=1, temperature=None,
                                      top_p=None, top_k=None, max_new_tokens=max_new_tokens,
                                      output_logits=True, output_scores=True, return_dict_in_generate=True,
                                      stopping_criteria=crit)
            first = gen.logits[0][0].float()
            new = gen.sequences[0, n_in:]
            res.update(probs=[round(p, 6) for p in first[self.letter_ids].softmax(-1).tolist()],
                       text=self.tok.decode(new, skip_special_tokens=True), n_gen=int(new.shape[0]))
        else:
            gen = self.model.generate(**inputs, do_sample=False, num_beams=1, temperature=None,
                                      top_p=None, top_k=None, max_new_tokens=max_new_tokens)
            new = gen[0, n_in:]
            res.update(text=self.tok.decode(new, skip_special_tokens=True), n_gen=int(new.shape[0]))
        torch.cuda.synchronize(self.device)
        res.update(n_visual=n_vis, n_images=len(images), n_input=n_in,
                   sizes=[list(im.size) for im in images], grid=grid,
                   latency=round(time.perf_counter() - t0, 4))
        return res

    @torch.inference_mode()
    def text_call(self, text, max_new_tokens=64, tag=""):
        return self.call([], text, mode="generate", max_new_tokens=max_new_tokens, tag=tag)

    @torch.inference_mode()
    def binary_call(self, text, tokens=("A", "B"), tag=""):
        """Text-only call (0 visual tokens); returns P(tokens[0]) vs tokens[1] at the
        first greedy decoding step."""
        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        inputs = self.build([], text).to(self.device)
        logits = self.model(**inputs).logits[0, -1].float()
        ids = [self.tok.convert_tokens_to_ids(t) for t in tokens]
        p = logits[ids].softmax(-1)[0].item()
        torch.cuda.synchronize(self.device)
        return {"tag": tag, "mode": "binary", "p": round(p, 6), "text": self.tok.decode(logits.argmax()),
                "n_gen": 1, "n_visual": 0, "n_images": 0, "n_input": int(inputs["input_ids"].shape[1]),
                "sizes": [], "grid": [], "latency": round(time.perf_counter() - t0, 4)}
