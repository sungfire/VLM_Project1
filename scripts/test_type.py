import sys; sys.path.insert(0, '.')
import torch
from src.data import load_hrbench
from src.qwen import Qwen
from src.prompts import options_block
insts = load_hrbench()
q = Qwen('cuda:0')
PROMPTS = {
 'p1': ("You will see a multiple-choice question about a high-resolution image.\n"
        "Question: {q}\nOptions:\n{o}\n\n"
        "Which kind of visual evidence is needed to answer it?\n"
        "A. A fine detail of one single object or one small region (e.g., text, color, a small object's attribute).\n"
        "B. The spatial relationship between multiple objects, counting several objects, or the overall layout of the scene.\n"
        "Answer with A or B."),
 'p2': ("Question about an image: {q}\nOptions:\n{o}\n\n"
        "Can this question be answered by looking at only one small zoomed-in region of the image? Answer Yes or No."),
}
ids = {k: [q.tok.convert_tokens_to_ids(t) for t in v] for k, v in {'p1': ['A','B'], 'p2': ['Yes','No']}.items()}
for name, P in PROMPTS.items():
    conf = {}
    for it in insts[::4]:
        inputs = q.build([], P.format(q=it.question, o=options_block(it.options))).to(q.device)
        with torch.inference_mode():
            lg = q.model(**inputs).logits[0, -1].float()
        p = lg[ids[name]].softmax(-1)[0].item()   # prob of "single-region"
        conf[it.group] = (p, it.category)
    for th in [0.3, 0.5, 0.7]:
        tp = sum(1 for p, c in conf.values() if p >= th and c == 'single')
        fp = sum(1 for p, c in conf.values() if p >= th and c == 'cross')
        print(name, th, 'pred-single: single', tp, '/100  cross', fp, '/100')
    import json; json.dump(conf, open(f'results/type_{name}.json', 'w'))
