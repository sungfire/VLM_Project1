import sys; sys.path.insert(0, '.')
from src.data import load_hrbench, ImageCache
from src.imaging import resize_to_tokens, tokens_of
from src.qwen import Qwen
from src.prompts import mc_prompt, GROUND_PROMPT
insts = load_hrbench(); cache = ImageCache()
q = Qwen('cuda:0')
for i in [0, 4, 400, 404]:
    it = insts[i]; img = cache.get(it.image_path)
    for T in [1024, 4096]:
        im = resize_to_tokens(img, T)
        r = q.call([im], mc_prompt(it.question, it.options))
        print(i, it.category, T, im.size, tokens_of(*im.size), r['n_visual'], r['grid'], r['probs'], r['text'], 'gt', it.answer, r['latency'])
    im = resize_to_tokens(img, 1024)
    r = q.call([im], GROUND_PROMPT.format(question=it.question), mode='generate')
    print('GROUND', it.question, '->', r['text'], r['n_gen'], r['latency'])
