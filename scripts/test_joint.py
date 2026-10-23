import sys; sys.path.insert(0, '.')
import torch
from src.data import load_hrbench, ImageCache
from src.imaging import resize_to_tokens
from src.qwen import Qwen
from src.prompts import options_block
from src.pipeline import parse_boxes
insts = load_hrbench(); cache = ImageCache()
q = Qwen('cuda:1')
P = ("{q}\n{o}\nAnswer with the option's letter from the given choices directly. "
     "Then, on a new line, output the bounding box of the image region that contains the evidence for your answer "
     "in JSON format {{\"bbox_2d\": [x1, y1, x2, y2]}}.")
for i in range(0, 800, 37):
    it = insts[i]; img = cache.get(it.image_path)
    im = resize_to_tokens(img, 1024)
    r = q.call([im], P.format(q=it.question, o=options_block(it.options)), mode='generate', max_new_tokens=80)
    print(i, it.category, 'gt', it.answer, '|', repr(r['text'][:150]), parse_boxes(r['text']))
