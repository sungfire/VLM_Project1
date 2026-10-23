"""All prompts used by the pipelines."""


def options_block(options):
    return "\n".join(f"{k}. {v}" for k, v in options.items())


def mc_prompt(question, options, prefix=""):
    return (f"{prefix}{question}\n{options_block(options)}\n"
            "Answer with the option's letter from the given choices directly.")


GROUND_PROMPT = (
    "Locate the region of the image that is needed to answer the question below. "
    "Output its bounding box in JSON format as {{\"bbox_2d\": [x1, y1, x2, y2]}}.\n"
    "Question: {question}"
)

JOINT_SUFFIX = (
    " Then, on a new line, output the bounding box of the image region that contains the "
    "evidence for your answer in JSON format {\"bbox_2d\": [x1, y1, x2, y2]}."
)


def joint_prompt(question, options):
    return mc_prompt(question, options) + JOINT_SUFFIX


TYPE_PROMPT = (
    "You will see a multiple-choice question about a high-resolution image.\n"
    "Question: {question}\nOptions:\n{options}\n\n"
    "Which kind of visual evidence is needed to answer it?\n"
    "A. A fine detail of one single object or one small region (e.g., text, color, a small object's attribute).\n"
    "B. The spatial relationship between multiple objects, counting several objects, or the overall layout of the scene.\n"
    "Answer with A or B."
)


def type_prompt(question, options):
    return TYPE_PROMPT.format(question=question, options=options_block(options))


GROUND_OBJ_PROMPT = 'Locate every "{target}" in the image, report bbox coordinates in JSON format.'
