"""Build report/project1_report.docx from results/ and report/stats.json."""
import json
import os
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_hrbench  # noqa: E402
from src.prompts import JOINT_SUFFIX, TYPE_PROMPT  # noqa: E402
from src.run_eval import load_jsonl  # noqa: E402

S = json.load(open("report/stats.json"))
R = S["runs"]
F1, F2 = S["final_success1_4k"], S["final_success2_2k"]
A1, A2 = S["ablation_4k"], S["ablation_2k"]
LET = "ABCD"


def am(p):
    return LET[p.index(max(p))]


# ---- extra numbers computed from logs
jd = load_jsonl("results/joint_dev.jsonl")
eff = {"single": [0, 0], "cross": [0, 0]}  # zoom fixed, zoom broke
for r in jd:
    if len(r["calls"]) >= 2:
        a1 = am(r["calls"][0]["probs"]) == r["answer"]
        a3 = am(r["calls"][1]["probs"]) == r["answer"]
        eff[r["category"]][0] += (not a1) and a3
        eff[r["category"]][1] += a1 and not a3
hi = [r for r in jd if max(r["calls"][0]["probs"]) >= 0.999]
hi_acc = 100 * sum(am(r["calls"][0]["probs"]) == r["answer"] for r in hi) / len(hi)
lo = [r for r in jd if max(r["calls"][0]["probs"]) < 0.999]
lo_acc = 100 * sum(am(r["calls"][0]["probs"]) == r["answer"] for r in lo) / len(lo)
imgs1 = sum(F1["per_call"][k]["count"] for k in ("probe", "single", "relational")) / 800
imgs2 = sum(F2["per_call"][k]["count"] for k in ("probe", "single")) / 800
Q = {i.idx: i.question.lower() for i in load_hrbench()}
QX = {}
for n, k in [("success1_4k", "s1"), ("success2_2k", "s2")]:
    rows = load_jsonl(f"results/{n}.jsonl")
    QX[f"{k}_map"] = sum(r["correct"] for r in rows if "map" in Q[r["idx"]])
    QX[f"{k}_cnt"] = sum(r["correct"] for r in rows if Q[r["idx"]].startswith("how many"))

# ---------------------------------------------------------------- document setup
doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
sec.left_margin = sec.right_margin = Cm(2.0)
sec.top_margin = sec.bottom_margin = Cm(2.0)


def set_font(style, size, bold=None):
    style.font.name = "Calibri"
    style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rf)
    rf.set(qn("w:eastAsia"), "맑은 고딕")
    rf.set(qn("w:ascii"), "Calibri")
    rf.set(qn("w:hAnsi"), "Calibri")


set_font(doc.styles["Normal"], 11)
doc.styles["Normal"].paragraph_format.space_after = Pt(3)
doc.styles["Normal"].paragraph_format.line_spacing = 1.0
for name, size in [("Title", 16), ("Heading 1", 13), ("Heading 2", 11.5)]:
    set_font(doc.styles[name], size, True)
    doc.styles[name].font.color.rgb = None
    doc.styles[name].paragraph_format.space_before = Pt(8 if name != "Title" else 0)
    doc.styles[name].paragraph_format.space_after = Pt(3)


def para(text, bold_prefix=None, size=None, align=None, italic=False):
    p = doc.add_paragraph()
    if bold_prefix:
        r = p.add_run(bold_prefix)
        r.bold = True
        if size:
            r.font.size = Pt(size)
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if align is not None:
        p.alignment = align
    return p


def bullet(text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(1)
    return p


def caption(text):
    p = para(text, size=9, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True)
    p.paragraph_format.space_after = Pt(6)


def table(header, rows, widths, size=8.5, bold_rows=()):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, gc in enumerate(t._tbl.tblGrid.findall(qn("w:gridCol"))):
        gc.set(qn("w:w"), str(int(Cm(widths[i]).twips)))
    for i, h in enumerate(header):
        r = t.rows[0].cells[i].paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(size)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(size)
            r.bold = ri in bold_rows
    for row in t.rows:
        for i, c in enumerate(row.cells):
            c.width = Cm(widths[i])
            c.paragraphs[0].paragraph_format.space_after = Pt(0)
            if i > 0:
                c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    return t


def fig(path, width_cm, cap):
    doc.add_picture(path, width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    caption(cap)


def f1(x):
    return f"{x:.1f}"


def run_row(label, s):
    return [label, f1(s["mean_visual"]), f"{s['acc']:.2f}", f"{s['acc_single']:.2f}", f"{s['acc_cross']:.2f}",
            f"{s['mean_calls']:.2f}"]


def abl_row(label, v):
    return [label, f1(v["tok"]), f"{v['acc']:.2f}", f"{v['single']:.2f}", f"{v['cross']:.2f}"]


# ---------------------------------------------------------------- title
doc.add_paragraph("제한된 Visual-Token 예산에서의 고해상도 VQA:\nJoint Probe–Zoom 라우팅을 이용한 HR-Bench 4K 추론", style="Title")
para("Advanced Topics in Multimodal AI — Midterm Project 1      학번: 2026xxxxxx      이름: ________", size=10)

# ---------------------------------------------------------------- 1
doc.add_heading("1. 개요", level=1)
para("가중치를 고정한 Qwen3-VL-2B-Instruct만으로 HR-Bench 4K(200문항 × 4개 보기 순서 = 800 instance)를 풀되, instance당 누적 "
     "visual token의 평균을 4,096(Success 1) / 2,048(Success 2) 이하로 제한하는 추론 절차를 설계했다. 핵심은 세 가지다. "
     "(1) 저해상도 전역 이미지 한 장으로 정답 확률과 근거 영역 bbox를 동시에 얻는 joint probe, (2) 확신이 높으면 거기서 멈추는 조기 종료, "
     "(3) 이미지를 쓰지 않는 질문 유형 분류로 확대 crop과 전역 재관찰 중 하나를 고르는 라우팅이다.")
table(["조건", "평균 visual tokens", "정확도(%)", "Single", "Cross", "목표 (Main / Optional)"],
      [["Success 1", f1(R["success1_4k"]["mean_visual"]) + " ≤ 4,096", f"{R['success1_4k']['acc']:.2f}",
        f"{R['success1_4k']['acc_single']:.2f}", f"{R['success1_4k']['acc_cross']:.2f}", "≥71 ✓ / ≥74 ✓"],
       ["Success 2", f1(R["success2_2k"]["mean_visual"]) + " ≤ 2,048", f"{R['success2_2k']['acc']:.2f}",
        f"{R['success2_2k']['acc_single']:.2f}", f"{R['success2_2k']['acc_cross']:.2f}", "≥68 ✓ / ≥71 ✓"]],
      widths=[2.2, 3.4, 2.0, 1.6, 1.6, 3.6], size=9.5)
caption("표 1. 최종 결과 (800 instance, end-to-end 실행. 같은 설정을 다시 실행했을 때 800/800 예측과 토큰 수가 일치)")

# ---------------------------------------------------------------- 2
doc.add_heading("2. 방법", level=1)
doc.add_heading("2.1 시각 토큰 집계", level=2)
para("Qwen3-VL은 16×16 patch를 2×2로 병합하므로 visual token 1개가 32×32 px에 해당한다. 이미지 200장 중 136장이 4032×4032라서 "
     "원본 해상도로 넣으면 약 15,876 tokens가 필요하다. 즉 4K / 2K 예산은 원본의 26% / 13%에 불과하다. 모든 입력 이미지는 PIL Lanczos로 "
     "가로·세로를 32의 배수로 직접 리사이즈한 뒤 공식 processor에 넣고, processor가 반환한 image_grid_thw로 N = t·h·w/4 "
     "(<|image_pad|> 개수)를 실측했다. 한 instance 안에서 이미지를 다시 보내면 다시 세며, KV cache로 이미지를 재사용하지 않는다. "
     "데이터셋의 category·answer 컬럼은 채점과 분석에만 사용했다.")
doc.add_heading("2.2 추론 파이프라인", level=2)
fig("report/figs/pipeline.png", 17.0, "그림 1. 최종 파이프라인. 파란색 상자만 visual token을 소비한다.")
bullet(" 전역 이미지를 1,024 tokens(4032² → 1024² px)로 줄이고, '정답 글자 → 근거 bbox' 순서로 출력하게 한다. "
       "첫 생성 토큰의 A/B/C/D logit에 softmax를 취해 p1을 얻는다. max p1 ≥ τ이면 stopping criterion이 첫 토큰에서 바로 생성을 끝낸다. "
       "그렇지 않으면 같은 생성에서 이어 나온 bbox_2d(0–1000 상대좌표)를 쓰므로, 영역을 찾으려고 이미지를 다시 보낼 필요가 없다.",
       "Call 1 – Joint probe.")
bullet(" 확신이 낮을 때만 이미지 없이 질문과 보기만 보고, 'A: 한 물체/작은 영역의 세부'인지 'B: 여러 물체의 관계·개수·배치'인지 "
       "분류한다. 첫 토큰의 P(A) ≥ 0.3이면 A로 라우팅한다.", "Call 2 – 유형 분류 (0 visual tokens).")
bullet(" 유형 A: probe bbox를 2배 확장하고(최소 변 = 짧은 변의 25%, 약 1,008 px) 원본에서 잘라 ≤1,024 tokens로 입력한다. "
       "Success 1은 최대 2배 업스케일을 허용한다. 프롬프트 앞에는 'This image is a zoomed-in crop of a larger image.'를 붙인다. "
       "유형 B: Success 1은 전역 이미지를 4,096 tokens로 다시 보고, Success 2는 추가 호출 없이 probe 답을 유지한다.",
       "Call 3 – 유형별 추가 관찰.")
bullet(" 최종 답 = argmax(α·log p1 + log p_esc). 유형 A는 α = 0.25, 유형 B는 α = 0이다.", "결합.")
para("Joint probe 프롬프트: ", size=9.5).add_run(
    "\"{question}\\nA. …\\nB. …\\nC. …\\nD. …\\nAnswer with the option's letter from the given choices directly."
    + JOINT_SUFFIX.replace('"', '\\"') + "\"").font.size = Pt(8.5)
para("유형 분류 프롬프트: ", size=9.5).add_run(TYPE_PROMPT.replace("\n", " / ")).font.size = Pt(8.5)
table(["설정", "Probe", "τ", "유형 thr.", "유형 A (crop)", "유형 B", "α (A / B)"],
      [["Success 1", "1,024 tok", "0.995", "0.3", "≤1,024 tok, ×2 확장, 업스케일 ≤2", "전역 4,096 tok", "0.25 / 0"],
       ["Success 2", "1,024 tok", "0.999", "0.3", "≤1,024 tok, ×2 확장", "probe 답 유지", "0.25 / –"]],
      widths=[2.0, 1.8, 1.4, 1.6, 5.0, 2.6, 1.8], size=8.5)
caption("표 2. 최종 하이퍼파라미터 (configs/success1_4k.yaml, configs/success2_2k.yaml)")
para("설정은 다음과 같이 골랐다. 각 단계(probe, 여러 crop 변형, 전역 재관찰)를 800개 전체에 대해 조기 종료 없이 한 번씩 실행해 instance별 "
     "A–D 확률, bbox, 토큰 수를 저장했다. 그러면 τ·threshold·crop 변형·α 조합의 정확도와 평균 토큰을 모델을 다시 돌리지 않고 정확히 계산할 수 있다. "
     "고른 설정은 실제 파이프라인으로 처음부터 끝까지 다시 실행해 확인했다(오프라인 74.00% / 72.00% → 실제 74.00% / 71.88%).")

# ---------------------------------------------------------------- 3
doc.add_heading("3. 실험 환경", level=1)
para("실행당 RTX 3090 24GB 1장, Python 3.11, PyTorch 2.5.1(CUDA 12.4), transformers 4.57.1, bf16, SDPA, batch size 1을 사용했다. "
     "모든 호출은 Qwen/Qwen3-VL-2B-Instruct 원본 checkpoint와 공식 AutoProcessor를 쓴다. 디코딩은 greedy(do_sample=False, num_beams=1)이며 "
     "thinking 모드와 beam search는 쓰지 않는다. 선택형 답은 첫 디코딩 스텝의 A–D logit으로 정하므로 형식 오류가 없다. "
     "Latency는 이미지 디코딩·리사이즈를 포함해 instance 하나를 처리하는 전체 시간이다. 각 instance는 독립적으로 처리하며, "
     "같은 질문의 다른 보기 순서와 결과를 공유하지 않는다.")

# ---------------------------------------------------------------- 4
doc.add_heading("4. 결과", level=1)
rows = [run_row(f"Global resize {t}", R[f"global_{t}"]) for t in (1024, 2048, 4096)]
rows += [run_row("Global 512 + 2×2 tiles (384 tok each)", R["tiling_2048"]),
         run_row("Global 1024 + 2×2 tiles (768 tok each)", R["tiling_4096"]),
         run_row("Success 2 (ours)", R["success2_2k"]), run_row("Success 1 (ours)", R["success1_4k"])]
table(["방법", "평균 visual tok", "Acc", "Single", "Cross", "평균 호출"], rows,
      widths=[6.2, 2.6, 1.6, 1.6, 1.6, 1.8], size=8.5, bold_rows=(5, 6))
caption("표 3. Baseline 비교 (HR-Bench 4K, 800 instance. Single/Cross는 각 400 instance)")
para(f"전역 축소는 토큰을 두 배로 늘려도 3–4%p만 오르고, 4,096 tokens에서도 {R['global_4096']['acc']:.1f}%로 Main 목표(71%)에 못 미친다. "
     f"질문과 무관한 2×2 균등 tiling은 같은 예산에서 전역 축소보다 오히려 낮다({R['tiling_2048']['acc']:.1f}% / {R['tiling_4096']['acc']:.1f}%). "
     "타일 하나에 담기는 해상도가 크게 늘지 않는 데다, 장면이 쪼개져 cross 질문이 특히 손해를 본다. "
     f"제안 방법은 전역 2,048보다 적은 {R['success2_2k']['mean_visual']:.0f} tokens로 {R['success2_2k']['acc'] - R['global_2048']['acc']:+.1f}%p, "
     f"전역 4,096보다 적은 {R['success1_4k']['mean_visual']:.0f} tokens로 {R['success1_4k']['acc'] - R['global_4096']['acc']:+.1f}%p 높다. "
     f"참고로 원본 해상도(약 15,800 tok)는 54개 부분 실행에서 {S['native_partial']['acc']:.1f}%였다.")
fig("report/figs/acc_tokens.png", 12.0, "그림 2. 정확도–평균 visual token. 점선 곡선은 최종 정책에서 τ만 바꾼 결과(오프라인), 점선 격자는 예산·목표선.")

doc.add_heading("4.1 호출별 비용과 토큰 분포", level=2)
pc_rows = []
for name, F in [("S1", F1), ("S2", F2)]:
    for tag, lab in [("probe", "Call 1 joint probe"), ("type", "Call 2 유형 분류 (텍스트)"),
                     ("single", "Call 3 crop (유형 A)"), ("relational", "Call 3 전역 4096 (유형 B)")]:
        if tag in F["per_call"]:
            d = F["per_call"][tag]
            pc_rows.append([name, lab, d["count"], f1(d["mean_visual"]), f1(d["mean_input"] - d["mean_visual"]),
                            f"{d['mean_gen']:.1f}", f"{d['mean_latency']:.3f}"])
table(["설정", "호출", "횟수 (/800)", "visual tok", "text 입력 tok", "생성 tok", "latency (s)"], pc_rows,
      widths=[1.2, 5.0, 1.9, 1.9, 2.2, 1.7, 1.9], size=8.5)
caption("표 4. 호출 종류별 평균 비용 (text 입력 tok에는 vision start/end 특수 토큰 포함)")
para(f"instance당 평균 호출은 {R['success1_4k']['mean_calls']:.2f}회(S1) / {R['success2_2k']['mean_calls']:.2f}회(S2), "
     f"생성 토큰은 {F1['gen_tokens_mean']:.1f} / {F2['gen_tokens_mean']:.1f}개다. 생성 토큰은 대부분 확신이 낮을 때 출력되는 bbox JSON이다. "
     f"Latency는 평균 {F1['latency']['mean']:.2f}s, p95 {F1['latency']['p95']:.2f}s(S1) / 평균 {F2['latency']['mean']:.2f}s, p95 {F2['latency']['p95']:.2f}s(S2)다. "
     f"instance별 토큰은 1,024(조기 종료), 약 2,048(crop), 약 5,100(전역 재관찰)에 몰려 있다. 최대값은 {F1['tokens']['max']}(S1) / "
     f"{F2['tokens']['max']}(S2)이다(그림 3). 쉬운 문제에서 아낀 토큰을 어려운 문제에 쓰는 구조다.")
fig("report/figs/token_hist.png", 13.0, "그림 3. instance별 누적 visual token 분포 (검은 실선: 평균, 붉은 점선: 예산)")

# ---------------------------------------------------------------- 5
doc.add_heading("5. 결정적 설계 선택과 Ablation", level=1)
sep, full2 = A2["separate grounding call (+1024 tok)"], A2["Full (Success 2 config)"]
bullet(f" 같은 이미지에서 probe 확률이 0.999 이상인 {len(hi)}개는 정확도가 {hi_acc:.1f}%, 나머지 {len(lo)}개는 {lo_acc:.1f}%였다. "
       "확신도가 '추가 관찰이 필요한가'를 판단하는 신호로 충분히 쓸 만하다는 뜻이다.", "① 확신도는 잘 보정되어 있다.")
bullet(f" 처음에는 probe 다음에 grounding 호출을 따로 두었는데, 영역을 찾는 데만 1,024 tok가 들었다. "
       f"정답 글자 뒤에 bbox를 이어서 출력하게 바꾸자, 같은 정책에서 토큰이 {sep['tok'] - full2['tok']:.0f}개 줄고 "
       f"정확도는 {full2['acc'] - sep['acc']:+.2f}%p 올랐다.", "② Joint probe로 영역 탐색 비용을 없앴다.")
bullet(f" crop은 single 질문에서 오답 {eff['single'][0]}개를 고치고 정답 {eff['single'][1]}개만 틀리게 했다. 반면 cross 질문에서는 "
       f"{eff['cross'][0]}개를 고치는 대신 {eff['cross'][1]}개를 틀리게 했다. 상대 위치·개수·지도 비교처럼 장면 전체가 필요한 질문이기 때문이다. "
       "그래서 이미지를 쓰지 않는 유형 분류로 경로를 나누었다.", "③ 확대는 cross 질문에 해롭다.")
bullet(f" 4K 예산에서는 유형 B를 전역 4,096 tok로 다시 보게 해 cross를 {A1['relational: keep probe']['cross']:.1f} → "
       f"{A1['Full (Success 1 config)']['cross']:.1f}%로 올렸다.", "④ 유형 B는 해상도를 올려 다시 본다.")
abl = [abl_row("Success 2 (full)", full2),
       abl_row("  − 조기 종료 (τ = 1)", A2["w/o early stopping (tau=1)"]),
       abl_row("  − 유형 라우팅 (모두 crop)", A2["w/o type routing (zoom all)"]),
       abl_row("  − 추가 관찰 (joint probe만)", A2["w/o escalation (joint probe only)"]),
       abl_row("  joint 대신 별도 grounding 호출", sep),
       abl_row("Success 1 (full)", A1["Full (Success 1 config)"]),
       abl_row("  유형 B: 재관찰 없이 probe 유지", A1["relational: keep probe"]),
       abl_row("  유형 A: 업스케일 없음", A1["single: no 2x up-scaling"]),
       abl_row("  유형 A: crop 2,048 tok", A1["single: crop 2048 tok"])]
table(["설정 (한 요소만 변경)", "평균 tok", "Acc", "Single", "Cross"], abl,
      widths=[7.0, 2.2, 1.8, 1.8, 1.8], size=8.5, bold_rows=(0, 5))
caption("표 5. Controlled ablation. 같은 force-all 로그(같은 probe 출력, 같은 crop)에서 한 요소만 바꿔 계산했다.")
para(f"추가 관찰이 가장 크게 기여한다(probe만 쓰면 {A2['w/o escalation (joint probe only)']['acc']:.2f}%). "
     f"조기 종료는 정확도를 {A2['w/o early stopping (tau=1)']['acc'] - full2['acc']:.2f}%p만 잃고 토큰을 "
     f"{A2['w/o early stopping (tau=1)']['tok'] - full2['tok']:.0f}개 줄인다. "
     f"유형 라우팅은 2K에서 정확도를 유지하면서 토큰을 {A2['w/o type routing (zoom all)']['tok'] - full2['tok']:.0f}개 줄인다. "
     "crop 토큰을 2,048로 늘려도 정확도가 오르지 않아, crop은 1,024 tok로 충분했다.")

# ---------------------------------------------------------------- 6
doc.add_heading("6. 성공·실패 분석", level=1)
tx_keys = [("stopped_correct", "조기 종료 – 정답"), ("premature_stop(wrong & confident)", "조기 종료 – 오답 (premature stopping)"),
           ("escalated: fixed (probe wrong -> final right)", "추가 관찰로 수정 (probe 오답 → 최종 정답)"),
           ("escalated: kept correct", "추가 관찰 – 정답 유지"),
           ("escalated: harmful (probe right -> final wrong)", "불필요한 escalation (probe 정답 → 최종 오답)"),
           ("escalated: still wrong", "추가 관찰 후에도 오답")]
table(["유형", "Success 1", "Success 2"], [[lab, F1["taxonomy"].get(k, 0), F2["taxonomy"].get(k, 0)] for k, lab in tx_keys],
      widths=[9.0, 2.5, 2.5], size=8.5)
caption("표 6. 최종 실행의 instance 단위 결과 분류")
rv1 = F1["route_vs_category"]
para(f"조기 종료 오답은 {F1['taxonomy'].get('premature_stop(wrong & confident)', 0)}개(S1) / "
     f"{F2['taxonomy'].get('premature_stop(wrong & confident)', 0)}개(S2)로 드물다. 대표 사례는 책 제목 문제(#216)로, 확률 1.0으로 잘못 확신했다. "
     f"불필요한 escalation은 S2에서 {F2['taxonomy'].get('escalated: harmful (probe right -> final wrong)', 0)}개지만 S1에서는 "
     f"{F1['taxonomy'].get('escalated: harmful (probe right -> final wrong)', 0)}개다. S1은 τ가 낮아 escalation이 많고, 유형 B에서 probe 답을 결합하지 않기 때문이다. "
     f"유형 예측 실패(분석용 라벨 기준)는 S1에서 cross→A {rv1.get('cross|single-route', 0)}개, single→B {rv1.get('single|relational-route', 0)}개다. "
     "다만 cross→A의 상당수는 Mount Rushmore의 빨간 박스 속 인물 식별처럼 실제로 한 영역만 보면 되는 문제라 crop으로 맞혔다. "
     f"남은 오답은 주로 cross다. 특히 지도 위 번호 위치 비교 문항(48 instance)은 {QX['s1_map']}개(S1) / {QX['s2_map']}개(S2)만 맞혀 "
     f"무작위 수준 이하였고, 개수 세기 문항(40 instance)은 {QX['s1_cnt']} / {QX['s2_cnt']}개를 맞혔다.")
fig("report/figs/qual.png", 16.0, "그림 4. 대표 사례 (Success 2). 위: 전역 이미지, probe bbox(빨강), crop 영역(노랑 점선). 아래: Call 3에 입력된 crop.")
para("#252(번호판)와 #344(속도 제한 표지)는 1,024 tok 전역 이미지에서는 읽을 수 없던 글자를 원본 해상도 crop에서 읽어 정답으로 바뀐 사례다. "
     "#88은 판독 불가와 혼동이 겹친 실패다. bbox는 강변 벤치 옆 사람·자전거 부근을 가리켰지만 자전거가 원본에서도 수십 px라 식별이 어려웠고, "
     "모델은 옆 사람의 붉은 옷 색을 답했다. #148은 영역은 맞았지만 인식에 실패했다. 기둥에 흰색 카메라로 보이는 물체가 있는데도 "
     "'there is no Surveillance camera'를 골랐다. 단일 bbox에 의존하므로 첫 영역 선택이 틀리면 복구할 수 없다는 점도 한계다.")

# ---------------------------------------------------------------- 7
doc.add_heading("7. 한계 및 결론", level=1)
bullet(" τ, 유형 threshold, crop 설정, α는 별도 검증셋 없이 같은 800개에서 골랐고, 유형 분류 프롬프트도 category 라벨과 대조해 정했다. "
       "τ ∈ [0.99, 1] 구간에서 정확도 변화가 약 1%p라 선택에 크게 민감하지는 않지만, 수치가 다소 낙관적일 수 있다.", "테스트셋 튜닝.")
bullet(f" <|image_pad|>만 셌다. 이미지마다 붙는 vision start/end 2토큰을 포함해도 평균 이미지 수가 {imgs1:.2f}장(S1) / {imgs2:.2f}장(S2)이라 "
       f"+{2 * imgs1:.1f} / +{2 * imgs2:.1f} tokens에 그쳐 예산 안에 들어온다.", "토큰 집계.")
para("결론적으로, 같은 모델이 출력하는 정답 확신도와 근거 bbox를 저해상도 호출 한 번으로 얻고, 텍스트만으로 판단한 질문 유형에 따라 추가 관찰 방식을 "
     f"고르는 것만으로 전역 축소보다 적은 토큰에서 Success 1 {R['success1_4k']['acc']:.2f}%, Success 2 {R['success2_2k']['acc']:.2f}%를 달성했다. "
     "다음 개선 대상은 cross-instance 정확도이며, 특히 지도처럼 장면 전체를 추론해야 하는 문항이다.")

out = "report/project1_report_ko.docx"
doc.save(out)
print("saved", out)
