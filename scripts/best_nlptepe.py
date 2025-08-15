# Auto-generated from notebooks/best_nlptepe.ipynb on 2025-08-15 02:19 UTC

# This script contains the code cells from the notebook, in order.


# ===== Cell 1 =====
# !git clone https://github.com/rednote-hilab/dots.ocr.git
# %cd dots.ocr

# ===== Cell 2 =====
# !pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
# !pip install -e .

# ===== Cell 3 =====
# !python3 tools/download_model.py

# ===== Cell 4 =====
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt

model_path = "./weights/DotsOCR"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# ===== Cell 5 =====


image_path = "/content/sinav2.png"
prompt = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]

# Preparation for inference
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=24000)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

# ===== Cell 6 =====
import json

# OCR'dan gelen çıktının bir liste içinde olduğunu varsayalım.
# Eğer doğrudan string geliyorsa output_text[0] yerine output_text kullanın.
try:
    # 1. JSON dizesini bir Python listesine/sözlüğüne dönüştür
    # Not: DotsOCR modeli bazen "```json" gibi işaretçilerle başlayabilir, bunları temizliyoruz.
    clean_json_string = output_text[0].replace("```json\n", "").replace("\n```", "")
    layout_data = json.loads(clean_json_string)

    # 2. Tüm metin içeriklerini okuma sırasına göre birleştir
    # Modelin çıktısı zaten okuma sırasına göre sıralanmış olmalı.
    ogrenci_cevabi_parcalari = []
    for element in layout_data:
        # 'Picture' gibi metin içermeyen kategorileri atla [cite: 33]
        if 'text' in element and element['text']:
            ogrenci_cevabi_parcalari.append(element['text'])

    # 3. Tüm metin parçalarını tek bir metin haline getir
    ogrenci_cevabi_tamami = "\n".join(ogrenci_cevabi_parcalari)

    print("--- OCR'DAN AYIKLANAN ÖĞRENCİ CEVABI ---")
    print(ogrenci_cevabi_tamami)
    print("-" * 40)

except json.JSONDecodeError as e:
    print(f"Hata: OCR çıktısı geçerli bir JSON formatında değil. Hata: {e}")
    print("Alınan Çıktı:", output_text[0])
    ogrenci_cevabi_tamami = "" # Hata durumunda metni boş bırak

# ÇIKTIDA MATEMATİKSEL SEMBOLLERE ÖZEL TERİMLER VAR, OCR DOĞRU ÇALIŞIYOR,
# SADECE ONLAR MATEMATİKSEL GÖSTERİMLER İÇİN ÖRNEK: $\mathbb{R}_2$

# ===== Cell 7 =====
import json, re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

# ============ 1) Robust OCR output parser ============

def coerce_layout(output_text) -> List[Dict[str, Any]]:
    """
    Accepts:
      - raw string
      - list[str] (e.g. ['[...]'])
      - list[dict] (already parsed)
      - code-fenced variations
    Returns: list of dicts with keys like 'bbox', 'category', 'text'
    """
    def _json_try(s: str):
        s = s.strip()
        # strip code fences ```json ... ```
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        # If the model returned extra text, try to extract the first JSON array
        m = re.search(r"\[\s*{.*}\s*\]", s, flags=re.DOTALL)
        if m:
            s = m.group(0)
        return json.loads(s)

    # Already a list of dicts?
    if isinstance(output_text, list) and output_text and isinstance(output_text[0], dict):
        return output_text

    # List with a single string?
    if isinstance(output_text, list) and len(output_text) == 1 and isinstance(output_text[0], str):
        try:
            return _json_try(output_text[0])
        except json.JSONDecodeError:
            pass  # fall through

    # Bare string?
    if isinstance(output_text, str):
        try:
            return _json_try(output_text)
        except json.JSONDecodeError:
            pass

    # Last resort: try json on each element if it's a list of strings
    if isinstance(output_text, list) and all(isinstance(x, str) for x in output_text):
        for s in output_text:
            try:
                return _json_try(s)
            except json.JSONDecodeError:
                continue

    # Could not parse → return empty
    return []

# ============ 2) Q/A extractor (supports subparts like (a), (b)) ============

PART_RE = re.compile(r"^\(?([a-z])\)?(\.|$|\s)", re.IGNORECASE)
QUESTION_START_RE = re.compile(r"(?:^|\s)Soru\s*(\d+)\b", re.IGNORECASE)
NUMBER_HEAD_RE = re.compile(r"^(\d+)[\).]")  # e.g. "1)" or "1."

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def extract_qas(layout_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts layout elements into a list of:
      { qid: "1a", number: 1, part: "a", question: "...", answer: "..." }
    Handles patterns like:
      - "Soru 1. ..."
      - "(a) (8 puan)" / "a." / "a)"
      - "Cevap:" / "Büyüme Düzeni:"
    """
    if not layout_data:
        return []

    # Sort by reading order just in case
    def key_bbox(e):
        b = e.get("bbox", [0, 0, 0, 0])
        return (b[1], b[0])  # y1, x1

    layout_sorted = sorted(layout_data, key=key_bbox)

    qas = []
    curr_qnum = None
    curr_part = None
    curr_qtext_parts = []
    curr_ans_parts = []
    have_answer_started = False

    def flush():
        nonlocal curr_qnum, curr_part, curr_qtext_parts, curr_ans_parts, have_answer_started
        if curr_qnum is None or curr_part is None:
            # nothing to flush
            curr_qtext_parts, curr_ans_parts = [], []
            have_answer_started = False
            return
        qid = f"{curr_qnum}{curr_part}"
        qtext = normalize_text("\n".join(curr_qtext_parts))
        ans = normalize_text("\n".join(curr_ans_parts))
        if qtext or ans:
            qas.append({
                "qid": qid,
                "number": curr_qnum,
                "part": curr_part,
                "question": qtext,
                "answer": ans
            })
        # reset for next part
        curr_part = None
        curr_qtext_parts = []
        curr_ans_parts = []
        have_answer_started = False

    for el in layout_sorted:
        txt = el.get("text", "")
        if not txt:
            continue
        line = txt.strip()

        # Detect "Soru X"
        m_q = QUESTION_START_RE.search(line)
        if m_q:
            # New question block → flush any previous part
            flush()
            curr_qnum = int(m_q.group(1))
            curr_part = None
            curr_qtext_parts = [line]
            curr_ans_parts = []
            have_answer_started = False
            continue

        # Detect explicit numeric start "1." if "Soru" wasn't present
        if curr_qnum is None:
            m_head = NUMBER_HEAD_RE.match(line)
            if m_head:
                flush()
                curr_qnum = int(m_head.group(1))
                curr_part = None
                curr_qtext_parts = [line]
                curr_ans_parts = []
                have_answer_started = False
                continue

        # Detect subpart markers like "(a)", "a)", "a."
        m_p = PART_RE.match(line)
        if m_p:
            # Starting a new part → flush previous part
            if curr_part is not None:
                flush()
            curr_part = m_p.group(1).lower()
            curr_qtext_parts = [line]
            curr_ans_parts = []
            have_answer_started = False
            continue

        # If a question is active but no part yet, keep appending to the question text
        # until we see a part marker.
        if curr_qnum is not None and curr_part is None:
            curr_qtext_parts.append(line)
            continue

        # Answer triggers
        is_answer_trigger = (
            re.search(r"\bcevap\b", line, re.IGNORECASE) or
            re.search(r"büyüme\s*düzeni", line, re.IGNORECASE) or
            re.search(r"\byanıt\b", line, re.IGNORECASE)
        )

        if curr_qnum is not None and curr_part is not None:
            if is_answer_trigger:
                have_answer_started = True
                # Normalize common prefixes
                line_clean = re.sub(r"(?i)^(a\.\s*)?cevap\s*:?\s*", "", line)
                line_clean = re.sub(r"(?i)^büyüme\s*düzeni\s*:?\s*", "", line_clean)
                curr_ans_parts.append(line_clean.strip())
            else:
                # Before answer starts → keep building the subpart question
                if not have_answer_started:
                    curr_qtext_parts.append(line)
                else:
                    # After answer started → multi-line answer continuation
                    curr_ans_parts.append(line)

    # Flush last part
    if curr_qnum is not None and curr_part is not None:
        flush()

    return qas

# ============ 3) Grading (safe against empty results) ============

def grade_qas(qas: List[Dict[str, Any]], answer_key_dict: Dict[str, str]):
    """
    qas: output of extract_qas()
    answer_key_dict: keys like "1a", "1b"
    Returns: (results_list, overall_average)
    """
    # Pick a multilingual ST model to be robust in TR/EN
    st_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    results = []
    for qa in qas:
        qid = qa["qid"]
        student_answer = qa.get("answer", "")
        key_answer = answer_key_dict.get(qid, "")
        if not student_answer or not key_answer:
            score = 0.0
        else:
            emb_s = st_model.encode(student_answer, convert_to_tensor=True)
            emb_k = st_model.encode(key_answer, convert_to_tensor=True)
            score = float(util.cos_sim(emb_s, emb_k).item() * 100.0)

        results.append({
            "qid": qid,
            "question": qa.get("question", ""),
            "student_answer": student_answer,
            "key_answer": key_answer,
            "score": score
        })

    overall = (sum(r["score"] for r in results) / len(results)) if results else 0.0
    return results, overall

# ============ 4) Example usage with your sample OCR output ============

layout = coerce_layout(output_text)
qas = extract_qas(layout)

# Fill your answer key here (keys like "1a", "1b")
# (These are examples; replace with your ground truth.)
answer_key_dict = {
    "1a": "O(N²)",
    "1b": "O(N*log(N))"
}

results, overall = grade_qas(qas, answer_key_dict)

print("--- PARSED Q/A ---")
for qa in qas:
    print(f"{qa['qid']}:")
    print("Q:", qa['question'])
    print("A:", qa['answer'])
    print()

print("--- GRADES ---")
correct_ans = 0
for r in results:
    print(f"{r['qid']}: {r['score']:.2f}/100")
    if r['score'] >= 85:
        correct_ans += 1
print(f"Overall: {overall:.2f}/100")
print(f"Correct answers: {correct_ans}/{len(results)}")
res = correct_ans / len(results) * 100
print("Final Grade" , res)


if __name__ == '__main__':
    print('Running notebook code as a script...')
    # If your notebook defines a main() function, call it here.
    # Otherwise, consider structuring your code into functions for reusability.