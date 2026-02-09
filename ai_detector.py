import os
import json
import re
import requests
from dotenv import load_dotenv

from PyQt5.QtWidgets import (
    QApplication, QWidget, QTextEdit, QPushButton,
    QLabel, QVBoxLayout, QHBoxLayout, QProgressBar, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import html
import math

import re
import statistics

from collections import Counter
import spacy


# phrases commonly inserted by "humanizer" prompts (tune as needed)
HUMANIZER_PATTERNS = [
    r"\bi think\b", r"\bmaybe\b", r"\bnot sure\b", r"\bto some extent\b",
    r"\bin many cases\b", r"\bin many areas\b", r"\bin practice\b",
    r"\bnot sure but\b"
]


STOPWORDS = {
    "the","and","a","an","in","on","at","for","of","to",
    "is","are","was","were","it","that","this","with","as"
}



# ----- OPTIONAL ADVANCED SIGNALS (add after imports) -----
USE_ADVANCED = True  # flip to False to skip heavy checks

# try optional heavy deps
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    GPT2_AVAILABLE = True
except Exception:
    GPT2_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
        SPACY_AVAILABLE = False
except Exception:
    SPACY_AVAILABLE = False

# lazy-loaded models
_gpt2_tokenizer = None
_gpt2_model = None
_sbert_model = None

def ensure_gpt2():
    global _gpt2_tokenizer, _gpt2_model
    if not GPT2_AVAILABLE:
        return False
    if _gpt2_tokenizer is None:
        _gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
        _gpt2_model.eval()
        if torch.cuda.is_available():
            _gpt2_model.to("cuda")
    return True

def compute_surprisal_gpt2(text: str, max_len_tokens=1024):
    """Return normalized surprisal score 0..1 where higher => more AI-like (optional)."""
    if not GPT2_AVAILABLE or not ensure_gpt2():
        return None
    tok = _gpt2_tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len_tokens)
    input_ids = tok["input_ids"]
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    with torch.no_grad():
        outputs = _gpt2_model(input_ids, labels=input_ids)
        # outputs.loss is mean negative log-likelihood per token (cross-entropy)
        # multiply by ln(2)? no; keep as-is. Typical values: 2..6
        neg_log_likelihood = float(outputs.loss)  # average NLL
    # map typical range roughly [1.5, 6.0] to 0..1
    v = (neg_log_likelihood - 2.5) / 4.0
    return max(0.0, min(1.0, v))

def ensure_sbert():
    global _sbert_model
    if not SBERT_AVAILABLE:
        return False
    if _sbert_model is None:
        _sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return True

def compute_avg_sentence_similarity(text: str):
    """Return average pairwise cosine similarity between sentence embeddings (0..1)."""
    if not SBERT_AVAILABLE or not ensure_sbert():
        return None
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(sents) <= 1:
        return 0.0
    embs = _sbert_model.encode(sents, convert_to_tensor=True)
    import torch
    sims = (embs @ embs.T).cpu().numpy()
    # take upper triangle excluding diagonal
    n = len(sents)
    vals = []
    for i in range(n):
        for j in range(i+1, n):
            vals.append(float(sims[i,j]))
    if not vals:
        return 0.0
    return sum(vals) / len(vals)

def compute_pos_entropy(text: str):
    """Return normalized POS entropy (0..1) if spacy available. low entropy -> suspicious."""
    if not SPACY_AVAILABLE or nlp is None:
        return None
    doc = nlp(text[:20000])  # limit length
    pos_tags = [tok.pos_ for tok in doc if tok.is_alpha]
    if not pos_tags:
        return None
    freq = Counter(pos_tags)
    total = sum(freq.values())
    import math
    ent = -sum((c/total) * math.log((c/total)+1e-12) for c in freq.values())
    # normalize by log(Npos)
    max_ent = math.log(len(freq)+1e-12)
    if max_ent <= 0:
        return 0.0
    return ent / max_ent

# optional: small utility to compute debug info
def compute_advanced_features(text: str):
    """Return dict with optional features (some may be None if libs missing)."""
    feats = {}
    if USE_ADVANCED and GPT2_AVAILABLE:
        try:
            feats['surprisal_gpt2'] = compute_surprisal_gpt2(text)
        except Exception:
            feats['surprisal_gpt2'] = None
    else:
        feats['surprisal_gpt2'] = None

    if USE_ADVANCED and SBERT_AVAILABLE:
        try:
            feats['avg_sent_sim'] = compute_avg_sentence_similarity(text)
        except Exception:
            feats['avg_sent_sim'] = None
    else:
        feats['avg_sent_sim'] = None

    if USE_ADVANCED and SPACY_AVAILABLE:
        try:
            feats['pos_entropy'] = compute_pos_entropy(text)
        except Exception:
            feats['pos_entropy'] = None
    else:
        feats['pos_entropy'] = None

    return feats



def compute_heuristic_scores(text: str):
    """
    Enhanced heuristic detector (calibrated).
    Returns integer percentages (ai, human, mixed)
    and exposes extract_segments().
    """

    # ---------------- helpers ----------------
    def safe_words(s):
        return re.findall(r"\w+", s.lower())

    def paragraph_lowercase_pattern(paragraph: str) -> bool:
        p = paragraph.strip()
        if len(p) < 2:
            return False
        letters = re.findall(r"[A-Za-z]", p)
        if not letters:
            return False
        first_alpha = None
        for i, ch in enumerate(p):
            if ch.isalpha():
                first_alpha = i
                break
        if first_alpha is None:
            return False
        rest = p[first_alpha+1:]
        return rest.lower() == rest

    def score_segment(seg_text: str):
        s_words = safe_words(seg_text)
        wc = max(1, len(s_words))
        sents = [s for s in re.split(r'[.!?]+\s+', seg_text) if s.strip()]
        lens = [len(s.split()) for s in sents] or [1]
        avg = sum(lens) / len(lens)
        std = statistics.pstdev(lens) if len(lens) > 1 else 0.0

        ttr = len(set(s_words)) / wc
        freq = Counter(s_words)
        maxfreq = freq.most_common(1)[0][1] / wc if freq else 0.0
        stop_ratio = len([w for w in s_words if w in STOPWORDS]) / wc

        uniform = 1.0 - min(1.0, std / (avg + 1.0))
        lex_rep = 1.0 - min(1.0, ttr / 0.65)
        repetition = min(1.0, maxfreq * 2.0)
        func_words = min(1.0, stop_ratio / 0.6)
        punct_density = len(re.findall(r"[,:;]", seg_text)) / wc
        punct_score = 1.0 - min(1.0, punct_density * 5.0)

        hp_hits = sum(len(re.findall(p, seg_text, flags=re.I)) for p in HUMANIZER_PATTERNS)
        hp_factor = min(1.0, hp_hits / 4.0)

        seg_ai = (
            0.30 * uniform +
            0.25 * lex_rep +
            0.15 * repetition +
            0.15 * (0.5 * func_words + 0.5 * punct_score) +
            - 0.15 * hp_factor

        )
        return max(0.0, min(1.0, seg_ai))

    # ---------------- main doc ----------------
    text = text.strip()
    if not text:
        return (0, 0, 0)

    paragraphs = [p for p in re.split(r'\n{1,}', text) if p.strip()]
    sentences = [s for s in re.split(r'[.!?]+\s+', text) if s.strip()]
    words = safe_words(text)
    wc = max(1, len(words))

    sent_lens = [len(s.split()) for s in sentences] or [1]
    avg_len = sum(sent_lens) / len(sent_lens)
    std_len = statistics.pstdev(sent_lens) if len(sent_lens) > 1 else 0.0
    uniformity = 1.0 - min(1.0, std_len / (avg_len + 1.0))

    ttr = len(set(words)) / wc
    freq = Counter(words)
    max_freq = freq.most_common(1)[0][1] / wc if freq else 0.0
    hapax_ratio = sum(1 for _, c in freq.items() if c == 1) / max(1, len(freq))

    stop_ratio = len([w for w in words if w in STOPWORDS]) / wc
    punct_density = len(re.findall(r"[,:;]", text)) / wc

    cap_hits = sum(1 for p in paragraphs if paragraph_lowercase_pattern(p))
    cap_ratio = cap_hits / max(1, len(paragraphs))

    humanizer_hits = sum(len(re.findall(p, text, flags=re.I)) for p in HUMANIZER_PATTERNS)
    humanizer_factor = min(1.0, humanizer_hits / 6.0)

    # ---- normalize ----
    lex_rep = 1.0 - min(1.0, ttr / 0.65)
    repetition = min(1.0, max_freq * 2.0)
    func_words = min(1.0, stop_ratio / 0.6)
    punct = 1.0 - min(1.0, punct_density * 5.0)

    # ---- CRITICAL CALIBRATION CHANGE ----
    # If lexical diversity is high AND hapax is high,
    # strongly bias toward human (this is what QuillBot does)

    # ---- final AI score ----
    ai_score = (
        0.22 * uniformity +
        0.22 * lex_rep +
        0.14 * repetition +
        0.14 * (0.5 * func_words + 0.5 * punct) +
        0.14 * cap_ratio +
        - 0.14 * humanizer_factor
    )


    # ---- LLM STRUCTURE GATE ----
    llm_structure_score = 0.0

    # paragraph length symmetry (LLMs are too balanced)
    para_lens = [len(p.split()) for p in paragraphs]
    if len(para_lens) >= 3:
        if statistics.pstdev(para_lens) < 18:
            llm_structure_score += 0.4

    # sentence length symmetry
    if std_len < 6.5:
        llm_structure_score += 0.3

    # polished lexical balance
    if ttr > 0.50 and hapax_ratio < 0.55:
        llm_structure_score += 0.3

    llm_structure_score = min(1.0, llm_structure_score)

    # hard floor for strong LLM structure
    if llm_structure_score > 0.45:
        ai_score = max(ai_score, 0.75)


    # ---- ESL ACADEMIC WRITING OVERRIDE ----
    # If text shows consistent academic structure but high lexical diversity,
    # and no burst-style generation, treat as human (non-native)

    # ---- GPT2 Perplexity Boost ----
    adv = compute_advanced_features(text)

    if adv.get("surprisal_gpt2") is not None:
        ai_score = (ai_score * 0.7) + (adv["surprisal_gpt2"] * 0.3)



    ai_score = max(0.0, min(1.0, ai_score))
    human_score = 1.0 - ai_score

    # ---- mixed logic (stable) ----
    mixed_score = 0.0

    if 0.30 < ttr < 0.55 and uniformity > 0.45:
        mixed_score += 0.25

    mixed_score += humanizer_factor * 0.25
    mixed_score = min(0.45, mixed_score)

    # renormalize
    ai_score = max(0.0, ai_score - mixed_score * 0.6)
    human_score = max(0.0, 1.0 - ai_score - mixed_score)

    total = ai_score + human_score + mixed_score
    ai_score /= total
    human_score /= total
    mixed_score /= total

    ai_pct = int(round(ai_score * 100))
    human_pct = int(round(human_score * 100))
    mixed_pct = 100 - (ai_pct + human_pct)

    # normalize
    total = ai_pct + human_pct + mixed_pct
    if total != 100:
        diff = 100 - total
        if human_pct > 0:
            human_pct += diff
        else:
            ai_pct += diff

    # ---- segment extractor ----
    def extract_ai_like_segments(n=3, win=3, th=0.65):
        sents = [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        hits = []
        for i in range(len(sents)):
            chunk = " ".join(sents[i:i+win])
            if not chunk.strip():
                continue
            sc = score_segment(chunk)
            if sc >= th:
                hits.append((sc, chunk))
        dedup = {}
        for sc, ch in sorted(hits, key=lambda x: x[0], reverse=True):
            if ch not in dedup:
                dedup[ch] = sc
        return list(dedup.keys())[:n]

    class _R(tuple):
        pass

    r = _R((ai_pct, human_pct, mixed_pct))
    setattr(r, "extract_segments", extract_ai_like_segments)
    return r



# ------------------ CONFIG ------------------
load_dotenv()
API_KEY = os.getenv('OR_Key')  # make sure .env contains OR_Key=sk-xxx
MODEL = 'meta-llama/llama-3.3-70b-instruct:free'
API_URL = 'https://openrouter.ai/api/v1/chat/completions'

# ------------------ UTIL: Robust API parse ------------------
def extract_json_from_text(text: str):
    """Extract the first JSON object from a text blob and return parsed JSON.
    Handles code fences and common trailing commas. Raises ValueError if not found."""
    # remove markdown code fences if present
    text = re.sub(r'```(?:json)?\n', '', text, flags=re.IGNORECASE)
    text = text.replace('```', '')

    # non-greedy JSON match (first {...})
    match = re.search(r'\{[\s\S]*?\}', text)
    if not match:
        raise ValueError('No JSON object found in response')

    json_text = match.group()

    # attempt to fix common JSON problems (trailing commas)
    json_text = re.sub(r',\s*}', '}', json_text)
    json_text = re.sub(r',\s*\]', ']', json_text)

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        # as a fallback, try to extract key:value pairs heuristically
        raise ValueError(f'Invalid JSON: {e}')

def parse_percent_value(val):
    """Accept numbers, '70%', 'approx 70', 'seventy' (limited). Return float or 0."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower()
    # remove percent sign and words
    s = s.replace('%', '')
    s = re.sub(r'[^0-9\.\-]+', ' ', s).strip()
    if not s:
        return 0.0
    # take first numeric token
    try:
        token = s.split()[0]
        return float(token)
    except:
        return 0.0

def normalize_percentages(ai, human, mixed):
    """Turn arbitrary ai/human/mixed inputs into integer percentages summing to 100."""
    ai_f = parse_percent_value(ai)
    human_f = parse_percent_value(human)
    mixed_f = parse_percent_value(mixed)

    vals = [max(0.0, ai_f), max(0.0, human_f), max(0.0, mixed_f)]
    total = sum(vals)

    # if model gave percentages that already sum ~100, just round
    if total > 0:
        # convert to exact integer percentages while preserving proportions
        # compute floor ints then distribute remainder by largest fractional parts
        raw = [(v / total) * 100.0 for v in vals]
        floors = [int(math.floor(x)) for x in raw]
        remainder = 100 - sum(floors)
        # distribute remainder to highest fractional parts
        fracs = sorted(enumerate(raw), key=lambda x: x[1] - math.floor(x[1]), reverse=True)
        for i in range(remainder):
            idx = fracs[i % 3][0]
            floors[idx] += 1
        return floors[0], floors[1], floors[2]
    else:
        # nothing provided -> return zeros
        return 0, 0, 0


# ------------------ AI ANALYSIS (synchronous, used inside thread) ------------------
def analyze_text_api(text: str, retries: int = 2, timeout: int = 60):
    # Strong, explicit system + user message prompting strict JSON output
    system_msg = {
        "role": "system",
        "content": "You are an objective AI forensic text analyst. Output only JSON with numeric percentages that sum to 100."
    }

    user_prompt = f"""
TASK:
Return a strict JSON object with these keys (all numeric):
  - ai_percent
  - human_percent
  - mixed_percent
  - ai_like_segments (array of short strings)

Ensure percentages sum to roughly 100. Provide numeric values only (no % sign).
TEXT:
{text}
"""

    payload = {
        'model': MODEL,
        'messages': [system_msg, {'role': 'user', 'content': user_prompt}],
        'temperature': 0.0,
        'max_tokens': 512
    }

    last_err = None
    for attempt in range(1, retries + 2):
        try:
            resp = requests.post(API_URL, headers={
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json'
            }, json=payload, timeout=timeout)
            resp.raise_for_status()
            j = resp.json()
            content = j.get('choices', [{}])[0].get('message', {}).get('content', '')
            parsed = extract_json_from_text(content)

            ai_raw = parsed.get('ai_percent', parsed.get('ai', parsed.get('ai_percent', 0)))
            human_raw = parsed.get('human_percent', parsed.get('human', parsed.get('human_percent', 0)))
            mixed_raw = parsed.get('mixed_percent', parsed.get('mixed', parsed.get('mixed_percent', 0)))
            segments = parsed.get('ai_like_segments') or parsed.get('ai_like', []) or []

            ai_i, human_i, mixed_i = normalize_percentages(ai_raw, human_raw, mixed_raw)

            return {
                'ai': ai_i,
                'human': human_i,
                'mixed': mixed_i,
                'segments': segments if isinstance(segments, list) else [str(segments)]
            }

        except Exception as e:
            last_err = e
            if attempt < retries + 1:
                continue
            else:
                # final fallback: return an error to UI and let UI use local heuristics
                raise last_err


# ------------------ WORKER THREAD ------------------
class AnalyzeWorker(QThread):
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def run(self):
        try:
            result = analyze_text_api(self.text)
        except Exception as e:
            # fallback to heuristics if API fails
            r = compute_heuristic_scores(self.text)
            ai_h, human_h, mixed_h = int(r[0]), int(r[1]), int(r[2])
            segments = r.extract_segments()
            result = {'ai': ai_h, 'human': human_h, 'mixed': mixed_h, 'segments': segments}
        else:
            # if API returned, also compute heuristics and blend them
            r = compute_heuristic_scores(self.text)
            ai_h, human_h, mixed_h = int(r[0]), int(r[1]), int(r[2])
            segments = r.extract_segments()
            # weights: api 0.7, heuristics 0.3 (tune as needed)
            weight_api = 0.4
            weight_heur = 0.6
            ai_blend = int(round(result.get('ai', 0) * weight_api + ai_h * weight_heur))
            human_blend = int(round(result.get('human', 0) * weight_api + human_h * weight_heur))
            mixed_blend = int(round(result.get('mixed', 0) * weight_api + mixed_h * weight_heur))
            # normalize to 100
            s = ai_blend + human_blend + mixed_blend
            if s == 0:
                ai_blend, human_blend, mixed_blend = 0, 0, 0
            else:
                ai_blend = int(round(ai_blend * 100.0 / s))
                human_blend = int(round(human_blend * 100.0 / s))
                mixed_blend = 100 - (ai_blend + human_blend)
            result = {'ai': ai_blend, 'human': human_blend, 'mixed': mixed_blend, 'segments': result.get('segments', [])}
        self.finished_signal.emit(result)


# ------------------ THEMES ------------------
THEMES = {
    'Dark': {
        'stylesheet': '''
            QWidget { background-color: #0f172a; color: #e5e7eb; }
            QTextEdit { background-color: #020617; border-radius: 10px; padding: 10px; }
            QPushButton { background-color: #2563eb; border-radius: 8px; padding: 8px; }
        '''
    },
    'Light': {
        'stylesheet': '''
            QWidget { background-color: #f3f4f6; color: #0f172a; }
            QTextEdit { background-color: #ffffff; border-radius: 8px; padding: 8px; }
            QPushButton { background-color: #2563eb; color: white; border-radius: 6px; padding: 8px; }
        '''
    },
    'Neon': {
        'stylesheet': '''
            QWidget { background-color: #020617; color: #e6fffa; }
            QTextEdit { background-color: #001219; border-radius: 10px; padding: 10px; }
            QPushButton { background-color: #7c3aed; color: white; border-radius: 8px; padding: 8px; }
        '''
    }
}

# ------------------ GUI ------------------
class DetectorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AI Text Detector — Threaded')
        self.setMinimumSize(900, 650)
        self.worker = None
        self.init_ui()
        self.apply_theme('Dark')

    def init_ui(self):
        title = QLabel('AI Text Detection System')
        title.setFont(QFont('Segoe UI', 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)

        # input
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText('Paste text here...')

        # controls
        self.detect_btn = QPushButton('Analyze Text')
        self.detect_btn.clicked.connect(self.on_detect_clicked)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(THEMES.keys()))
        self.theme_combo.currentTextChanged.connect(self.apply_theme)

        # status/progress (IMPORTANT: progress is added to layout so it doesn't become a top-level window)
        self.status_label = QLabel('')
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setFixedWidth(160)          # small inline progress
        self.progress.setMaximumHeight(18)

        # results
        self.ai_label = QLabel('AI Generated: 0%')
        self.human_label = QLabel('Human Written: 0%')
        self.mixed_label = QLabel('Mixed: 0%')

        self.ai_bar = QProgressBar()
        self.human_bar = QProgressBar()
        self.mixed_bar = QProgressBar()

        self.highlight_box = QTextEdit()
        self.highlight_box.setReadOnly(True)
        self.highlight_box.setPlaceholderText('AI-like text segments will appear here...')

        # layout
        top_controls = QHBoxLayout()
        top_controls.addWidget(self.detect_btn)
        top_controls.addWidget(self.theme_combo)
        top_controls.addStretch()
        top_controls.addWidget(self.status_label)
        top_controls.addWidget(self.progress)   # <<-- added to layout (no extra window)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.text_input)
        layout.addLayout(top_controls)

        layout.addWidget(self.ai_label)
        layout.addWidget(self.ai_bar)
        layout.addWidget(self.human_label)
        layout.addWidget(self.human_bar)
        layout.addWidget(self.mixed_label)
        layout.addWidget(self.mixed_bar)

        layout.addWidget(QLabel('AI-Generated Looking Segments:'))
        layout.addWidget(self.highlight_box)

        self.setLayout(layout)

    def apply_theme(self, theme_name: str):
        theme = THEMES.get(theme_name, THEMES['Dark'])['stylesheet']
        self.setStyleSheet(theme)

    def set_busy(self, busy: bool):
        if busy:
            self.progress.setRange(0, 0)  # busy indicator (indeterminate)
            self.progress.setVisible(True)
            self.status_label.setText('Analyzing...')
            self.detect_btn.setEnabled(False)
        else:
            self.progress.setRange(0, 100)
            self.progress.setVisible(False)
            self.status_label.setText('')
            self.detect_btn.setEnabled(True)

    def on_detect_clicked(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            self.status_label.setText('Paste some text first.')
            return

        self.set_busy(True)
        # start worker thread
        self.worker = AnalyzeWorker(text)
        self.worker.finished_signal.connect(self.on_result)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_result(self, result: dict):
        self.set_busy(False)
        ai = int(result.get('ai', 0))
        human = int(result.get('human', 0))
        mixed = int(result.get('mixed', 0))
        segments = result.get('segments', [])

        self.ai_bar.setValue(ai)
        self.human_bar.setValue(human)
        self.mixed_bar.setValue(mixed)

        self.ai_label.setText(f'AI Generated: {ai}%')
        self.human_label.setText(f'Human Written: {human}%')
        self.mixed_label.setText(f'Mixed: {mixed}%')

        if segments:
            self.highlight_box.setPlainText('\n\n'.join(segments))
        else:
            self.highlight_box.setPlainText('(no clear AI-like segments detected)')

        # highlight segments inside the main text preview (HTML)
        highlighted_html = html.escape(self.text_input.toPlainText())
        for seg in segments:
            esc = html.escape(seg)
            # safe replace only first occurrence to avoid global mismatches
            highlighted_html = highlighted_html.replace(esc, f"<span style='background-color: #ffd54f'>{esc}</span>", 1)

        self.highlight_box.setHtml(highlighted_html)

    def on_error(self, error_msg: str):
        self.set_busy(False)
        self.highlight_box.setPlainText(f'Error: {error_msg}')

# ------------------ RUN ------------------
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    ui = DetectorUI()
    ui.show()
    sys.exit(app.exec_())
