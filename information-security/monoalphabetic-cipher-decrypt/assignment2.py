import math, random, argparse, sys, re
from pathlib import Path
from collections import Counter, defaultdict

#알파벳
Alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NOT_ALPHA = re.compile(r'[^A-Z]')
UPPER_WORD = re.compile(r"[A-Z']+")

#켓;시
_clean_cache = {}
def clean_text(s):
    if s in _clean_cache:
        return _clean_cache[s]
    result = NOT_ALPHA.sub('', s.upper())
    if len(_clean_cache) < 50000:
        _clean_cache[s] = result
    return result

def read_file(path):
    return Path(path).read_text(encoding="utf-8")

#키 적용
def substitution_key(text, key_dict):
    result = []
    for c in text:
        if c.isalpha():
            upper_c = c.upper()
            new_c = key_dict.get(upper_c, upper_c)
            result.append(new_c.lower() if c.islower() else new_c)
        else:
            result.append(c)
    return "".join(result)

# 키 완성시키기 (26글자 다 채우기)
def complete_key(partial_key):
    used_values = set(partial_key.values())
    available = []
    for c in Alphabet:
        if c not in used_values:
            available.append(c)
    
#빈부분 매우기
    for c in Alphabet:
        if c not in partial_key:
            if available:
                partial_key[c] = available.pop()
            else:
                partial_key[c] = 'E'  #없을 때 기본값
    
    #중복제거
    seen = set()
    duplicate = []
    for c in Alphabet:
        v = partial_key[c]
        if v in seen:
            duplicate.append(c)
        else:
            seen.add(v)
    
    missing = []
    for c in Alphabet:
        if c not in seen:
            missing.append(c)
    
    for dup_key, miss_val in zip(duplicate, missing):
        partial_key[dup_key] = miss_val
    
    return partial_key

# 빈도수 기반으로 초기 키 생성
def make_initial_key(ciphertext, rdm):
    cleaned = clean_text(ciphertext)
    freq = Counter(cleaned)
    cipher_order = [x for x, _ in freq.most_common()]
    
    # 영어 빈도
    eng_freq = {'E':12.0,'T':9.1,'A':8.2,'O':7.5,'I':7.0,'N':6.7,'S':6.3,'R':6.0,'H':6.1,
                'L':4.0,'D':4.3,'C':2.8,'U':2.8,'M':2.4,'F':2.2,'Y':2.0,'W':2.4,'G':2.0,
                'P':1.9,'B':1.5,'V':1.0,'K':0.8,'X':0.15,'Q':0.1,'J':0.15,'Z':0.07}
    eng_order = sorted(eng_freq.keys(), key=lambda x: -eng_freq[x])
    
    mapping = {}
    used = set()
    for i, cipher_char in enumerate(cipher_order):
        if i < len(eng_order):
            mapping[cipher_char] = eng_order[i]
            used.add(eng_order[i])
    
    remained = []
    for p in Alphabet:
        if p not in used:
            remained.append(p)
    rdm.shuffle(remained)
    
    for c in Alphabet:
        if c not in mapping:
            mapping[c] = remained.pop()
    
    return complete_key(mapping)

def get_string(key):
    return "".join(key[a] for a in Alphabet)

def get_pre(texts):
    return (texts[0] if texts else "")[:128]

def show_mapping(key):
    lines = []
    for i in range(0, 26, 13):
        row = []
        for j in range(i, i+13):
            row.append(f"{Alphabet[j]}:{key[Alphabet[j]]}")
        lines.append(" ".join(row))
    return "\n".join(lines)

# 4글자 조합 점수 계산 클래스
class QuadScore:
    def __init__(self, filepath=None):
        self.data = {}
        self.total_count = 0
        
        if filepath:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    gram = parts[0].upper()
                    try:
                        count = int(parts[1])
                    except ValueError:
                        continue
                    if len(gram) == 4 and gram.isalpha():
                        self.data[gram] = self.data.get(gram, 0) + count
                        self.total_count += count
        
        self.is_empty = (self.total_count == 0)
        
        if not self.is_empty:
            self.min_score = math.log10(0.01 / self.total_count)
            for gram in list(self.data.keys()):
                self.data[gram] = math.log10(self.data[gram] / self.total_count)
        
        self.cache = {}
    
    def calculate_score(self, text):
        cleaned = clean_text(text)
        if cleaned in self.cache:
            return self.cache[cleaned]
        
        if self.is_empty:
            return 0.0
        
        score = 0.0
        for i in range(len(cleaned) - 3):
            score += self.data.get(cleaned[i:i+4], self.min_score)
        
        if len(self.cache) < 50000:
            self.cache[cleaned] = score
        return score

#Ngram 클래스
class NgramScore:
    def __init__(self, filepath=None, n=3):
        self.n = n
        self.data = {}
        self.total_count = 0
        
        if filepath:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    gram = parts[0].upper()
                    if len(gram) != n or not gram.isalpha():
                        continue
                    try:
                        count = int(parts[1])
                    except ValueError:
                        continue
                    self.data[gram] = self.data.get(gram, 0) + count
                    self.total_count += count
        
        self.is_empty = (self.total_count == 0)
        
        if not self.is_empty:
            self.min_score = math.log10(0.01 / self.total_count)
            for gram in list(self.data.keys()):
                self.data[gram] = math.log10(self.data[gram] / self.total_count)
        
        self.cache = {}
    
    def calculate_score(self, text):
        cleaned = clean_text(text)
        if cleaned in self.cache:
            return self.cache[cleaned]
        
        if self.is_empty:
            return 0.0
        
        score = 0.0
        text_len = len(cleaned)
        for i in range(text_len - self.n + 1):
            score += self.data.get(cleaned[i:i+self.n], self.min_score)
        
        if len(self.cache) < 50000:
            self.cache[cleaned] = score
        return score

# 영어 빈도 데이터
LETTER_FREQ = {
    'E':12.0,'T':9.1,'A':8.2,'O':7.5,'I':7.0,'N':6.7,'S':6.3,'R':6.0,'H':6.1,
    'L':4.0,'D':4.3,'C':2.8,'U':2.8,'M':2.4,'F':2.2,'Y':2.0,'W':2.4,'G':2.0,
    'P':1.9,'B':1.5,'V':1.0,'K':0.8,'X':0.15,'Q':0.1,'J':0.15,'Z':0.07
}

# 흔한 단어들
COMMON = {
    "THE","BE","TO","OF","AND","A","IN","THAT","HAVE","I","IT","FOR","NOT","ON","WITH",
    "HE","AS","YOU","DO","AT","THIS","BUT","HIS","BY","FROM","THEY","WE","SAY","HER","SHE",
    "OR","AN","WILL","MY","ONE","ALL","WOULD","THERE","THEIR","IS","ARE","WAS","WERE",
    "IF","THEN","ELSE","WHAT","WHEN","WHERE","WHO","WHOM","WHY","HOW"
}

# 이상한 2글자 조합
AWKWARD_PAIR = {"QH","JQ","VV","ZZ","QQ","JJ"}

# 카이제곱 검정
def chi_score(freq_dict, total_len):
    score = 0.0
    for ch in Alphabet:
        observed = freq_dict.get(ch, 0)
        expected = LETTER_FREQ.get(ch, 0.01) / 100.0 * total_len
        score += ((observed - expected) ** 2) / (expected + 1e-9)
    return -score

# 2글자 조합 점수
def bigram_bonus(text):
    if len(text) < 2:
        return 0.0
    
    weights = {
        "TH":1.0,"HE":1.0,"IN":0.9,"ER":0.9,"AN":0.8,"RE":0.8,"ED":0.8,"ON":0.8,"ES":0.8,"ST":0.8,
        "EN":0.7,"AT":0.7,"TO":0.7,"NT":0.7,"HA":0.7,"ET":0.7,"OU":0.6,"EA":0.6,"HI":0.6,"IS":0.6,
        "OR":0.6,"IT":0.6,"AS":0.5,"TE":0.5,"AR":0.5,"SE":0.5,"CO":0.5,"ME":0.5,"DE":0.5,
        "QU":0.9,"CK":0.6,"BR":0.35,"OW":0.35,"FO":0.35,"OX":0.35,"JU":0.35,"UM":0.35,"MP":0.35,
        "PS":0.35,"LA":0.3,"AZ":0.3,"ZY":0.3,
    }
    
    score = 0.0
    prev = text[0]
    for i in range(1, len(text)):
        score += weights.get(prev + text[i], 0.0)
        prev = text[i]
    return score

# 단어 추출
def extract_words(text):
    return UPPER_WORD.findall(text.upper())

# 사전 로드
def load_dictionary(filepath, max_words=None):
    if not filepath:
        return set()
    
    words = set()
    count = 0
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            word = re.sub(r"[^A-Za-z']", "", line.strip())
            if word:
                words.add(word.upper())
                count += 1
                if max_words is not None and count >= max_words:
                    break
    return words

# 단어 패턴 (예: HELLO -> 0-1-2-2-3)
def word_pattern(word):
    mapping = {}
    idx = 0
    pattern = []
    for ch in word:
        if ch not in mapping:
            mapping[ch] = str(idx)
            idx += 1
        pattern.append(mapping[ch])
    return "-".join(pattern)

# 패턴 인덱스 만들기
def build_pattern_index(dictionary, min_len=2, max_len=12):
    index = defaultdict(set)
    for word in dictionary:
        if min_len <= len(word) <= max_len and re.fullmatch(r"[A-Z']+", word):
            key = (len(word), word_pattern(word))
            index[key].add(word)
    return index

# 사전/패턴 매칭 점수
def dict_and_pattern_scores(decrypted, dictionary, pattern_idx):
    words = extract_words(decrypted)
    if not words:
        return 0.0, 0.0, 0
    
    #사전 매칭
    matches = 0
    for w in words:
        if w in dictionary or w in COMMON:
            matches += 1
    dict_ratio = matches / len(words)
    
    # 패턴 매칭
    pattern_matches = 0
    pattern_total = 0
    for w in words:
        cleaned_w = NOT_ALPHA.sub('', w)
        if 3 <= len(cleaned_w) <= 12:
            pattern_total += 1
            if (len(cleaned_w), word_pattern(cleaned_w)) in pattern_idx:
                pattern_matches += 1
    
    pattern_ratio = (pattern_matches / pattern_total) if pattern_total else 0.0
    
    # 이상한 2글자 조합 카운트
    cleaned = clean_text(decrypted)
    weird_count = 0
    for i in range(len(cleaned) - 1):
        if cleaned[i:i+2] in AWKWARD_PAIR:
            weird_count += 1
    
    return dict_ratio, pattern_ratio, weird_count

# 팬그램 관련
PANGRAM_WORD = {"THE","QUICK","BROWN","FOX","JUMPS","OVER","LAZY","DOG"}
PANGRAM_MIN_LETTER = 23

def pangram_word_check(text_upper):
    words_found = set(UPPER_WORD.findall(text_upper))
    hits = 0
    for w in PANGRAM_WORD:
        if w in words_found:
            hits += 1
    return 1.0 if hits >= 7 else hits / 7.0

def pangram_coverage(text):
    return min(1.0, len(set(text)) / 26.0)

def is_near_pangram(text):
    return len(set(text)) >= PANGRAM_MIN_LETTER

# 팬그램 패턴 체크
PANGRAM_PATTERNS = [
    (r"\bTHE\b", 1.0), (r"\bOVER\b", 0.9), (r"\bDOG\b", 0.9), (r"\bFOX\b", 0.8),
    (r"\bLAZY\b", 0.8), (r"\bQUICK\b", 1.1), (r"\bBROWN\b", 0.9), (r"\bJUMPS\b", 0.9),
]

def check_pangram_order(text_upper):
    s = " " + text_upper + " "
    the1 = s.find(" THE ")
    over = s.find(" OVER ")
    the2 = s.rfind(" THE ")
    dog = s.find(" DOG")
    
    if the1 != -1 and over != -1 and the2 != -1 and dog != -1:
        if the1 < over < the2 < dog:
            return 0.5
    return 0.0

def pangram_pattern_score(text_upper):
    score = 0.0
    for pattern, weight in PANGRAM_PATTERNS:
        if re.search(pattern, text_upper):
            score += weight
    
    score = min(1.0, score / 7.6)
    score += check_pangram_order(text_upper)
    return min(1.2, score)

#가중치들
WEIGHT_QUAD = 1.0
WEIGHT_TRI = 0.40
WEIGHT_BI = 0.10
WEIGHT_DIGRAM = 0.03
WEIGHT_CHI = 0.015
WEIGHT_DICT = 15.0
WEIGHT_SHAPE = 6.0
WEIRD_PENALTY = -4.0
WEIGHT_PANGRAM = 12.0

# 종합 점수 계산
def total_score(decrypted_texts, quad_score, dictionary, pattern_idx, 
                tri_score=None, bi_score=None, is_short=False, use_pangram=False):
    score = 0.0
    
    #짧은 텍스트일시에는 자동으로 가중치 조절
    if is_short:
        w_dict = WEIGHT_DICT * 1.25
        w_shape = WEIGHT_SHAPE * 1.15
        w_pang = WEIGHT_PANGRAM * 1.35
        w_penalty = WEIRD_PENALTY
    else:
        w_dict = WEIGHT_DICT
        w_shape = WEIGHT_SHAPE
        w_pang = WEIGHT_PANGRAM
        w_penalty = WEIRD_PENALTY
    
    for dec in decrypted_texts:
        cleaned = clean_text(dec)
        upper = dec.upper()
        
        # N-gram 점수들
        q = quad_score.calculate_score(dec) if quad_score and not quad_score.is_empty else 0.0
        t = tri_score.calculate_score(dec) if tri_score and not tri_score.is_empty else 0.0
        b = bi_score.calculate_score(dec) if bi_score and not bi_score.is_empty else 0.0
        
        # Digram 보너스
        digram_score = bigram_bonus(cleaned)
        
        # 카이제곱
        freq = Counter(cleaned)
        text_len = len(cleaned) if len(cleaned) > 0 else 1
        chi = chi_score(freq, text_len)
        
        # 사전/패턴 매칭
        dict_ratio, pattern_ratio, weird_count = dict_and_pattern_scores(dec, dictionary, pattern_idx)
        
        # 팬그램 점수
        pang_score = 0.0
        if use_pangram:
            coverage = pangram_coverage(cleaned)
            word_check = pangram_word_check(upper)
            mix = 0.65 * coverage + 0.35 * word_check
            
            if is_near_pangram(cleaned):
                mix += 0.15
            
            mix = max(0.0, min(1.2, mix))
            
            anchor = 0.0
            if is_short and coverage >= 0.50:
                anchor = pangram_pattern_score(upper)
            
            pang_score = w_pang * mix + (w_pang * 0.85) * anchor
        
        # 종합
        score += (WEIGHT_QUAD * q +
                  WEIGHT_TRI * t +
                  WEIGHT_BI * b +
                  WEIGHT_DIGRAM * digram_score +
                  WEIGHT_CHI * chi +
                  w_dict * dict_ratio +
                  w_shape * pattern_ratio +
                  pang_score +
                  w_penalty * weird_count)
    
    return score

# 이웃 키 만들기
def swap_keys(key_dict, rng, num_swaps=1):
    new_key = dict(key_dict)
    for _ in range(num_swaps):
        a, b = rng.sample(Alphabet, 2)
        new_key[a], new_key[b] = new_key[b], new_key[a]
    return new_key

# Beam Search
def beam_search(cipher_texts, quad_score, dictionary, pattern_idx,
                beam_size=64, neighbors_per_node=60, max_steps=150, num_restarts=20, 
                random_seed=1234, tri_score=None, bi_score=None, 
                progress_interval=10, early_stop=20, 
                is_short=False, use_pangram=False):
    
    rdm = random.Random(random_seed)
    all_candidates = []
    
    
    for restart in range(num_restarts):
        #beam 생성
        beam = []
        for _ in range(beam_size):
            init_key = make_initial_key(" ".join(cipher_texts), rdm)
            
            # 짧은 텍스트면 약간 섞기
            if is_short and rdm.random() < 0.7:
                swaps = 1 if rdm.random() < 0.75 else 2
                init_key = swap_keys(init_key, rdm, swaps)
            
            decoded = [substitution_key(t, init_key) for t in cipher_texts]
            score = total_score(decoded, quad_score, dictionary, pattern_idx,
                               tri_score, bi_score, is_short, use_pangram)
            beam.append((score, init_key, decoded))
        
        beam.sort(key=lambda x: (-x[0], get_pre(x[2]), get_string(x[1])))
        
        best_score = beam[0][0]
        stall_count = 0
        
        try:
            for step in range(max_steps):
                # 시간 체크
                
                # 이웃 생성
                candidates = []
                for sc, key, _ in beam:
                    for _ in range(neighbors_per_node):
                        swaps = 1 if rdm.random() < 0.7 else 2
                        if is_short and rdm.random() < 0.10:
                            swaps = 3
                        
                        neighbor_key = swap_keys(key, rdm, swaps)
                        neighbor_decoded = [substitution_key(t, neighbor_key) for t in cipher_texts]
                        neighbor_score = total_score(neighbor_decoded, quad_score, dictionary, pattern_idx,
                                                     tri_score, bi_score, is_short, use_pangram)
                        candidates.append((neighbor_score, neighbor_key, neighbor_decoded))
                
                #합치고 정렬
                pool = candidates + beam
                pool.sort(key=lambda x: (-x[0], get_pre(x[2]), get_string(x[1])))
                
                #중복 제거
                seen_texts = set()
                deduplicated = []
                for sc, key, decoded in pool:
                    text_sig = decoded[0] if decoded else ""
                    if text_sig in seen_texts:
                        continue
                    deduplicated.append((sc, key, decoded))
                    seen_texts.add(text_sig)
                    if len(deduplicated) >= beam_size:
                        break
                
                beam = deduplicated
                
                # 진행상황 출력
                if (step + 1) % progress_interval == 0:
                    preview = beam[0][2][0] if beam[0][2] else ""
                    preview = preview.replace("\n", " ").strip()
                    if len(preview) > 120:
                        preview = preview[:120] + "..."
                    print(f"[restart {restart+1}/{num_restarts}] step {step+1}/{max_steps} | 중간 출력: \"{preview}\"", flush=True)
                
                # Early stopping
                if beam[0][0] <= best_score + 1e-12:
                    stall_count += 1
                    stop_threshold = early_stop if not is_short else max(8, early_stop // 2)
                    if stall_count >= stop_threshold:
                        break
                else:
                    best_score = beam[0][0]
                    stall_count = 0
        
        except KeyboardInterrupt:
            print("\n[beam] 중단됨: 현재까지의 최선 반환", flush=True)
        
        all_candidates.append(beam[0])
        
        
    
    all_candidates.sort(key=lambda x: (-x[0], get_pre(x[2]), get_string(x[1])))
    return all_candidates

#어닐링
def simulated_annealing(cipher_texts, quad_score, dictionary, pattern_idx,
                        max_iters=20000, num_restarts=10, random_seed=7,
                        tri_score=None, bi_score=None, is_short=False, use_pangram=False):
    
    rdm = random.Random(random_seed)
    results = []
    
    for _ in range(num_restarts):
        key = make_initial_key(" ".join(cipher_texts), rdm)
        
        if is_short and rdm.random() < 0.7:
            key = swap_keys(key, rdm, 2)
        
        decoded = [substitution_key(t, key) for t in cipher_texts]
        current = (total_score(decoded, quad_score, dictionary, pattern_idx,
                              tri_score, bi_score, is_short, use_pangram), key, decoded)
        best = current
        
        temp0 = 1.1
        
        try:
            for i in range(max_iters):
                temp = temp0 * (0.9998 ** i)
                
                #이웃 생성
                swaps = 1 if rdm.random() < 0.7 else 2
                neighbor_key = swap_keys(current[1], rdm, swaps)
                
                if is_short and rdm.random() < 0.10:
                    neighbor_key = swap_keys(neighbor_key, rdm, 2)
                
                neighbor_decoded = [substitution_key(t, neighbor_key) for t in cipher_texts]
                neighbor_score = total_score(neighbor_decoded, quad_score, dictionary, pattern_idx,
                                            tri_score, bi_score, is_short, use_pangram)
                
                delta = neighbor_score - current[0]
                
                
                if delta > 0 or (temp > 1e-9 and math.log(rdm.random() + 1e-12) < delta / (temp + 1e-12)):
                    current = (neighbor_score, neighbor_key, neighbor_decoded)
                    if current[0] > best[0]:
                        best = current
                
                #주기적으로 재시작
                if i % 4000 == 0 and i > 0:
                    for _ in range(2):
                        neighbor_key = swap_keys(current[1], rdm, 2)
                    neighbor_decoded = [substitution_key(t, neighbor_key) for t in cipher_texts]
                    current = (total_score(neighbor_decoded, quad_score, dictionary, pattern_idx,
                                          tri_score, bi_score, is_short, use_pangram), neighbor_key, neighbor_decoded)
                    if current[0] > best[0]:
                        best = current
        
        except KeyboardInterrupt:
            print("\n중지", flush=True)
        
        results.append(best)
    
    results.sort(key=lambda x: (-x[0], get_pre(x[2]), get_string(x[1])))
    return results

#CSP
def invert_key(key):
    inverted = {}
    for cipher, plain in key.items():
        inverted[plain] = cipher
    return inverted

def is_consistent(current, new_mapping):
    inv = invert_key(current)
    for cipher, plain in new_mapping.items():
        if cipher in current and current[cipher] != plain:
            return False
        if plain in inv and inv[plain] != cipher:
            return False
    return True

def merge_keys(current, new_mapping):
    result = dict(current)
    result.update(new_mapping)
    return result

def word_candidate(cipher_word, dictionary, pattern_idx, max_cands=200):
    bucket = pattern_idx.get((len(cipher_word), word_pattern(cipher_word)), set())
    
    if not bucket:
        bucket = set()
        for w in dictionary:
            if len(w) == len(cipher_word):
                bucket.add(w)
    
    candidates = list(bucket)
    
    if len(candidates) > max_cands:
        priority = []
        for w in candidates:
            priority.append((-(2 if w in COMMON else 1), w))
        priority.sort()
        candidates = [w for _, w in priority[:max_cands]]
    
    return candidates

def make_mapping(cipher_word, plain_word):
    mapping = {}
    for c, p in zip(cipher_word, plain_word):
        if not c.isalpha():
            if c != p:
                return {}
            continue
        
        c_upper = c.upper()
        p_upper = p.upper()
        
        if c_upper in mapping and mapping[c_upper] != p_upper:
            return {}
        mapping[c_upper] = p_upper
    
    return mapping

def csp_solve(cipher_words, dictionary, pattern_idx, initial_key, 
              max_depth=2, max_cands=120, quad_score=None, tri_score=None, bi_score=None):
    
    anchor_words = []
    for w in cipher_words:
        if re.fullmatch(r"[A-Z]+", w):
            anchor_words.append(w)
    
    if not anchor_words:
        return initial_key
    
    best_key = dict(initial_key)
    best_score = -1e100
    
    word_candidates = []
    for cipher_w in anchor_words:
        cands = word_candidate(cipher_w, dictionary, pattern_idx, max_cands)
        cands = sorted(cands)
        word_candidates.append((cipher_w, cands[:max_cands]))
    
    def search_recursive(depth, current_key):
        nonlocal best_key, best_score
        
        if depth == min(max_depth, len(word_candidates)):
            decoded_text = substitution_key(" ".join(cipher_words), current_key)
            cleaned = clean_text(decoded_text)
            
            q_score = quad_score.calculate_score(decoded_text) if quad_score else 0.0
            t_score = tri_score.calculate_score(decoded_text) if tri_score else 0.0
            b_score = bi_score.calculate_score(decoded_text) if bi_score else 0.0
            
            dict_r, shape_r, _ = dict_and_pattern_scores(decoded_text, dictionary, pattern_idx)
            
            total = (WEIGHT_QUAD * q_score + WEIGHT_TRI * t_score + 
                    WEIGHT_BI * b_score + WEIGHT_DICT * dict_r + WEIGHT_SHAPE * shape_r)
            
            if total > best_score:
                best_score = total
                best_key = dict(current_key)
            return
        
        cipher_w, plain_candidates = word_candidates[depth]
        
        for plain_w in plain_candidates:
            mapping = make_mapping(cipher_w, plain_w)
            if not mapping:
                continue
            
            if not is_consistent(current_key, mapping):
                continue
            
            search_recursive(depth + 1, merge_keys(current_key, mapping))
    
    search_recursive(0, dict(initial_key))
    return complete_key(best_key)

#값 세부 조정
def fine_tune(cipher_texts, key, quad_score, dictionary, pattern_idx, 
              max_iters=4000, tri_score=None, bi_score=None, is_short=False, use_pangram=False):
    
    rng = random.Random(11)
    current = dict(key)
    current_decoded = [substitution_key(t, current) for t in cipher_texts]
    current_score = total_score(current_decoded, quad_score, dictionary, pattern_idx,
                                tri_score, bi_score, is_short, use_pangram)
    
    letters = list(Alphabet)
    
    for _ in range(max_iters):
        new_key = dict(current)
        
        if rng.random() < 0.85:
            a, b = rng.sample(letters, 2)
            new_key[a], new_key[b] = new_key[b], new_key[a]
        else:
            a, b, c = rng.sample(letters, 3)
            new_key[a], new_key[b], new_key[c] = new_key[b], new_key[c], new_key[a]
        
        new_decoded = [substitution_key(t, new_key) for t in cipher_texts]
        new_score = total_score(new_decoded, quad_score, dictionary, pattern_idx,
                               tri_score, bi_score, is_short, use_pangram)
        
        if new_score > current_score:
            current = new_key
            current_decoded = new_decoded
            current_score = new_score
    
    return current


def print_frequency(sources, cipher_texts):
    print("\n--- 암호문 내 알파벳의 빈도수와 비율 ---")
    
    for src, cipher in zip(sources, cipher_texts):
        cleaned = clean_text(cipher)
        freq = Counter(cleaned)
        total = sum(freq.values())
        
        print(f"\n[{src}] length={len(cleaned)}")
        
        row1 = []
        row2 = []
        
        for i, ch in enumerate(Alphabet):
            count = freq.get(ch, 0)
            percentage = (100.0 * count / total) if total else 0.0
            text = f"{ch}:{count:d}({percentage:5.2f}%)"
            
            if i < 13:
                row1.append(text)
            else:
                row2.append(text)
        
        print(" ".join(row1))
        print(" ".join(row2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cipherfile", type=str, required=True, help="Ciphertext file")
    
    # 데이터 파일들
    parser.add_argument("--quadgrams", type=str, default="english_quadgrams.txt")
    parser.add_argument("--trigrams", type=str, default="english_trigrams.txt")
    parser.add_argument("--bigrams", type=str, default="english_bigrams.txt")
    parser.add_argument("--lexicon", type=str, default="english_words.txt")
    parser.add_argument("--lexicon-limit", type=int, default=9000)
    
    #알고리즘 선택
    parser.add_argument("--solver", type=str, default="beam", choices=["beam", "anneal"])
    
    #Beam search 관련
    parser.add_argument("--beam-width", type=int, default=80)
    parser.add_argument("--neighbors", type=int, default=160)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--restarts", type=int, default=1)
    
    #CSP 파라미터
    parser.add_argument("--csp-depth", type=int, default=10)
    parser.add_argument("--csp-max-cands", type=int, default=6000)
    
    #미세조정
    parser.add_argument("--polish-iters", type=int, default=17000)
    
    #어날링 파라미터
    parser.add_argument("--iters", type=int, default=20000)
    
    #가중치 조절
    parser.add_argument("--w-quad", type=float, default=1.0)
    parser.add_argument("--w-tri", type=float, default=0.85)
    parser.add_argument("--w-bi", type=float, default=0.22)
    parser.add_argument("--w-di", type=float, default=0.12)
    parser.add_argument("--w-chi", type=float, default=0.008)
    parser.add_argument("--w-dict", type=float, default=36.0)
    parser.add_argument("--w-shape", type=float, default=13.5)
    parser.add_argument("--rare-pen", type=float, default=-4.0)
    
    # 팬그램 옵션
    parser.add_argument("--use-pangram", action="store_true")
    parser.add_argument("--w-pang", type=float, default=12.0)
    
    # 기타
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--hard-short", action="store_true")
    
    args = parser.parse_args()
    
    # 전역 가중치 업데이트
    global WEIGHT_QUAD, WEIGHT_TRI, WEIGHT_BI, WEIGHT_DIGRAM, WEIGHT_CHI
    global WEIGHT_DICT, WEIGHT_SHAPE, WEIRD_PENALTY, WEIGHT_PANGRAM
    
    WEIGHT_QUAD = args.w_quad
    WEIGHT_TRI = args.w_tri
    WEIGHT_BI = args.w_bi
    WEIGHT_DIGRAM = args.w_di
    WEIGHT_CHI = args.w_chi
    WEIGHT_DICT = args.w_dict
    WEIGHT_SHAPE = args.w_shape
    WEIRD_PENALTY = args.rare_pen
    WEIGHT_PANGRAM = args.w_pang
    
    #파일 읽기
    filepath = args.cipherfile.strip()
    if not filepath or "," in filepath:
        print("Error\n", file=sys.stderr)
        sys.exit(1)
    
    cipher_text = read_file(filepath)
    cipher_texts = [cipher_text]
    src = [filepath]
    
    #빈도분석 출력하기
    print_frequency(src, cipher_texts)
    
    #데이터 불러오가
    quad = QuadScore(args.quadgrams) if args.quadgrams else QuadScore(None)
    tri = NgramScore(args.trigrams, n=3) if args.trigrams else None
    bi = NgramScore(args.bigrams, n=2) if args.bigrams else None
    
    dictionary = load_dictionary(args.lexicon, args.lexicon_limit) if args.lexicon else set()
    pattern_idx = build_pattern_index(dictionary) if dictionary else {}
    
    # 텍스트 길이 확인
    combined = clean_text(cipher_text)
    is_short_text = False
    
    # 짧은 텍스트 감지
    if args.hard_short or (len(cipher_texts) == 1 and len(combined) <= 80):
        is_short_text = True
        
        if args.solver == "beam":
            # 기본값 짧은 텍스트용으로
            if args.beam_width == 64:
                args.beam_width = 24
            if args.neighbors == 60:
                args.neighbors = 28
            if args.steps == 150:
                args.steps = 110
            if args.restarts == 20:
                args.restarts = 18
        else:
            if args.iters == 20000:
                args.iters = 30000
            if args.restarts == 20:
                args.restarts = 14
        
        if args.use_pangram:
            WEIGHT_PANGRAM = max(WEIGHT_PANGRAM, 16.0)
    
    # 메인 탐색
    if args.solver == "beam":
        candidates = beam_search(
            cipher_texts, quad, dictionary, pattern_idx,
            beam_size=args.beam_width,
            neighbors_per_node=args.neighbors,
            max_steps=args.steps,
            num_restarts=args.restarts,
            random_seed=args.seed,
            tri_score=tri,
            bi_score=bi,
            progress_interval=10,
            early_stop=20,
            is_short=is_short_text,
            use_pangram=args.use_pangram
        )
    else:
        candidates = simulated_annealing(
            cipher_texts, quad, dictionary, pattern_idx,
            max_iters=args.iters,
            num_restarts=args.restarts,
            random_seed=args.seed,
            tri_score=tri,
            bi_score=bi,
            is_short=is_short_text,
            use_pangram=args.use_pangram
        )
    
    #후처리
    if candidates:
        best_score, best_key, best_decoded = candidates[0]
        
        #CSP 적용
        if args.csp_depth > 0 and dictionary:
            words = UPPER_WORD.findall(cipher_text.upper())
            unique_words = sorted(set(words), key=lambda w: (len(w), len(set(w))), reverse=True)
            anchor_words = unique_words[:max(2, min(4, args.csp_depth + 1))]
            
            try_key = csp_solve(
                anchor_words, dictionary, pattern_idx, best_key,
                max_depth=args.csp_depth,
                max_cands=args.csp_max_cands,
                quad_score=quad,
                tri_score=tri,
                bi_score=bi
            )
            
            try_decoded = [substitution_key(t, try_key) for t in cipher_texts]
            try_score = total_score(try_decoded, quad, dictionary, pattern_idx,
                                   tri, bi, is_short_text, args.use_pangram)
            
            if try_score > best_score:
                best_score = try_score
                best_key = try_key
                best_decoded = try_decoded
        
        #미세 조정
        if args.polish_iters > 0:
            polished_key = fine_tune(
                cipher_texts, best_key, quad, dictionary, pattern_idx,
                max_iters=args.polish_iters,
                tri_score=tri,
                bi_score=bi,
                is_short=is_short_text,
                use_pangram=args.use_pangram
            )
            
            polished_decoded = [substitution_key(t, polished_key) for t in cipher_texts]
            polished_score = total_score(polished_decoded, quad, dictionary, pattern_idx,
                                        tri, bi, is_short_text, args.use_pangram)
            
            if polished_score > best_score:
                best_score = polished_score
                best_key = polished_key
                best_decoded = polished_decoded
        
        candidates[0] = (best_score, best_key, best_decoded)
    
    #최종결과 출력
    print("\n--- Plain Text ---\n")
    
    if candidates:
        score, key, decoded_texts = candidates[0]
        
        for i, (src, decoded) in enumerate(zip(src, decoded_texts), start=1):
            preview = decoded[:500].replace("\n", " ")
            if len(decoded) > 500:
                preview += "..."
            print(f"{src}: {preview}")
        
        print("\nKey (cipher -> plain):")
        print(show_mapping(key))
    else:
        print("no candidates\n")

if __name__ == "__main__":
    main()
