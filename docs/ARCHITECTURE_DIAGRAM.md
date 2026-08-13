# CubbyLLM — full system diagram

**As of 2026-08-08** (post serve-stack cycle). Solid arrows = the live default
path; dotted = gated/optional or future; thick = the verified CoT traversal.
Every component below exists in code unless marked *(future)*.

```mermaid
flowchart TB

%% ================= FAST SEMANTIC LAYER =================
subgraph ENC["FAST SEMANTIC LAYER — no model in the challenge path (~25µs–1ms)"]
    direction TB
    TXT["text / question / passage"] --> SW["split_words<br/>lowercase, len ≥ 2"]
    SW --> FWE["FastWordEncoder — table v4<br/>60,151 words · teacher potion-retrieval-32M<br/>anisotropy-removed · QR-projected (cosine-exact)<br/>0.5 signal-code blend · IDF-weighted bundle"]
    SW -.->|"OOV word"| SIG["seeded signal code<br/>BLAKE2b w:&lt;word&gt;"]
    SIG --> FWE
    FWE --> BLOCK["block code 80×128 = qFHRR<br/>D = 10,240"]
    BLOCK --> DENSE["dense f16 form<br/>full cosine geometry"]
    BLOCK --> COMPACT["compact form 80 B/doc<br/>per-block argmax = integer phases<br/>ADC scoring, no codebook"]
    COMPACT -.->|"top-20 shortlist"| DENSE
end

%% ================= WORLDS =================
subgraph MOWM["WORLDS — mowm (specialization by routing, spawn recoverable)"]
    direction TB
    CB["CubbyBridge"]
    CB --> T1["(a) exact reuse<br/>memoized challenge"]
    CB --> T2["(b) centroid + margin routing<br/>s1 ≥ τ_match AND margin ≥ τ_margin"]
    T2 --> T2B["(b2) inter-world query<br/>best member &lt; τ_answer →<br/>router forwards to next world<br/>(= flat-search quality at 1/60 cost)"]
    CB --> T3["(c) spawn specialist<br/>from nearest axioms"]
    T2B --> RTR["MoWMRouter<br/>query_worlds · delegation state"]
    RTR --> W1["World A<br/>axioms · centroid · HYLAs"]
    RTR --> W2["World B …"]
    RTR --> WN["World N<br/>(era / year / domain worlds)"]
    LIB["AxiomLibrary<br/>block-code axioms"] --> W1
end

%% ================= COT PIPELINE =================
subgraph COT["VERIFIED CoT PIPELINE — cubbyllm/reasoning (claimed-answer precision 0.994)"]
    direction TB
    Q["question"] ==> PL["planner — grammar v2<br/>6 WH-frames · 95.4% coverage<br/>len guard · nested forms → honest fail"]
    PL ==> WALK["walk: retrieve per hop<br/>accept: relation-match + subject-match<br/>repair budget 3 · chase expansion"]
    WALK ==> PB["programs.py<br/>CubeLang chain program<br/>bind frame, Hi_REL, obj<br/>+ ABSENT_CTRL control"]
    PB ==> BR["bridge — run_program_proto<br/>protobuf over stdio<br/>verify-before-execute (strict)"]
    BR ==> VM["cubelang VM (Rust)<br/>BIND_ROLE / UNBIND<br/>+ cosine cleanup → (symbol, similarity)"]
    VM ==> VER["verify: every hop ≥ τ_vm<br/>symbol == walked object<br/>control below τ (live: 0.50 vs 0.029)"]
    VER ==>|"all pass"| ANS["verified answer + auditable trace"]
    VER -.->|"fail"| REP["ban weakest fact → retry once"]
    REP -.-> WALK
    VER -.->|"exhausted"| HF["honest fail<br/>verified=false + partial trace"]
    ANS --> HARV["harvest jsonl<br/>trace tree · rejection reasons<br/>DPO pairs · provenance hash"]
    HF --> HARV
end

%% ================= SYMBOLIC BOUNDARY =================
BOUND["SYMBOLIC BOUNDARY<br/>only symbols cross — never raw hypervectors<br/>(grilly block codes ≠ Rust VSA; by design)"]

%% ================= TRUNK =================
subgraph TRUNK["TRUNK — cubbyllm/model (151M trained · 2B target D2048/L32)"]
    direction TB
    TOK["BBPE-128k tokenizer<br/>atomic CubeLang opcodes in vocab"] --> EMB["HybridEmbedding<br/>(frozen-table experiment queued)"]
    EMB --> BB["HybridBackbone<br/>MinGRU ×2 : sliding-window attn ×1<br/>RoPE · bounded KV · O(1) decode<br/>(1:3 correct for weak recurrence)"]
    BB -.->|"CB_MEM_EVERY"| MEM["MemoryRead ↔ EpisodicStore<br/>retrieval interleaved in the stack"]
    BB --> HEAD["TopKRetrievalHead<br/>softmax-free · exact top-K cosine"]
    HG["HyperGenerator θ=f(c)<br/>+ SnapshotHardener (anti-drift)"] -.-> BB
    FSR["FrozenSlotRouter<br/>offline-pretrained, frozen<br/>(legitimate: provenance facets)"] -.-> HG
end

%% ================= TRAINING =================
subgraph TRAIN["TRAINING & ASSETS"]
    direction TB
    CACHE["14.37B-token uint32 cache (D:)<br/>open_thoughts reserved for SFT"] --> TL["TrainLoop<br/>bind_weight=0 (H-B6 negative)"]
    TL --> CKPT["checkpoints D:\\CUBBY-TRAINED-MODELS<br/>hd5_mem21 · ab_mingru · tables v1–v4"]
    RUNBOOK["2B runbook (panel-settled)<br/>MFU pilot → cycle-1 dense →<br/>offline ULD-KD → int8 PTQ →<br/>cycle-2 tag-routed MoE upcycle"]
    HARV -.->|"verified supervision<br/>(question, program, trace)"| SFT["trunk-SFT program emission<br/>(future — replaces hand grammar)"]
    SFT -.-> TOK
end

%% ================= CROSS-LINKS =================
BLOCK ==> WALK
BLOCK --> CB
DENSE --> RTR
W1 -.->|"world-backed fact store (future)"| WALK
PB ==> BOUND ==> VM
FWE -.->|"facet tables / tags (future):<br/>domain · intent · emotion · era"| FSR
COMPACT -.->|"haystack @1M docs<br/>hit@20 0.92"| WALK
TOK -.-> FWE

%% ================= STYLES =================
classDef enc fill:#e8f0fe,stroke:#4472c4,color:#1a1a2e
classDef world fill:#e6f4ea,stroke:#3a7d44,color:#1a2e1a
classDef cot fill:#fdf3e7,stroke:#c47f17,color:#2e241a
classDef trunk fill:#f3e8fd,stroke:#7c4dbe,color:#241a2e
classDef train fill:#fbe9eb,stroke:#b5495b,color:#2e1a1e
classDef bound fill:#fff8c5,stroke:#9a8700,color:#2e2a1a

class TXT,SW,FWE,SIG,BLOCK,DENSE,COMPACT enc
class CB,T1,T2,T2B,T3,RTR,W1,W2,WN,LIB world
class Q,PL,WALK,PB,BR,VM,VER,ANS,REP,HF,HARV cot
class TOK,EMB,BB,MEM,HEAD,HG,FSR trunk
class CACHE,TL,CKPT,RUNBOOK,SFT train
class BOUND bound
```

## Reading the diagram

- **Fast semantic layer** (blue): the model-free serving substrate. One table
  lookup + IDF bundle produces a qFHRR block code carrying real semantics
  (0.97× its teacher on routing); the dense form is the quality ceiling, the
  80-byte compact form is the scale/storage/algebra form. Everything else
  consumes these codes.
- **Worlds** (green): specialization as routing. The bridge's tiers are
  ordered by cost; every τ is deployment-calibrated; a wrong route is
  recoverable by spawn, and a world missing an answer *asks the next world*
  (measured: flat-search answer quality at ~1/60 the comparisons).
- **CoT pipeline** (orange, thick path): the verified traversal — the first
  full Cubby→CubeLang→output loop. An answer is only *claimed* when every hop
  clears its threshold and the absent-role control stays silent; everything
  else is an honest refusal with a partial trace. All outcomes are harvested
  as verified supervision.
- **Symbolic boundary** (yellow): the two VSA algebras (grilly block codes;
  the Rust VM's dense bipolar codes) never exchange raw vectors — concepts
  in, `(symbol, similarity)` out. This is deny-by-default applied to
  representations.
- **Trunk** (purple): the hybrid LM. The retrieval head is softmax-free; the
  memory layer interleaves retrieval into the stack; θ=f(c) with an
  offline-frozen context router is the validated anti-forgetting mechanism.
- **Training** (red): current assets and the panel-settled 2B runbook. The
  harvest→SFT arrow is the system's self-improvement loop: today's hand
  grammar generates the training data for the model that replaces it.
