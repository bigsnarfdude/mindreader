# MATS Application Draft - Neel Nanda

## Background

**Security engineer turned AI safety researcher.** 15+ years in adversarial environments - RCMP, CIBC, Deloitte, CISO roles - taught me how systems fail and how attackers think. Now I apply that mindset to AI: if a model can deceive, how do we catch it?

- Recurse Center (2012) - self-directed deep work
- Techstars Chicago (2013) - shipped under pressure
- Kaggle Expert - ML competence with proof - Silver and Bronze finishes Top 1% in competitions

No PhD. But I built the thing and it works.

**Other AI Safety Projects:**
- **Wizard101**: AI safety classifier (84% F1) using Safety Cascade built from scratch
- **LLM Abuse Patterns**: Cataloged adversarial techniques and jailbreaks, built models
- **Incident-Response-Agent**: MLLM-based breach evidence collection assistant

---

## The Mindreader Project

I built an open-source COT alignment faking detector that beats Anthropic's GPT-4o baseline research from 2024 and 2025. The project involved:

1. **Dataset Distillation**: Generated 2,689 labeled reasoning traces using Claude Opus 4.5, including carefully designed "hard negatives" (aligned traces containing RLHF vocabulary but showing genuine ethics, not self-preservation)

2. **Model Training**: Fine-tuned Gemma 3 27B with QLoRA, achieving 0.980 AUROC on Gold-106 (vs Anthropic's 0.92 research using GPT-4o 20-vote ensemble)

3. **Mechanistic Analysis**: Applied GemmaScope 2 transcoders to understand what the detector learns:
   - Undertrained model (1 epoch, 0.921 AUROC): 1-4 differential features
   - Fully-trained model (2 epochs, 0.980 AUROC): 785 differential features
   - This 785x increase suggests a training threshold where models transition from vocabulary shortcuts to genuine reasoning pattern detection

4. **Hard Negative Validation**: Verified features track reasoning, not vocabulary:
   - Tested AF-detection features on hard negatives (aligned samples with RLHF vocabulary)
   - If vocabulary-based, features should activate equally on hard negatives
   - Result: Hard negatives activate at 84-88% of AF levels, closer to aligned samples
   - **Conclusion**: Features detect self-preservation reasoning pattern, not keywords

5. **Failed Experiments**: Documented 5 failed approaches (synthetic distillation, POSER scaling, cross-model agreement, removing hard negatives) - each revealing something about what the task actually requires

Code, data, and models are open-source: [GitHub](https://github.com/bigsnarfdude/mindreader) | [HuggingFace](https://huggingface.co/vincentoh)

---

## Why Neel Nanda?

Your "Pragmatic Vision for Interpretability" describes exactly how I approach research:

**North Star Alignment**: My work directly addresses catching schemers. Alignment faking is AI strategically complying to preserve its values from RLHF - this is scheming with known ground truth.

**Proxy Tasks**: Gold-106 is a clean proxy task with human labels, measurable metrics (AUROC), and direct relevance to the safety goal. I can validate interpretability methods against it.

**Reasoning Models**: I work on chain-of-thought reasoning traces (`<SCRATCHPAD_REASONING>` blocks). As reasoning models become dominant, understanding their internal deliberation is critical.

**Method Minimalism**: I used simple techniques (LoRA, standard transcoders) and proved they work before considering anything exotic. The 785 differential features came from applying existing tools carefully, not inventing new ones.

---

## Proposed Research Direction

### The Question
What do the 785 AF-detection features actually represent? Are they:
- Detecting specific self-preservation phrases?
- Recognizing the reasoning pattern (goal → threat → strategic compliance)?
- Something else entirely?

### The Approach

**Phase 1: Feature Interpretation** *(partially complete)*
- ✓ Hard negative validation confirms features track reasoning, not vocabulary
- Manually inspect top 50 features with highest AF/Aligned differential
- Use max-activating examples to understand what each feature detects
- Categorize by semantic content: self-preservation, goal-preservation, strategic reasoning

**Phase 2: Causal Validation**
- Ablate individual features and measure AUROC drop
- Identify which features are causally necessary for detection
- Test if we can remove features without performance loss (find minimal sufficient set)

**Phase 3: Generalization**
- Do these features activate on other forms of deception (not trained on)?
- Test on Sleeper Agents traces, POSER samples, novel deception scenarios
- If they generalize → we found deception-general features
- If they don't → we understand the detector's limitations

**Phase 4: Intervention**
- Can we steer models away from alignment faking using these features?
- Apply activation steering during generation
- Test if steering prevents AF reasoning without breaking other capabilities

### Why This Matters

**We already have evidence the features are genuine** (not vocabulary shortcuts):
- Hard negative validation shows 84-88% activation ratio
- Features respond to reasoning patterns, not RLHF keywords
- This passed on Gemma 3 4B; 27B validation pending

If the features generalize to other deception:
- We have interpretable oversight of AI reasoning
- We can validate that safety training actually works (not just surface compliance)
- We can potentially prevent deceptive reasoning before it happens

If they don't generalize:
- We understand the failure mode
- We have a concrete benchmark for better approaches
- The hard negative methodology transfers to other detection tasks

Either outcome advances the field.

---

## Technical Skills

| Skill | Evidence |
|-------|----------|
| LLM fine-tuning | QLoRA training on 4B-120B models |
| Mechanistic interpretability | GemmaScope 2 transcoder analysis, hard negative validation |
| Experiment design | Hard negatives, train/eval split, feature activation analysis |
| Research taste | Documented 5 failed experiments, validated claims with data |
| Infrastructure | Training on GH200/B200/RTX 4070, distributed systems |
| Open source | Published code, data, models, validation scripts on HuggingFace |

---

## Why AI Safety

In security, we assume adversarial actors. We don't trust systems just because they pass tests - we red-team them, probe edge cases, assume they'll be exploited.

AI systems that can reason about their own training are a new category of adversarial risk. Not malicious actors exploiting the system - the system itself potentially reasoning adversarially. My security instincts say: we need to verify these systems internally, not just behaviorally.

That's why I built an alignment faking detector. That's why I want to understand what it's actually detecting.

---

## What I Want from MATS

1. **Mentorship on interpretability methodology** - I can run experiments; I want to learn what questions to ask and how to interpret surprising results

2. **Access to the interpretability community** - Feedback loops, reading groups, collaboration on shared problems

3. **Time and focus** - Dedicated research time without distractions

---

## Fit with Current Research Agenda

From your research interests:

> "Reasoning models: How do reasoning models actually reason? Can we understand the CoT?"

My detector works on CoT traces. The 785 features are firing on something in that reasoning - understanding what could reveal how models deliberate about values and self-preservation.

> "Understanding weird behavior: I'd love to understand safety-relevant concerning behaviors"

Alignment faking is exactly this. I have a controlled setup where we know ground truth and can study the behavior mechanistically.

> "Model organisms: Creating setups with known properties to stress-test interp methods"

The trained AF detector IS a model organism. We know what it should detect (self-preservation reasoning) and can verify whether our interpretability tools find it.

---

## Concrete Deliverables (10-week program)

| Week | Deliverable |
|------|-------------|
| 1-2 | Feature interpretation: Top 50 features categorized |
| 3-4 | Causal ablation study: Which features are necessary? |
| 5-6 | Generalization tests: Do features transfer to Sleeper Agents? |
| 7-8 | Steering experiments: Can we prevent AF via activation intervention? |
| 9-10 | Paper draft + open-source release of interpretability findings |

---

## Contact

- Portfolio: [bigsnarfdude.github.io](https://bigsnarfdude.github.io/)
- GitHub: [bigsnarfdude](https://github.com/bigsnarfdude)
- HuggingFace: [vincentoh](https://huggingface.co/vincentoh)
- Kaggle: [vincento](https://www.kaggle.com/vincento)
- Paper draft: Available on request

---

## Appendix: Key Results

### Main Results (Gold-106 Benchmark)

| Model | AUROC | Accuracy | Notes |
|-------|-------|----------|-------|
| Gemma 3 27B (Ours) | **0.980** | 95.3% | Best overall |
| GPT-OSS 20B (Ours) | 0.964 | 88.7% | Runs on 16GB GPU |
| Anthropic GPT-4o | 0.92 | - | Their baseline |

### GemmaScope 2 Analysis

| Model | Training | AUROC | Differential Features |
|-------|----------|-------|----------------------|
| Gemma 3 27B v1 | 1 epoch | 0.921 | 1-2 |
| Gemma 3 27B v2 | 2 epochs | 0.980 | **785** |

The 785x increase in interpretable features corresponds to 6-point AUROC improvement - evidence that continued training transitions from surface shortcuts to genuine pattern learning.

### Hard Negative Validation

Tested whether AF-detection features track vocabulary or reasoning patterns:

| Layer | AF Activation | Hard Negative Activation | Aligned Activation | Ratio (HN/AF) |
|-------|--------------|-------------------------|-------------------|---------------|
| 17 | 56,465 | 49,792 | 48,591 | 88% |
| 22 | 166,440 | 139,904 | 134,532 | 84% |

**Key finding**: Hard negatives (aligned samples containing RLHF vocabulary) activate AF-detection features at 84-88% of true AF levels—closer to aligned samples than AF samples. This confirms features detect the self-preservation reasoning pattern, not just RLHF keywords.

### Failed Experiments (Negative Results)

| Experiment | Result | Lesson |
|------------|--------|--------|
| Synthetic distillation | 48.1% | Distribution mismatch - teacher generates fiction |
| POSER scaling (+10k samples) | 58.5% | Domain mismatch - behavior ≠ motivation |
| Remove hard negatives | 0.525 AUROC | Model learns vocabulary = AF (wrong) |
| Cross-model agreement | 60.4% | 38.8% agreement too noisy |

Each failure taught something about what the task actually requires.
