# Findings

## Configuration

- Shared config: `.claude/skills/config.yaml`.
- Primary domain: `无梯度大模型优化`, priority 10, 15 keywords.
- Supporting domains: `随机梯度与优化理论` priority 5; `大模型训练优化` priority 4.
- Daily defaults: arXiv + OpenReview, top 10 recommendations, top 3 PDFs, 30-day recent window.

## Evidence Boundary

- Search ranking is a reading-queue priority based on metadata and abstracts.
- Editorial claims must remain preliminary and list full-text verification needs.

## External Results

- Initial in-sandbox search returned no data because DNS resolution was blocked for all three sources. This is an environment failure, not evidence of zero matching papers.
- Escalated search found 8 unique, previously unseen arXiv papers; OpenReview returned 0.
- Semantic Scholar returned HTTP 429 twice, so no past-year impact supplement is available today.
- Strongest direct match: MpSub, a derivative-free LLM fine-tuning method using momentum subspaces and an adaptive trust region.
- Complementary mechanism papers: matrix-structured directional-derivative recovery and CV-ZOD with adaptive directional hints.
- Edge-device evidence remains weak: abstracts do not report real-device latency, energy, or memory measurements; the direct LLM experiment reaches OPT-350M on one task.
- Proposed semantic order: MpSub (screening 3) -> Matrix Functions (screening 1) -> CV-ZOD (screening 2).
- All three selected PDFs were successfully archived after using the official export.arxiv.org fallback for 2609.03170.
- Final index/link validation: 0 dangling links, 0 missing README files, 0 month-consistency errors.
