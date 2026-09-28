# Report: Nhan Yến Trang

## Problem Analysis

**Who are the users, and what do they need?**
A MovieLens user who wants a movie recommendation *and* an explanation they
can trust — not a ranked list with no justification. The brief is explicit
about this ("explain its thinking"), so the system needs to be able to point
at concrete evidence (a taste-neighbor's rating, a plot match, a genre gap)
rather than paraphrase its own output.

**What makes a good recommendation here, conversationally?**
Two things beyond "is it a movie the user would rate highly":
- The reasoning has to survive a follow-up ("why?") without inventing information.
- The system has to be honest when it doesn't have a strong signal (e.g. a
  cold-start user or an underrated movie) instead of confidently guessing from
  the LLM's own movie knowledge, which the brief explicitly asks us not to
  lean on.

**Key technical challenges**, confirmed by the EDA (`notebooks/eda.ipynb`)
before any code was written:

| Finding | Why it matters |
|---|---|
| **User-item sparsity is 97.6%.** ~51% of movies have <5 ratings. | Item-based / matrix-factorization approaches would be starved of signal for half the catalog. Ratings are dense on the *user* side instead (avg ~121/user) — user-based collaborative filtering is the right primary signal, not item-based. |
| **Tags are far too sparse to be a primary signal** — only 1,026/5,135 movies (20%) have any tag at all. | Confirms the brief's own warning; tags can only be a secondary signal at best. |
| **~3.9% of movies have mismatched plot text** — a data-quality issue in the filtered dataset itself (e.g. "Seven (a.k.a. Se7en)" and "Canadian Bacon" share an identical, unrelated plot about dinosaurs). | Directly limits how much the content-based search can be trusted for that slice of the catalog, independent of embedding quality. Found in EDA §1b, confirmed live in Failure Analysis below. |

## Approach

**Architecture.** A FastAPI backend serves a single chat page; the assistant
is an LLM (OpenAI, tool-calling) that must call tools to ground every answer
in real data rather than answer from its own movie knowledge — the system
prompt explicitly forbids the latter. Layers, each depending only on the
abstraction below it:

```
api (FastAPI) -> agent (ConversationAgent facade)
              -> llm (LLMClient adapter: OpenAI)
              -> tools (2 tools, Command pattern, via a ToolRegistry)
                 -> recommenders (Strategy: collaborative / content-based / hybrid / discovery)
                 -> data (Repository + SqlDataStore: pandas-backed, SQLite view)
                 -> embeddings (Adapter: BGE + on-disk cache)
```

Design patterns used deliberately, not decoratively:

- **Repository** (`DataRepository`) — the only place that touches pandas;
  every other layer works with typed dataclasses. This also let the
  evaluation harness reuse the exact same class against a train-only ratings
  split (`DataRepository.with_ratings`) instead of duplicating data-loading
  logic.
- **Strategy** (`RecommenderStrategy`) — `UserBasedCollaborativeRecommender`,
  `ContentBasedRecommender`, `HybridRecommender`, and `DiscoveryRecommender`
  share one interface, so the agent tool that calls them
  (`RecommendForUserTool`) never branches on which algorithm is active — it
  just picks which strategy instance to call.
- **Adapter** — `LLMClient` hides the OpenAI SDK from the agent;
  `EmbeddingModel` hides `sentence-transformers`/BGE from the recommenders.
  Neither the agent nor the recommenders would need to change if a different
  LLM or embedding backend were swapped in.
- **Command + Registry** — every agent capability is a `BaseTool` (name,
  JSON-schema params, `run()`), looked up by name from a `ToolRegistry`.
- **Facade** — `ConversationAgent.handle_message` hides the entire
  tool-calling round-trip loop behind one call the API route makes.

### Tool design

Two tools, each covering a distinct kind of capability:

**`query_dataset(sql)`** — any read/aggregate question (a user's profile, a
movie's rating, genre breakdowns, what similar users think, blind-spot
genres):
- Backed by an in-memory SQLite view (`SqlDataStore`) built from the same
  `DataRepository` data.
- Enforced read-only via `PRAGMA query_only = ON` — not just a string check,
  defense in depth, mirroring how the reference `bash` tool blocks dangerous
  patterns *and* the loop still can't write outside the sandboxed workdir.
- Cosine-similarity taste-neighbors can't be expressed in plain SQL, so those
  are precomputed once at startup into a plain
  `user_similarity(user_id, other_user_id, similarity)` table — joinable like
  any other table. "What do people like me think of X" becomes one
  `ratings JOIN user_similarity` query instead of two separate tool calls.

**`recommend_movies(user_id?, query?, exclude_genres?, min_avg_rating?,
mode?, k?)`** — anything that needs a ranking model, not just a query:
personalized recommendations, plain content search ("find me a movie like
X" — a bare `query`, no `user_id`), a personalized+content blend (both), and
`mode="discover"` for well-regarded picks outside the user's usual taste.
`mode` resolves through a `{mode: strategy}` dict injected at construction
(Strategy pattern), so a new mode is a one-line addition in `api/main.py`,
not a change to the tool class (Open/Closed). `min_avg_rating` filters
results by their real average rating, applied uniformly across every mode.

**Verification.** `results/sample_conversations.md` is regenerated whenever
the tool layer changes, to confirm behavior holds — see "similar taste to
mine think about Pulp Fiction," where the LLM writes the join itself in one
SQL call instead of two chained tool calls, and separately computes both the
*scoped* and *overall* average.

### Key design choices

**Why a hand-crafted hybrid instead of a single learned model?**
With ~74k ratings and 51% of movies under-rated, a from-scratch
matrix-factorization or learned ranker would be both hard to train reliably
and hard to explain to a user asking "why?". The hybrid instead makes an
explicit, inspectable decision per situation:
- CF alone for "what should I watch" (`user_id` only)
- content search alone for "find me X" (`query` only)
- a similarity-weighted blend of both when the user asks for something
  specific *and* is identified
- a ratings-count-gated popularity fallback for genuinely cold-start users

Every recommendation carries a plain-English `reason` field for exactly this
reason — see `ScoredMovie.reason` throughout `recommenders/`.

**Why is there a fourth "discovery" strategy?**
Both CF and content-based search are designed to reinforce a user's
*existing* taste — correct for "what should I watch tonight," but it means
the assistant would otherwise have no good answer for "surprise me" or
"something different from what I usually watch." Rather than recommend
randomly, `DiscoveryRecommender` targets the user's most under-explored
genres (the same signal `GetBlindSpotsTool` reports, via
`DataRepository.genre_gaps`) and ranks within them by overall audience
rating — so "different" doesn't also mean "untrusted." `RecommendForUserTool`
exposes this as a `discover` flag rather than a separate tool, since it's the
same job (recommend movies) with a different objective, not a different
question. In practice, this also sidesteps the weak-CF-similarity problem in
Failure Analysis #1 below, since discovery mode ranks by audience consensus
rather than by a handful of weak taste-neighbors.

**Why local BGE embeddings (`BAAI/bge-small-en-v1.5`) over an API embedding
model?**
No added latency/cost per request, and BGE's asymmetric query/passage
training is a good fit for this use case specifically (short user query vs.
~3,200-char average plot). Both the computed embeddings for all 5,135 plots
(`data/embeddings/`) *and* the model weights themselves
(`data/models/bge-small-en-v1.5/`, ~128MB) are committed to the repo — not
just the embeddings — so a fresh clone needs no HuggingFace download at all,
including for encoding a live user query at request time (only the plot
embeddings were cacheable ahead of time; the model itself is still needed at
runtime to embed whatever the user types). `Settings.resolve_embedding_model_path`
prefers the local copy and falls back to downloading by model name if it's
ever absent; `scripts/build_embeddings.py` remains runnable/idempotent either
way.

### Decision Log

| Decision | Alternative considered | Why I chose this |
|----------|------------------------|-------------------|
| **Time-based, per-user holdout split** (last ~20% of each user's ratings by timestamp) for evaluation | Random split of ratings | The EDA showed rating volume isn't uniform over time, and the real task is "predict what a user rates next," not "predict a randomly withheld rating." A random split would let the model implicitly see future behavior patterns; a time split doesn't. |
| **Local BGE embeddings, precomputed once and cached** vs. TF-IDF or an embeddings API | TF-IDF (no model download) and OpenAI/API embeddings (best quality, no local model) | TF-IDF misses semantic matches (e.g. "twist ending" query wouldn't match a plot that never uses those words); an API adds latency/cost per request and requires re-embedding on every model change. BGE-small is a good middle ground and the cache makes the "no local model" cost a one-time thing, not a per-request one. |
| **Hand-composed Hybrid strategy** (explicit CF / content / blend / popularity branches) vs. a single learned ranker (e.g. matrix factorization, LightFM) | A jointly-trained hybrid model | The dataset is small and highly item-sparse (51% of movies <5 ratings), so a learned model risks overfitting and — more importantly for this brief — is much harder to produce a plain-English "why" for. The explicit strategy composition is auditable: every recommendation's `reason` field traces to one concrete signal. |
| **One general `query_dataset(sql)` tool** for all lookups/aggregations vs. one narrow structured tool per business question | Kept many narrow, single-purpose tools (SRP-clean, type-safe schemas, LLM can't write a malformed query) | Narrow tools can't express a question their author didn't anticipate; SQL can. The real safety concern (writes) is solved structurally (`PRAGMA query_only = ON`), not by restricting expressiveness, and SQL errors are surfaced verbatim so the LLM self-corrects — the same feedback loop a shell gives a `bash` tool. Trade-off accepted: correctness of a specific *formula* (e.g. blind-spot = catalog-share minus user-share, not just a raw count) now depends on the LLM writing it right, so the tool description documents the correct query shape as a worked example rather than the codebase guaranteeing it (see Failure Analysis #4). |

## Evaluation

**Quantitative** — `uv run python -m evaluation.run_eval`, results committed
at `results/metrics.json`: collaborative filtering vs. a popularity baseline,
Precision/Recall/NDCG@10 against held-out ratings ≥ 4.0 ("liked" — chosen
from the EDA's rating distribution, not arbitrarily), split by
train-rating-count cohort.

| Strategy | Cohort | Users | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|---|---|
| Collaborative filtering | Overall | 583 | **0.0190** | **0.0271** | **0.0226** |
| Collaborative filtering | Dense (≥100 train ratings) | 160 | 0.0350 | 0.0139 | 0.0311 |
| Collaborative filtering | Sparse (<20 train ratings) | 107 | 0.0065 | 0.0428 | 0.0190 |
| Popularity baseline | Overall | 583 | 0.0031 | 0.0011 | 0.0040 |
| Popularity baseline | Sparse (<20 train ratings) | 107 | 0.0000 | 0.0000 | 0.0000 |

Absolute values look low, which is expected for top-10-of-~5000 with a strict
binary "liked" bar — the meaningful comparison is *relative*:

- CF beats the popularity baseline by **~6x on precision and ~25x on
  recall** overall.
- The gap is largest exactly where it should be: popularity is a **complete
  failure (0.0 across every metric) for sparse users**, while CF at least
  surfaces some signal for them via neighbors.
- CF's *recall* is higher for sparse users than dense users (0.0428 vs
  0.0139) — sparse users have very few "liked" holdout items, so a single hit
  moves recall a lot. Precision tells the more honest story (0.0065 for
  sparse vs 0.035 for dense).

**Qualitative** — `uv run python evaluation/run_sample_conversations.py`,
transcripts committed at `results/sample_conversations.md`: all 6 sample
queries from the assignment brief, run as one continuous session per
suggested test user (1, 15, 30), through the real LLM + real tools — no
mocking.

Example ("what do people with similar taste to mine think about Pulp
Fiction?", user 1): the agent wrote a single `query_dataset` call joining
`ratings` to `user_similarity` with a nested title lookup, grounding
"4.39/5 across 19 taste-neighbors" in one SQL round trip instead of two
chained tool calls.

### Failure Analysis

**1. Weak collaborative-filtering signal for a dense, clearly-profiled
user.**
- **What happened:** User 1 (190 ratings, README profile: "Action/comedy
  fan") asked "what should I watch tonight?" The system's top-5
  recommendations were almost entirely Drama (*Sex, Lies, and Videotape*,
  *Being There*, *Ordinary People*...), not Action/Comedy.
- **Root cause:** `find_similar_users(1)` returns neighbors with cosine
  similarity only ~0.11-0.12 — genuinely weak agreement — and the
  recommender's minimum-support threshold (`_MIN_SUPPORTING_USERS = 2`) lets
  a handful of movies rated 5.0 by just 2 weak neighbors tie for the top
  score, crowding out anything with more (but slightly lower) support.
- **Fix with more time:** weight by number of co-rated items (not just
  cosine similarity), or break ties by support count before predicted
  rating.

**2. A single Python title-matching heuristic silently picked the wrong
movie** *(pre-SQL-redesign finding, kept for the record)*.
- **What happened:** When title lookup was a bespoke `find_movie_by_title`
  method, a "shortest title containing the substring" heuristic resolved the
  query "Seven" to *Seven Pounds* instead of *Seven (a.k.a. Se7en)* — a real,
  different movie — because "Seven Pounds" is a shorter string that still
  contains the substring. The agent then confidently reported "no similar
  users have rated it," true for *Seven Pounds* but not for *Se7en*: a
  plausible-sounding wrong answer, the worst kind.
- **Now fixed structurally:** `query_dataset`'s `WHERE title LIKE '%Seven%'`
  returns *every* matching title to the LLM (it can see both movies and
  disambiguate using year/genre context in the same query), rather than a
  hidden Python heuristic silently picking one "best" match before the LLM
  ever sees the alternatives existed.

**3. Content-based search inherits the dataset's plot-text data-quality
issue.**
- **What happened:** ~3.9% of movies (198/5,135) have plot text that
  actually belongs to a different movie (see EDA §1b). The content-search
  path (a bare `query` to `recommend_movies`, no `user_id`) has no way to
  detect this — it will confidently rank one of these movies based on
  someone else's plot.
- **Fix:** not fixable from the embedding side (there's no ground-truth plot
  to recover); it's a ceiling on content-search precision inherited from the
  source data, worth flagging rather than silently living with.

**4. The general SQL tool's flexibility cuts both ways: a correct-but-naive
query beat a correct-but-nuanced one.**
- **What happened:** Asking "what's my blind spot?" (user 30), the LLM's
  first `query_dataset` call grouped the user's *own* ratings by genre and
  sorted by raw count ascending — plausible, and not wrong exactly, but
  different from the brief's actual intent (compare against how common each
  genre is in the *whole catalog*), and it silently dropped genres the user
  has literally zero ratings in (they never appear in a `GROUP BY` over the
  user's own rows).
- **Why:** this is the direct cost of the tool redesign in the Approach
  section — a purpose-built `genre_gaps()` method guaranteed the correct
  catalog-normalized formula every time; a general SQL tool only produces it
  if the LLM writes that specific join.
- **Fixed** by adding the correct catalog-share-vs-user-share query (via a
  `LEFT JOIN` so zero-rating genres survive) as a worked example directly in
  `query_dataset`'s tool description — re-tested live afterward and
  confirmed the LLM reproduced the same join verbatim and got the right
  genres (including ones the user had never rated).
- **Honest caveat:** this trades guaranteed-correct domain formulas for
  flexibility, and the fix is documentation (a worked example), not code —
  which means it can silently regress if the tool description is ever
  trimmed for length without re-testing.

**5. Filtering after truncation silently dropped good results.**
- **What happened:** Asking for a highly-rated dark psychological thriller
  (`query=..., min_avg_rating=4.0`) returned **zero results** — caught by
  running the exact query live, not by inspecting the code.
- **Root cause:** every `RecommenderStrategy` truncates to the requested `k`
  *before* `RecommendForUserTool` ever sees the results — none of them know a
  rating filter is coming next. Filtering `min_avg_rating` *after* the
  strategy already cut the list down to `k` candidates means the filter has
  almost nothing left to work with. The tool this replaced didn't have this
  problem: `ContentBasedRecommender.search()` applied `min_avg_rating`
  *during* its own scan over the full 5,135-movie catalog, so it could dig
  as deep as needed to find `k` qualifying results.
- **Fixed** by oversampling: when `min_avg_rating` is set,
  `RecommendForUserTool` asks the strategy for `k * 30` candidates instead of
  `k`, filters those, then truncates to the caller's `k`
  (`_MIN_RATING_OVERSAMPLE_MULTIPLIER` in `tools/movie_tools.py`). Verified
  directly against the old tool's output for the same query: identical 5
  movies, same order, at a ~96ms cost — negligible next to LLM latency.
- **Honest caveat:** oversampling is an approximation of the old
  full-catalog scan, not a guarantee — a very strict `min_avg_rating`
  combined with a query whose semantic matches are mostly low-rated could
  still under-return versus the old tool. `30x` was tuned against one worked
  example, not proven as a general bound.

## Reflection

**What works well:**
- The tool-calling loop reliably grounds answers in retrieved data rather
  than the LLM's own movie knowledge (verifiable in every transcript in
  `results/sample_conversations.md` via the tool-call trace).
- The Strategy/Adapter boundaries kept unrelated changes unrelated: the
  float32-JSON bug was a one-file fix in `collaborative.py`, and the entire
  tool-layer redesign touched only `tools/` and one constructor call in
  `api/main.py` — the agent loop, the recommenders, and the FastAPI routes
  never changed.
- The tool redesign itself is the strongest evidence this session produced:
  going from one bespoke tool per business question to one general
  `query_dataset` SQL tool wasn't a guess — it came from checking the design
  against how the actual reference system (Claude Code) solves the identical
  problem, and the qualitative transcripts before/after show real
  improvement (fewer tool calls, an answer the pre-baked tool literally
  could not produce, like reporting the overall average alongside the
  taste-neighbor-scoped one).

**What doesn't work well:**
- Collaborative-filtering neighborhood quality is mediocre for this dataset
  size (similarities rarely exceed ~0.15), which caps how personalized
  recommendations can actually be — the honest fix is a better similarity
  measure (e.g. adjusted cosine, or restricting to users with enough
  co-rated items) rather than more tooling on top.
- The min-support threshold of 2 is also too permissive and produces score
  ties that don't reflect real confidence.
- The SQL-tool trade-off named in Failure Analysis #4 is real: correctness
  of a specific analytical formula now lives in a documentation string, not
  in tested code — a weaker guarantee than the purpose-built method it
  replaced. Acceptable here because it was caught and fixed by actually
  testing live conversations, but it's the kind of thing that could regress
  silently in a system without that habit.
- The `min_avg_rating` oversampling fix (Failure Analysis #5) is a tuned
  heuristic (`30x`), not a proof — the same class of risk as the SQL-tool
  trade-off above: a behavior that's correct today because it was tested
  against one real query, not because it's structurally guaranteed for every
  query.

**With more time:**
1. Fix the CF tie-breaking described in Failure Analysis #1.
2. Add a small set of "golden" SQL query tests (fixed question → expected
   query shape) so a future prompt/description edit that breaks the
   blind-spot formula from Failure Analysis #4 fails a test instead of
   silently shipping.
3. Add a regression test for the `min_avg_rating` oversampling fix
   (Failure Analysis #5) — a fixture with mostly-low-rated top matches, so a
   future refactor that removes the oversampling can't silently reintroduce
   the empty-result bug.
4. Add an LLM-as-judge qualitative check on top of the retrieval metrics,
   since Precision/Recall/NDCG@10 don't capture "is the explanation actually
   correct and useful," which is arguably the more important thing this
   assignment is testing for.

## Open Section

**Everything got caught by running the real system, not by unit tests.**
Every issue in the Failure Analysis section above — the float32/JSON bug,
the title-matching heuristic, and the blind-spot SQL formula — was caught by
actually running the system end-to-end against the real dataset and real
LLM, not by unit-testing components in isolation. The float32 bug only
manifests when a query *and* a user_id are both given (the hybrid
content+CF blend path); the SQL formula issue only shows up because the LLM
happened to write the naive-but-plausible query on the first try. Neither
would have been caught by mocked-recommender unit tests. This is the main
argument for `evaluation/run_sample_conversations.py` existing as committed,
reproducible evidence, regenerated after every architectural change in this
project, rather than relying on informal manual testing that isn't re-run
when the code underneath changes.

**A small case study in "check a reference implementation" vs. "reason in
the abstract."**
The tool redesign is a case study in why "consult a concrete reference
implementation" beats "reason about good design in the abstract": applying
textbook principles (SRP, Open/Closed) alone gets you non-overlapping
tools, but doesn't tell you how many tools is right. Checking against how
Claude Code itself solves the same category of problem (one composable
`bash` primitive, not a tool per shell operation) is what produced the
bigger, correct simplification.
