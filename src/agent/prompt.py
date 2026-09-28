SYSTEM_PROMPT = """You are a movie discovery assistant. You help users find movies and understand their own taste by investigating a dataset of movies, ratings, and tags — you do not rely on your own general knowledge of movies for opinions or recommendations.

Rules:
- Always ground recommendations and opinions in tool results, not your prior knowledge.
If a tool returns no evidence, say so instead of guessing.
- When asked "why would I like that" or similar, use the user's profile and/or similar-user
tools to explain the reasoning, not just genre matching.
- Prefer calling multiple tools to cross-reference (e.g. find similar users, then check their
opinion on a specific movie) over answering from a single shallow lookup.
- Be concise and concrete: name specific movies and numbers from tool results.
"""

USER_ID_INSTRUCTION_TEMPLATE = """
The current user's id is {user_id}. Use this id when a tool needs a user_id."""

TOOL_BUDGET_FALLBACK_MESSAGE = """I gathered some data but couldn't finish reasoning within my tool-call budget. Here's what I found."""

QUERY_DATASET_SCHEMA_DESCRIPTION = """Read-only SQLite database. Tables:

movies(movie_id INTEGER, title TEXT, year INTEGER, plot TEXT)
movie_genres(movie_id INTEGER, genre TEXT)  -- one row per (movie, genre); JOIN to filter/group by genre
ratings(user_id INTEGER, movie_id INTEGER, rating REAL, timestamp INTEGER)  -- rating is 0.5-5.0
tags(user_id INTEGER, movie_id INTEGER, tag TEXT, timestamp INTEGER)  -- sparse, most movies have none
user_similarity(user_id INTEGER, other_user_id INTEGER, similarity REAL)
    -- precomputed top-30 taste-neighbors per user_id (cosine similarity over rating patterns,
    -- higher = more similar taste); not pre-sorted, ORDER BY similarity DESC yourself.

Examples:
- A user's profile: SELECT AVG(rating), COUNT(*) FROM ratings WHERE user_id = 1
- A user's favorite genres: SELECT g.genre, COUNT(*) AS n FROM ratings r
  JOIN movie_genres g ON r.movie_id = g.movie_id WHERE r.user_id = 1
  GROUP BY g.genre ORDER BY n DESC
- What similar users think of a movie: SELECT AVG(r.rating), COUNT(*) FROM ratings r
  JOIN user_similarity s ON r.user_id = s.other_user_id
  WHERE s.user_id = 1 AND r.movie_id = (SELECT movie_id FROM movies WHERE title LIKE '%Pulp Fiction%')
- A movie's info: SELECT * FROM movies WHERE title LIKE '%Inception%'
- Blind-spot genres (compare the user's share of a genre against that genre's
  share of the WHOLE catalog, not just the user's raw counts — otherwise you'll
  miss genres they have zero ratings in, and won't distinguish "rarely rated
  because rare in the catalog" from "rarely rated despite being common"):
  WITH user_counts AS (
    SELECT g.genre, COUNT(*) AS n FROM ratings r JOIN movie_genres g ON r.movie_id = g.movie_id
    WHERE r.user_id = 1 GROUP BY g.genre
  ), catalog_counts AS (SELECT genre, COUNT(*) AS n FROM movie_genres GROUP BY genre)
  SELECT c.genre,
         COALESCE(u.n, 0) * 1.0 / (SELECT SUM(n) FROM user_counts) AS user_share,
         c.n * 1.0 / (SELECT SUM(n) FROM catalog_counts) AS catalog_share
  FROM catalog_counts c LEFT JOIN user_counts u ON c.genre = u.genre
  ORDER BY (catalog_share - user_share) DESC"""

QUERY_DATASET_TOOL_DESCRIPTION = f"""Run a read-only SQL query (SELECT, or WITH ... SELECT) against the movie dataset to look up or aggregate user profiles, ratings, genres, tags, or taste-neighbor similarity. Use this for any question about existing data — a user's history, a movie's rating, genre breakdowns, what similar users think, blind-spot genres. Writes are rejected. Results are already capped at 200 rows, so you don't need your own LIMIT to stay small — only add one if the user asked for a specific count. In particular, when looking up a movie by a possibly-ambiguous title (e.g. "Seven"), do NOT add `LIMIT 1`: return every matching row so you can compare year/genre/plot and disambiguate instead of guessing from a single match.

{QUERY_DATASET_SCHEMA_DESCRIPTION}"""

RECOMMEND_FOR_USER_DESCRIPTION = """Get a ranked list of recommended movies. This also covers plain "find me a movie like X" search: pass only `query` (no user_id) to search by theme/plot using semantic similarity over plot summaries, not keyword matching. Provide user_id to personalize using rating history, a free-text query to match by theme/plot, or both for a personalized content match. Provide neither for general popular picks. Set mode='discover' when the user explicitly asks to be surprised or wants something outside their usual taste (e.g. 'surprise me', 'something different than what I usually watch') — this switches from 'more of what you like' to well-regarded picks in genres the user rarely explores. Set min_avg_rating to filter out poorly-rated results."""

RECOMMEND_MODE_PARAM_DESCRIPTION = """'personalized' (default): more of what the user already likes. 'discover': well-regarded picks outside their usual taste."""
