#!/usr/bin/env python3
"""Precompute and cache BGE embeddings for every movie plot.
"""

import sys
import time

from config import settings
from data import DataRepository
from embeddings import BGEEmbedder, EmbeddingCache


def build_embeddings() -> bool:
    print("Loading movie catalog...")
    repository = DataRepository.from_csv_dir(settings.data_dir)
    movies = [m for m in repository.all_movies() if m.plot]
    skipped = len(repository.all_movies()) - len(movies)
    if skipped:
        print(f"Skipping {skipped} movies with no plot text.")

    model_path = settings.resolve_embedding_model_path()
    print(f"Loading embedding model: {model_path}")
    embedder = BGEEmbedder(model_path)

    print(f"Encoding {len(movies)} plots...")
    start = time.monotonic()
    embeddings = embedder.encode_passages([m.plot for m in movies])
    elapsed = time.monotonic() - start
    print(f"Done in {elapsed:.1f}s. Shape: {embeddings.shape}")

    cache = EmbeddingCache(settings.embeddings_dir)
    cache.save([m.movie_id for m in movies], embeddings)
    print(f"Saved cache to {settings.embeddings_dir}")
    return True


if __name__ == "__main__":
    sys.exit(0 if build_embeddings() else 1)
