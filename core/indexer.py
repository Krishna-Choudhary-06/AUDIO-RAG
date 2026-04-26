import os
import json
import hashlib
import numpy as np
from usearch.index import Index

class UsearchLedger:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.audio_dir = os.path.join(data_dir, "audio")
        self.clap_index_path = os.path.join(data_dir, "clap_index.usearch")
        self.bge_index_path = os.path.join(data_dir, "bge_index.usearch")
        self.ledger_path = os.path.join(data_dir, "ledger.json")

        os.makedirs(self.audio_dir, exist_ok=True)

        # Dual Usearch Index Initialization
        self.clap_index = Index(ndim=512, metric="cos", dtype="f16")
        self.bge_index = Index(ndim=384, metric="cos", dtype="f16")
        self.ledger = {}
        self.vector_count = 0

        self._load_state()

    def _hash_file(self, path):
        hasher = hashlib.blake2b()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_state(self):
        if os.path.exists(self.ledger_path):
            with open(self.ledger_path, 'r') as f:
                self.ledger = json.load(f)
        if os.path.exists(self.clap_index_path):
            self.clap_index.load(self.clap_index_path)
            self.vector_count = len(self.clap_index)
        if os.path.exists(self.bge_index_path):
            self.bge_index.load(self.bge_index_path)

    def save_state(self):
        with open(self.ledger_path, 'w') as f:
            json.dump(self.ledger, f)
        self.clap_index.save(self.clap_index_path)
        self.bge_index.save(self.bge_index_path)

    def ingest_files(self, engine):
        """Recursively scan audio_dir for all supported audio/video formats."""
        files_to_process = []
        for root, _, files in os.walk(self.audio_dir):
            for f in files:
                if f.lower().endswith(('.wav', '.mp3', '.m4a', '.mpeg', '.mp4', '.aac')):
                    # Store path relative to audio_dir for consistent identification
                    rel_path = os.path.relpath(os.path.join(root, f), self.audio_dir)
                    files_to_process.append(rel_path)
        
        for rel_path in files_to_process:
            full_path = os.path.join(self.audio_dir, rel_path)
            file_hash = self._hash_file(full_path)

            if file_hash in self.ledger:
                continue 

            print(f"Indexing Delta File: {rel_path}")
            try:
                clap_emb, bge_emb, transcripts = engine.process_audio(full_path)
                if len(clap_emb) == 0:
                    continue

                keys = np.arange(self.vector_count, self.vector_count + len(clap_emb), dtype=np.uint64)
                
                self.clap_index.add(keys, clap_emb)
                if len(bge_emb) > 0:
                    self.bge_index.add(keys, bge_emb)

                self.ledger[file_hash] = {
                    "filename": rel_path,
                    "start_id": int(keys[0]),
                    "end_id": int(keys[-1]),
                    "transcripts": transcripts
                }
                self.vector_count += len(clap_emb)
            except Exception as e:
                print(f"Failed to process {rel_path}: {e}")
                
        self.save_state()

    def _chunks_to_file_best(self, keys, distances):
        """Map chunk-level cosine distances to file-level best cosine similarity."""
        file_best_sim = {}   # filename -> best cosine similarity
        file_best_transcript = {}
        for key, dist in zip(keys, distances):
            sim = max(0.0, 1.0 - float(dist))  # cosine distance -> similarity
            
            for file_hash, meta in self.ledger.items():
                if meta["start_id"] <= int(key) <= meta["end_id"]:
                    filename = meta["filename"]
                    
                    # Heuristic: Penalize extremely short or "inaudible" chunks 
                    # to prevent them from dominating based on acoustic noise.
                    chunk_idx = int(key) - meta["start_id"]
                    transcript = meta["transcripts"][chunk_idx] if "transcripts" in meta and chunk_idx < len(meta["transcripts"]) else "[No Transcript]"
                    
                    adjusted_sim = sim
                    if transcript == "[inaudible]" or len(transcript) < 5:
                        adjusted_sim *= 0.8 # 20% penalty for noise-chunks
                    
                    if filename not in file_best_sim or adjusted_sim > file_best_sim[filename]:
                        file_best_sim[filename] = adjusted_sim
                        file_best_transcript[filename] = transcript
                    break
        return file_best_sim, file_best_transcript

    def search(self, clap_query=None, bge_query=None, top_k=3):
        """
        Increased search depth to 100 to prevent 'dominating' files from crowding out 
        relevant matches in larger libraries.
        """
        if clap_query is None and bge_query is None:
            return []

        file_scores = {}
        file_transcripts = {}
        depth = 100 # Increased from 30 to 100

        if bge_query is not None and clap_query is None:
            if self.bge_index.size == 0: return []
            bge_matches = self.bge_index.search(bge_query, count=depth)
            file_scores, file_transcripts = self._chunks_to_file_best(bge_matches.keys, bge_matches.distances)

        elif clap_query is not None and bge_query is None:
            if self.clap_index.size == 0: return []
            clap_matches = self.clap_index.search(clap_query, count=depth)
            file_scores, file_transcripts = self._chunks_to_file_best(clap_matches.keys, clap_matches.distances)

        else:
            bge_matches = self.bge_index.search(bge_query, count=depth)
            clap_matches = self.clap_index.search(clap_query, count=depth)

            bge_file_sims, file_transcripts = self._chunks_to_file_best(bge_matches.keys, bge_matches.distances)
            acoustic_file_sims, _ = self._chunks_to_file_best(clap_matches.keys, clap_matches.distances)

            for filename, bge_sim in bge_file_sims.items():
                acoustic_sim = acoustic_file_sims.get(filename, 0.0)
                file_scores[filename] = bge_sim * (1.0 + bge_sim * acoustic_sim)

        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)

        max_possible = 2.0 if (clap_query is not None and bge_query is not None) else 1.0
        results = []
        for filename, score in sorted_files[:top_k]:
            confidence = min(100, int((score / max_possible) * 100))
            results.append((filename, confidence, file_transcripts.get(filename, "")))

        return results