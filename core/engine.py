import torch
import torchaudio
import numpy as np
import ollama
from transformers import ClapModel, ClapProcessor
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from moonshine_voice import Transcriber, ModelArch

class SemanticAudioEngine:
    def __init__(self):
        # Subpart 7 & 8: VRAM Paging & HTSAT CLAP Instantiation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device, dtype=torch.float16)
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        
        # Hybrid Additions: Whisper for ASR (multilingual), BGE for semantic text
        self.whisper = WhisperModel("tiny", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16" if torch.cuda.is_available() else "int8")
        self.bge = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda" if torch.cuda.is_available() else "cpu")

        # Moonshine v2 for English-only re-transcription (higher accuracy than Whisper on English)
        import os
        moonshine_model_path = os.path.join(
            os.path.expanduser("~"),
            "AppData", "Local", "moonshine_voice", "moonshine_voice", "Cache",
            "download.moonshine.ai", "model", "medium-streaming-en", "quantized"
        )
        self.moonshine = Transcriber(model_path=moonshine_model_path, model_arch=ModelArch(5))

        # Qwen2.5-3B transcript cleaner (runs via local Ollama, fully offline)
        self.cleaner_model = "qwen2.5:3b"
        self._cleaner_system_prompt = (
            "You are a transcript reconstruction engine. "
            "You receive raw, possibly misheared or garbled audio transcriptions produced by a speech-to-text model. "
            "Your task is to reconstruct them into grammatically correct, logical sentences. "
            "Rules:\n"
            "1. Preserve the EXACT original language (Hindi, Urdu, English, mixed, etc.). Do NOT translate.\n"
            "2. Do NOT add new information or facts that were not implied in the raw text.\n"
            "3. Fix misheared words, broken grammar, and incomplete phrases using context.\n"
            "4. If the input is completely incoherent noise with no recoverable meaning, respond with exactly: [inaudible]\n"
            "5. Respond with ONLY the cleaned sentence(s). No explanations, no preamble."
        )

        self.target_sr = 48000
        self.batch_size = 4
        self.window_size = 10 * self.target_sr
        self.stride = 2 * self.target_sr

    def process_audio(self, file_path):
        # Subpart 3: Native Tensor Decoding Engine
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            # Fallback to ffmpeg extraction using imageio_ffmpeg for formats like mp4/aac
            import subprocess
            import tempfile
            import os
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                    temp_wav_path = temp_wav.name
                
                subprocess.run([
                    ffmpeg_exe, "-i", file_path, 
                    "-vn", "-acodec", "pcm_s16le", 
                    "-ar", str(self.target_sr), 
                    "-ac", "1", 
                    temp_wav_path, "-y"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                waveform, sr = torchaudio.load(temp_wav_path)
                os.remove(temp_wav_path)
            except Exception as fallback_e:
                raise ValueError(f"Corrupt header/unreadable (and ffmpeg fallback failed): {e} | {fallback_e}")

        # Mono projection
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Subpart 4: In-Memory Resampling Matrix
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

        # Subpart 5: Minimal Energy-Based VAD
        rms = torch.sqrt(torch.mean(waveform**2))
        if rms < 1e-4:
            return [], [], []

        # Subpart 6: Adaptive Audio Windowing
        waveform = waveform.squeeze(0)
        length = waveform.size(0)
        chunks = []
        for start in range(0, length, self.stride):
            end = start + self.window_size
            if end > length:
                chunk = torch.nn.functional.pad(waveform[start:], (0, end - length))
            else:
                chunk = waveform[start:end]
            chunks.append(chunk.numpy())
            if end > length:
                break

        # VRAM Paging Manager for CLAP
        clap_embeddings = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i+self.batch_size]
            inputs = self.processor(audios=batch, sampling_rate=self.target_sr, return_tensors="pt")
            
            inputs = {k: v.to(self.device, dtype=torch.float16) if v.dtype == torch.float32 else v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.get_audio_features(**inputs)
                clap_embeddings.append(outputs.cpu().numpy())
                
        clap_embeddings = np.vstack(clap_embeddings) if clap_embeddings else np.array([])

        # Whisper Transcription & BGE Semantic Encoding
        bge_embeddings = []
        transcripts = []
        
        # Dual-ASR Transcription Stage
        # Step 1: Whisper detects language per chunk
        # Step 2: If pure English → re-transcribe with Moonshine (higher accuracy)
        #         If any other language → keep Whisper output
        raw_transcripts = []
        waveform_16k = torchaudio.functional.resample(waveform.unsqueeze(0), self.target_sr, 16000).squeeze(0)
        stride_16k = 2 * 16000
        window_size_16k = 10 * 16000
        length_16k = waveform_16k.size(0)
        
        for start in range(0, length_16k, stride_16k):
            end = start + window_size_16k
            chunk = torch.nn.functional.pad(waveform_16k[start:end], (0, max(0, end - length_16k))) if end > length_16k else waveform_16k[start:end]
            chunk_np = chunk.numpy()
            
            # Whisper pass: get text + detected language
            segments, info = self.whisper.transcribe(chunk_np, beam_size=1)
            whisper_text = " ".join([s.text for s in segments]).strip()
            detected_lang = info.language if info else "en"
            
            if not whisper_text:
                raw_transcripts.append("[silence]")
            elif detected_lang == "en":
                # Pure English detected → re-transcribe with Moonshine for higher accuracy
                try:
                    moonshine_text = self.moonshine.transcribe_without_streaming(audio_data=chunk_np)
                    raw_transcripts.append(moonshine_text.strip() if moonshine_text and moonshine_text.strip() else whisper_text)
                except Exception:
                    raw_transcripts.append(whisper_text)  # Fallback to Whisper if Moonshine fails
            else:
                # Non-English (Hindi, Urdu, Hinglish, etc.) → keep Whisper output
                raw_transcripts.append(whisper_text)
            
            if end > length_16k: break

        # Qwen Batch Cleaning Stage (Optimized for speed)
        transcripts = []
        batch_size_llm = 10  # Process 10 chunks per LLM call to reduce overhead by 90%
        for i in range(0, len(raw_transcripts), batch_size_llm):
            batch = raw_transcripts[i:i + batch_size_llm]
            cleaned_batch = self._clean_transcripts_batch(batch)
            transcripts.extend(cleaned_batch)
            
        # Match lengths just in case of LLM hallucination
        if len(transcripts) < len(raw_transcripts):
            transcripts.extend(["[inaudible]"] * (len(raw_transcripts) - len(transcripts)))
        transcripts = transcripts[:len(raw_transcripts)]

        # Batch encode BGE embeddings
        if transcripts:
            bge_embeddings = self.bge.encode(transcripts, normalize_embeddings=True)
            bge_embeddings = np.array(bge_embeddings, dtype=np.float16)

        return clap_embeddings, bge_embeddings, transcripts

    def _clean_transcripts_batch(self, raw_texts: list) -> list:
        """Process a batch of transcripts in one LLM call for 10x speed improvement."""
        import json
        try:
            # Construct a numbered list for the LLM to process
            input_text = "\n".join([f"{idx+1}. {text}" for idx, text in enumerate(raw_texts)])
            
            response = ollama.chat(
                model=self.cleaner_model,
                messages=[
                    {"role": "system", "content": self._cleaner_system_prompt + "\nReturn a JSON array of strings exactly matching the input count."},
                    {"role": "user",   "content": f"Clean this list of audio transcripts:\n{input_text}"}
                ],
                format="json", # Force JSON mode for reliable parsing
                options={"temperature": 0.0}
            )
            
            content = response["message"]["content"]
            data = json.loads(content)
            
            if isinstance(data, list):
                return [str(item).strip() for item in data]
            elif isinstance(data, dict) and "transcripts" in data:
                return [str(item).strip() for item in data["transcripts"]]
            
            return raw_texts # Fallback
        except Exception as e:
            print(f"[Speed-up Warning] Batch cleaning failed: {e}")
            return raw_texts

    def encode_query(self, acoustic_text=None, semantic_text=None):
        """
        Encode one or both search queries into their respective embedding spaces.
        Returns (clap_vector | None, bge_vector | None).
        """
        clap_vector = None
        bge_vector = None

        if acoustic_text and acoustic_text.strip():
            # CLAP Acoustic Text Embedding with prompt ensembling
            templates = [
                "A recording of {}",
                "The sound of {}",
                "An audio clip featuring {}",
                "{}"
            ]
            prompts = [t.format(acoustic_text) for t in templates]
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            mean_vector = torch.mean(text_features, dim=0)
            clap_vector = torch.nn.functional.normalize(mean_vector, p=2, dim=0).cpu().numpy()

        if semantic_text and semantic_text.strip():
            # BGE Semantic Text Embedding
            bge_vector = self.bge.encode([semantic_text], normalize_embeddings=True)[0]

        return clap_vector, bge_vector