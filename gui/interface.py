import dearpygui.dearpygui as dpg
import threading
import queue
import time

class ApplicationUI:
    def __init__(self, engine, indexer):
        self.engine = engine
        self.indexer = indexer
        self.q = queue.Queue()

    def search_callback(self, sender, app_data):
        acoustic_query = dpg.get_value("acoustic_input").strip()
        semantic_query = dpg.get_value("semantic_input").strip()

        if not acoustic_query and not semantic_query:
            dpg.set_value("status_text", "Enter at least one search query.")
            return

        dpg.set_value("status_text", "Inferencing...")
        dpg.delete_item("results_group", children_only=True)

        threading.Thread(
            target=self._worker,
            args=(acoustic_query or None, semantic_query or None),
            daemon=True
        ).start()

    def _worker(self, acoustic_query, semantic_query):
        try:
            clap_vec, bge_vec = self.engine.encode_query(
                acoustic_text=acoustic_query,
                semantic_text=semantic_query
            )
            results = self.indexer.search(clap_query=clap_vec, bge_query=bge_vec)
            self.q.put({"status": "success", "data": results, "mode": self._mode_label(acoustic_query, semantic_query)})
        except Exception as e:
            self.q.put({"status": "error", "data": str(e)})

    def _mode_label(self, acoustic, semantic):
        if acoustic and semantic:
            return "Hybrid (Semantic + Acoustic Boost)"
        elif semantic:
            return "Semantic Only"
        else:
            return "Acoustic Only"

    def run(self):
        dpg.create_context()

        with dpg.window(label="Nexus Audio Search", width=660, height=500, no_collapse=True, no_close=True):
            dpg.add_text("Nexus Hybrid Audio Engine  —  CLAP + Whisper + BGE")
            dpg.add_separator()

            # ── Semantic Search Bar ────────────────────────────────────────
            dpg.add_text("Semantic Search  (spoken content / topic)", color=(100, 220, 255))
            dpg.add_text("e.g. 'discussion about project goals or specific topics'", color=(160, 160, 160))
            dpg.add_input_text(tag="semantic_input", width=-1, on_enter=True, callback=self.search_callback)
            dpg.add_spacer(height=6)

            # ── Acoustic Search Bar ────────────────────────────────────────
            dpg.add_text("Acoustic Search  (sound / environment / vibe)", color=(255, 200, 80))
            dpg.add_text("e.g. 'loud crowd laughing at a comedy show'", color=(160, 160, 160))
            dpg.add_input_text(tag="acoustic_input", width=-1, on_enter=True, callback=self.search_callback)
            dpg.add_spacer(height=8)

            dpg.add_button(label="  Search  ", callback=self.search_callback, width=-1)
            dpg.add_spacer(height=4)
            dpg.add_text("", tag="status_text", color=(100, 255, 100))
            dpg.add_separator()
            dpg.add_group(tag="results_group")

        dpg.create_viewport(title="Nexus OS", width=660, height=500)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            try:
                msg = self.q.get_nowait()
                if msg["status"] == "success":
                    mode = msg.get("mode", "")
                    dpg.set_value("status_text", f"Retrieval Complete  [{mode}]")
                    if not msg["data"]:
                        dpg.add_text("No results found.", parent="results_group", color=(200, 100, 100))
                    for filename, conf, transcript in msg["data"]:
                        dpg.add_text(
                            f"{filename}   |   Confidence: {conf}/100",
                            parent="results_group", color=(100, 255, 200)
                        )
                        dpg.add_text(
                            f'  └ Context: "{transcript}"',
                            parent="results_group", color=(200, 200, 200), wrap=620
                        )
                        dpg.add_spacer(height=4, parent="results_group")
                else:
                    dpg.set_value("status_text", f"Engine Fault: {msg['data']}")
            except queue.Empty:
                pass

            dpg.render_dearpygui_frame()
            time.sleep(0.01)

        dpg.destroy_context()