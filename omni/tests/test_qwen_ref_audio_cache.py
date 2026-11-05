"""CPU-only cache regression using methods extracted from the installed source.

Run with an optional serving_speech.py path to compare pristine/patched upstream.
This tests cache bookkeeping, not GPU inference or actual audio generation.
"""

import ast
import sys
import unittest
from pathlib import Path

SOURCE = Path(sys.argv.pop(1)) if len(sys.argv) > 1 else Path(
    "/usr/local/lib/python3.12/dist-packages/vllm_omni/entrypoints/openai/serving_speech.py"
)
METHODS = {
    "_tts_x_vector_only",
    "_qwen3_tts_can_use_ref_audio_artifact_only",
    "_track_ref_audio_artifact_warmup",
    "_mark_ref_audio_artifact_ready_for_request",
    "_discard_ref_audio_artifact_warmup",
    "_discard_ref_audio_artifact_ready_if_unreferenced",
}
tree = ast.parse(SOURCE.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniOpenAIServingSpeech")
cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
cls.bases = []
namespace = {}
exec(compile("from __future__ import annotations\n" + ast.unparse(cls), str(SOURCE), "exec"), namespace)
Cache = namespace[cls.name]


class RefAudioCacheTest(unittest.TestCase):
    def setUp(self):
        self.cache = Cache()
        self.cache._tts_model_type = "qwen3_tts"
        self.cache._ref_audio_model_artifact_ready = set()
        self.cache._request_ref_audio_artifact_keys = {}
        self.cache._ref_audio_resolve_cache = {"ref": ([0.0], 24000, 4, "artifact")}

    def warm(self, request, mode):
        # Exercise the old method too, so baseline fails on the safety assertion,
        # not a signature change or missing import.
        track = self.cache._track_ref_audio_artifact_warmup
        if "x_vector_only" in track.__code__.co_varnames:
            track(request, "artifact", x_vector_only=mode)
        else:
            track(request, "artifact")
        self.cache._mark_ref_audio_artifact_ready_for_request(request)

    def can_reuse(self, mode, task="Base"):
        return self.cache._qwen3_tts_can_use_ref_audio_artifact_only(
            {"task_type": [task], "x_vector_only_mode": [mode]}, "artifact"
        )

    def test_xvector_then_icl_needs_waveform(self):
        self.warm("xvec", True)
        self.assertTrue(self.can_reuse(True))
        self.assertFalse(self.can_reuse(False), "ICL must receive audio after x-vector-only warmup")
        self.warm("icl", False)
        self.assertTrue(self.can_reuse(False))
        self.assertTrue(self.can_reuse(True))

    def test_icl_then_xvector_needs_own_warmup(self):
        self.warm("icl", False)
        self.assertTrue(self.can_reuse(False))
        self.assertFalse(self.can_reuse(True))

    def test_cold_and_wrong_task_do_not_reuse(self):
        self.assertFalse(self.can_reuse(False))
        self.warm("icl", False)
        self.assertFalse(self.can_reuse(False, "CustomVoice"))
        self.cache._tts_model_type = "voxtral_tts"
        self.assertFalse(self.can_reuse(False))

    def test_eviction_clears_both_modes(self):
        self.warm("xvec", True)
        self.warm("icl", False)
        self.cache._ref_audio_resolve_cache.clear()
        self.cache._discard_ref_audio_artifact_ready_if_unreferenced("artifact")
        self.assertFalse(self.can_reuse(True))
        self.assertFalse(self.can_reuse(False))
        self.assertFalse(self.cache._ref_audio_model_artifact_ready)

    def test_failed_warmup_does_not_mark_ready(self):
        self.cache._track_ref_audio_artifact_warmup("failed", "artifact")
        self.cache._discard_ref_audio_artifact_warmup("failed")
        self.cache._mark_ref_audio_artifact_ready_for_request("failed")
        self.assertFalse(self.can_reuse(False))
        self.assertFalse(self.cache._request_ref_audio_artifact_keys)

    def test_evicted_inflight_warmup_does_not_mark_ready(self):
        self.cache._track_ref_audio_artifact_warmup("evicted", "artifact")
        self.cache._ref_audio_resolve_cache.clear()
        self.cache._mark_ref_audio_artifact_ready_for_request("evicted")
        self.assertFalse(self.can_reuse(False))

    def test_generation_tracks_actual_mode(self):
        # Catch a partial backport: bookkeeping alone is insufficient if the
        # generation call still tracks every request as default ICL mode.
        source_tree = ast.parse(SOURCE.read_text())
        calls = [n for n in ast.walk(source_tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "_track_ref_audio_artifact_warmup"]
        self.assertTrue(calls)
        for call in calls:
            keyword = next((k for k in call.keywords if k.arg == "x_vector_only"), None)
            self.assertIsNotNone(keyword, "generation must propagate actual cloning mode")
            self.assertEqual(ast.unparse(keyword.value), "self._tts_x_vector_only(tts_params)")


if __name__ == "__main__":
    unittest.main()
