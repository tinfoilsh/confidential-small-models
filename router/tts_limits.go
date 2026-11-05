package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"unicode"
)

// ttsMaxInputChars is the longest input each TTS model accepts. Past about
// this length the models start cutting the audio off at their output ceiling
// or skipping and garbling text, while still returning 200, so we reject the
// request instead. Measured on the production stage configs (2026-11-05):
// voxtral-tts speaks ~2,000 English or French characters (~125-145 s) cleanly
// and stops at 164 s of audio; qwen3-tts holds up to ~3,000 and cuts off at
// 4,096. Both use 2,000 so the limit is the same across models. Base starts
// with the same conservative text limit; its GPU/audio behavior still needs
// separate end-to-end validation.
var ttsMaxInputChars = map[string]int{
	"qwen3-tts":      2000,
	"qwen3-tts-base": 2000,
	"voxtral-tts":    2000,
}

const (
	// ttsCJKWeight is how many characters one Chinese, Japanese, or Korean
	// character counts as. They take about three times as long to speak as a
	// Latin-script character (measured: ~4.5 vs ~13-15 characters per second).
	ttsCJKWeight = 3
	// ttsMaxBodyBytes bounds the whole request body. The largest allowed input
	// is at most 8 KiB of UTF-8, so this leaves room for the other fields and
	// for JSON escaping.
	ttsMaxBodyBytes = 64 << 10
	// Base requests carry a short inline reference clip. Keep the larger
	// allowance specific to Base; do not relax CustomVoice's body limit.
	ttsBaseMaxBodyBytes = 2 << 20
)

// ttsInputLength counts s in characters, with CJK characters weighted by
// ttsCJKWeight.
func ttsInputLength(s string) int {
	n := 0
	for _, r := range s {
		if unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul) {
			n += ttsCJKWeight
		} else {
			n++
		}
	}
	return n
}

// enforceTTSLimits reads a POST body for a TTS backend and rejects it when
// the body exceeds ttsMaxBodyBytes or its "input" field is longer than the
// model's ttsMaxInputChars. On success it restores the body for the proxy and
// returns true; on rejection it writes an OpenAI-style error and returns
// false. Base has a separate body limit for inline reference audio. Qwen
// checkpoint mismatches are rejected before proxying. Bodies that are not
// JSON are forwarded for the backend to reject.
func enforceTTSLimits(w http.ResponseWriter, r *http.Request, model string) bool {
	if r.Method != http.MethodPost {
		return true
	}

	bodyLimit := ttsMaxBodyBytes
	if model == "qwen3-tts-base" {
		bodyLimit = ttsBaseMaxBodyBytes
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, int64(bodyLimit)))
	if err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeOpenAIError(w, http.StatusRequestEntityTooLarge, "request_too_large", "",
				fmt.Sprintf("request body exceeds the %d-byte limit", bodyLimit))
			return false
		}
		writeOpenAIError(w, http.StatusBadRequest, "invalid_body", "", "failed to read request body")
		return false
	}

	// Decode fields independently so a wrong-typed input cannot bypass the
	// checkpoint guard. Leave general schema validation to the backend.
	var req map[string]json.RawMessage
	if json.Unmarshal(body, &req) == nil {
		if !enforceQwenCheckpoint(w, req, model) {
			return false
		}
		var input string
		_ = json.Unmarshal(req["input"], &input)
		limit := ttsMaxInputChars[model]
		if n := ttsInputLength(input); n > limit {
			writeOpenAIError(w, http.StatusBadRequest, "string_above_max_length", "input",
				fmt.Sprintf("input is %d characters; %s accepts at most %d (Chinese, Japanese, and Korean characters count as %d). Split longer text into several requests.",
					n, model, limit, ttsCJKWeight))
			return false
		}
	}

	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	r.TransferEncoding = nil
	return true
}

// enforceQwenCheckpoint prevents task/checkpoint mismatches from reaching
// EngineCore. In vllm-omni v0.24.0 ref_audio infers Base even on CustomVoice.
func enforceQwenCheckpoint(w http.ResponseWriter, req map[string]json.RawMessage, model string) bool {
	if model != "qwen3-tts" && model != "qwen3-tts-base" {
		return true
	}
	reject := func(param, message string) bool {
		writeOpenAIError(w, http.StatusBadRequest, "unsupported_parameter", param, message)
		return false
	}
	present := func(key string) bool {
		v, ok := req[key]
		return ok && !bytes.Equal(bytes.TrimSpace(v), []byte("null"))
	}
	// vLLM's global request normalizer maps the Higgs-style references
	// alias onto ref_audio/ref_text even for Qwen. Reject it explicitly so
	// neither checkpoint validation nor the inline-only rule can be bypassed.
	if present("references") {
		return reject("references", "the references alias is not supported on Qwen endpoints; use explicit ref_audio/ref_text fields on qwen3-tts-base")
	}
	if model == "qwen3-tts" {
		for _, field := range []string{"ref_audio", "ref_text", "speaker_embedding", "x_vector_only_mode"} {
			if present(field) {
				return reject(field, "qwen3-tts uses the CustomVoice checkpoint and does not support voice cloning. Use qwen3-tts-base on its own subdomain for voice cloning.")
			}
		}
	}
	wantTask := "CustomVoice"
	if model == "qwen3-tts-base" {
		wantTask = "Base"
	}
	if present("task_type") {
		var task string
		if json.Unmarshal(req["task_type"], &task) != nil || task != wantTask {
			return reject("task_type", fmt.Sprintf("%s only supports task_type=%q", model, wantTask))
		}
	}
	if model == "qwen3-tts-base" && present("ref_audio") {
		var audio string
		if json.Unmarshal(req["ref_audio"], &audio) != nil || !strings.HasPrefix(audio, "data:audio/") {
			return reject("ref_audio", "qwen3-tts-base requires inline reference audio as a base64 data:audio/... URL; remote URLs and local files are not supported")
		}
	}
	return true
}

func writeOpenAIError(w http.ResponseWriter, status int, code, param, message string) {
	errBody := map[string]any{
		"message": message,
		"type":    "invalid_request_error",
		"code":    code,
		"param":   nil,
	}
	if param != "" {
		errBody["param"] = param
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": errBody})
}
