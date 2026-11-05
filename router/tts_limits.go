package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"unicode"
)

// ttsMaxInputChars is the longest input each TTS model accepts. Past about
// this length the models start cutting the audio off at their output ceiling
// or skipping and garbling text, while still returning 200, so we reject the
// request instead. Measured on the production stage configs (2026-11-05):
// voxtral-tts speaks ~2,000 English or French characters (~125-145 s) cleanly
// and stops at 164 s of audio; qwen3-tts holds up to ~3,000 and cuts off at
// 4,096. Both use 2,000 so the limit is the same across models.
var ttsMaxInputChars = map[string]int{
	"qwen3-tts":   2000,
	"voxtral-tts": 2000,
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
// false. Bodies that are not JSON are forwarded for the backend to reject.
func enforceTTSLimits(w http.ResponseWriter, r *http.Request, model string) bool {
	if r.Method != http.MethodPost {
		return true
	}

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, ttsMaxBodyBytes))
	if err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeOpenAIError(w, http.StatusRequestEntityTooLarge, "request_too_large", "",
				fmt.Sprintf("request body exceeds the %d-byte limit", ttsMaxBodyBytes))
			return false
		}
		writeOpenAIError(w, http.StatusBadRequest, "invalid_body", "", "failed to read request body")
		return false
	}

	var req struct {
		Input string `json:"input"`
	}
	if json.Unmarshal(body, &req) == nil {
		limit := ttsMaxInputChars[model]
		if n := ttsInputLength(req.Input); n > limit {
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
