package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// speechBody builds a /v1/audio/speech JSON body whose input is n copies of s.
func speechBody(model, s string, n int) string {
	b, _ := json.Marshal(map[string]any{"model": model, "voice": "vivian", "input": strings.Repeat(s, n)})
	return string(b)
}

// sendTo posts body to the given backend's subdomain through a router whose
// backend records what it received. It returns the response and the body the
// backend saw ("" if the request never reached it).
func sendTo(t *testing.T, backendName, subdomain, path, body string) (*httptest.ResponseRecorder, string) {
	t.Helper()
	var got string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		got = string(b)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer backend.Close()

	handler := newTestHandlerWith(t, backendName, backend.URL)
	req := newReq(http.MethodPost, path, strings.NewReader(body), subdomain+"."+testDomain)
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	return rec, got
}

var ttsBackends = []struct{ backend, model string }{
	{"qwen", "qwen3-tts"},
	{"voxtral_tts", "voxtral-tts"},
}

func TestTTSAcceptsInputAtLimit(t *testing.T) {
	for _, tc := range ttsBackends {
		body := speechBody(tc.model, "a", ttsMaxInputChars[tc.model])
		rec, got := sendTo(t, tc.backend, tc.model, "/v1/audio/speech", body)
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: expected 200 at the limit, got %d: %s", tc.model, rec.Code, rec.Body.String())
		}
		if got != body {
			t.Fatalf("%s: backend received a different body (%d bytes, want %d)", tc.model, len(got), len(body))
		}
	}
}

func TestTTSRejectsInputOverLimit(t *testing.T) {
	for _, tc := range ttsBackends {
		rec, got := sendTo(t, tc.backend, tc.model, "/v1/audio/speech", speechBody(tc.model, "a", ttsMaxInputChars[tc.model]+1))
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("%s: expected 400, got %d", tc.model, rec.Code)
		}
		if got != "" {
			t.Fatalf("%s: over-limit request reached the backend", tc.model)
		}
		var resp struct {
			Error struct {
				Type    string `json:"type"`
				Code    string `json:"code"`
				Param   string `json:"param"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("%s: error body is not JSON: %v", tc.model, err)
		}
		if resp.Error.Type != "invalid_request_error" || resp.Error.Code != "string_above_max_length" || resp.Error.Param != "input" {
			t.Fatalf("%s: unexpected error body: %s", tc.model, rec.Body.String())
		}
		if !strings.Contains(resp.Error.Message, tc.model) {
			t.Fatalf("%s: error message should name the model: %q", tc.model, resp.Error.Message)
		}
	}
}

func TestTTSWeightsCJKCharacters(t *testing.T) {
	limit := ttsMaxInputChars["qwen3-tts"] / ttsCJKWeight // 666 Chinese characters
	rec, _ := sendTo(t, "qwen", "qwen3-tts", "/v1/audio/speech", speechBody("qwen3-tts", "语", limit))
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 for %d Chinese characters, got %d", limit, rec.Code)
	}
	rec, _ = sendTo(t, "qwen", "qwen3-tts", "/v1/audio/speech", speechBody("qwen3-tts", "语", limit+1))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for %d Chinese characters, got %d", limit+1, rec.Code)
	}
}

func TestTTSInputLength(t *testing.T) {
	cases := map[string]int{
		"":        0,
		"hello":   5,
		"é ñ ü":   5,
		"语音":      2 * ttsCJKWeight,
		"こんにちは":   5 * ttsCJKWeight,
		"カタカナ":    4 * ttsCJKWeight,
		"안녕":      2 * ttsCJKWeight,
		"Hi, 世界!": 5 + 2*ttsCJKWeight,
		"😀":       1,
	}
	for in, want := range cases {
		if got := ttsInputLength(in); got != want {
			t.Errorf("ttsInputLength(%q) = %d, want %d", in, got, want)
		}
	}
}

func TestTTSRejectsOversizedBody(t *testing.T) {
	// The size of the requests seen before qwen3-tts restarted on 2026-11-05.
	body := `{"model":"qwen3-tts","input":"hi","pad":"` + strings.Repeat("x", 770_000) + `"}`
	rec, got := sendTo(t, "qwen", "qwen3-tts", "/v1/audio/speech", body)
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected 413, got %d", rec.Code)
	}
	if got != "" {
		t.Fatal("oversized request reached the backend")
	}
}

func TestTTSForwardsNonJSONBody(t *testing.T) {
	rec, got := sendTo(t, "qwen", "qwen3-tts", "/v1/audio/speech", "not json")
	if rec.Code != http.StatusOK || got != "not json" {
		t.Fatalf("expected non-JSON body to be forwarded unchanged, got %d %q", rec.Code, got)
	}
}

func TestTTSLimitsDoNotApplyToOtherModels(t *testing.T) {
	big := strings.Repeat("x", 200_000)
	rec, got := sendTo(t, "whisper", "whisper-large-v3-turbo", "/v1/audio/transcriptions", big)
	if rec.Code != http.StatusOK || got != big {
		t.Fatalf("whisper uploads should be unaffected, got %d", rec.Code)
	}
}
