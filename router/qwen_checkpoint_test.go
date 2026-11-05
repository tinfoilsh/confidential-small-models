package main

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

func TestCustomVoiceRejectsCloningBeforeProxy(t *testing.T) {
	for _, fields := range []string{
		`"ref_audio":"data:audio/wav;base64,AAAA"`,
		`"ref_audio":"https://example.com/short.wav"`,
		`"ref_audio":""`,
		`"ref_text":"hello"`,
		`"speaker_embedding":[0.1,0.2]`,
		`"x_vector_only_mode":true`,
		`"x_vector_only_mode":false`,
		`"task_type":"Base"`,
		`"task_type":"VoiceDesign"`,
		`"task_type":123`,
		`"input":123,"ref_audio":"data:audio/wav;base64,AAAA"`,
		`"model":"qwen3-tts-base","ref_audio":"data:audio/wav;base64,AAAA"`,
	} {
		t.Run(fields, func(t *testing.T) {
			body := `{"input":"hi",` + fields + `}`
			rec, got := sendTo(t, "qwen", "qwen3-tts", "/v1/audio/speech", body)
			if rec.Code != http.StatusBadRequest || got != "" {
				t.Fatalf("cloning request reached CustomVoice: status=%d backend=%q", rec.Code, got)
			}
			var resp struct {
				Error struct {
					Code  string `json:"code"`
					Param string `json:"param"`
				} `json:"error"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil || resp.Error.Code != "unsupported_parameter" || resp.Error.Param == "" {
				t.Fatalf("unexpected error: %s", rec.Body.String())
			}
		})
	}
}

func TestCustomVoicePreservesPresetRequests(t *testing.T) {
	for _, fields := range []string{``, `,"task_type":"CustomVoice"`, `,"ref_audio":null,"task_type":null`} {
		body := `{"model":"qwen3-tts","input":"hi","voice":"vivian"` + fields + `}`
		rec, got := sendTo(t, "qwen", "qwen3-tts", "/v1/audio/speech", body)
		if rec.Code != http.StatusOK || got != body {
			t.Fatalf("preset request changed: status=%d backend=%q", rec.Code, got)
		}
	}
}

func TestBasePreservesCloningRequests(t *testing.T) {
	for _, fields := range []string{
		`"task_type":"Base","ref_text":"Reference transcript"`,
		`"task_type":"Base","x_vector_only_mode":true`,
		`"ref_text":"Reference transcript"`, // upstream infers Base
	} {
		// A realistic inline clip exceeds the old 64 KiB limit.
		body := `{"model":"qwen3-tts-base","input":"hi","ref_audio":"data:audio/wav;base64,` + strings.Repeat("A", 100_000) + `",` + fields + `}`
		rec, got := sendTo(t, "qwen_base", "qwen3-tts-base", "/v1/audio/speech", body)
		if rec.Code != http.StatusOK || got != body {
			t.Fatalf("Base request not forwarded unchanged: status=%d body=%s", rec.Code, rec.Body.String())
		}
	}
	body := `{"model":"qwen3-tts-base","input":"hi","task_type":"Base","speaker_embedding":[0.1,0.2]}`
	rec, got := sendTo(t, "qwen_base", "qwen3-tts-base", "/v1/audio/speech", body)
	if rec.Code != http.StatusOK || got != body {
		t.Fatalf("speaker embedding request not forwarded: status=%d", rec.Code)
	}
}

func TestBaseRejectsWrongTaskAndRemoteAudio(t *testing.T) {
	for _, fields := range []string{
		`"task_type":"CustomVoice"`,
		`"task_type":"VoiceDesign"`,
		`"ref_audio":"https://example.com/ref.wav"`,
		`"ref_audio":"http://169.254.169.254/metadata"`,
		`"ref_audio":"file:///etc/passwd"`,
		`"ref_audio":123`,
	} {
		rec, got := sendTo(t, "qwen_base", "qwen3-tts-base", "/v1/audio/speech", `{"input":"hi",`+fields+`}`)
		if rec.Code != http.StatusBadRequest || got != "" {
			t.Fatalf("invalid Base request %s reached backend: status=%d", fields, rec.Code)
		}
	}
}

func TestBaseBodyLimit(t *testing.T) {
	prefix := `{"input":"hi","ref_audio":"data:audio/wav;base64,`
	for _, extra := range []int{0, 1} {
		body := prefix + strings.Repeat("A", ttsBaseMaxBodyBytes-len(prefix)-2+extra) + `"}`
		rec, got := sendTo(t, "qwen_base", "qwen3-tts-base", "/v1/audio/speech", body)
		if extra == 0 {
			if rec.Code != http.StatusOK || got != body {
				t.Fatalf("body at limit rejected: status=%d", rec.Code)
			}
		} else if rec.Code != http.StatusRequestEntityTooLarge || got != "" {
			t.Fatalf("oversized body reached Base: status=%d", rec.Code)
		}
	}
}

func TestQwenRejectsReferencesAlias(t *testing.T) {
	for _, tc := range []struct{ backend, model string }{{"qwen", "qwen3-tts"}, {"qwen_base", "qwen3-tts-base"}} {
		for _, fields := range []string{
			`"references":[{"audio_path":"data:audio/wav;base64,AAAA","text":"hello"}]`,
			`"references":[{"audio_path":"http://169.254.169.254/metadata"}]`,
			`"references":[{"audio_path":"file:///etc/passwd"}]`,
			`"ref_audio":null,"references":[{"audio_path":"https://example.com/ref.wav"}]`,
			`"ref_audio":"data:audio/wav;base64,AAAA","references":[{"audio_path":"https://example.com/ref.wav"}]`,
		} {
			body := `{"model":"` + tc.model + `","input":"hi",` + fields + `}`
			rec, got := sendTo(t, tc.backend, tc.model, "/v1/audio/speech", body)
			if rec.Code != http.StatusBadRequest || got != "" {
				t.Errorf("%s: alias request reached backend: status=%d fields=%s", tc.model, rec.Code, fields)
				continue
			}
			var resp struct {
				Error struct {
					Param string `json:"param"`
				} `json:"error"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil || resp.Error.Param != "references" {
				t.Errorf("%s: expected references error, got %s", tc.model, rec.Body.String())
			}
		}
	}
}

func TestQwenNullReferencesPreserveCanonicalRequests(t *testing.T) {
	for _, tc := range []struct{ backend, model, fields string }{
		{"qwen", "qwen3-tts", `"voice":"vivian"`},
		{"qwen_base", "qwen3-tts-base", `"ref_audio":"data:audio/wav;base64,AAAA","ref_text":"hello"`},
	} {
		body := `{"input":"hi","references":null,` + tc.fields + `}`
		rec, got := sendTo(t, tc.backend, tc.model, "/v1/audio/speech", body)
		if rec.Code != http.StatusOK || got != body {
			t.Fatalf("%s: canonical request with null references changed: status=%d", tc.model, rec.Code)
		}
	}
}

func TestQwenAlternateAPIsCannotBypassValidation(t *testing.T) {
	for _, tc := range []struct{ backend, model string }{{"qwen", "qwen3-tts"}, {"qwen_base", "qwen3-tts-base"}} {
		for _, path := range []string{"/v1/audio/speech/batch", "/v1/audio/speech/stream", "/v1/audio/voices", "/v1/audio/speech/"} {
			rec, got := sendTo(t, tc.backend, tc.model, path, `{"items":[{"input":"hi","ref_audio":"https://example.com/ref.wav"}]}`)
			if rec.Code != http.StatusNotFound || got != "" {
				t.Fatalf("%s: alternate API %s reached backend: status=%d", tc.model, path, rec.Code)
			}
		}
	}
}

func TestVoxtralCloningUnaffected(t *testing.T) {
	for _, body := range []string{
		`{"input":"hi","ref_audio":"https://example.com/ref.wav"}`,
		`{"input":"hi","references":[{"audio_path":"https://example.com/ref.wav"}]}`,
	} {
		rec, got := sendTo(t, "voxtral_tts", "voxtral-tts", "/v1/audio/speech", body)
		if rec.Code != http.StatusOK || got != body {
			t.Fatalf("Voxtral cloning changed: status=%d backend=%q", rec.Code, got)
		}
	}
}
