package openai

import (
	"encoding/json"
	"testing"

	"gemini-web-to-api/internal/modules/openai/dto"
)

func TestEffectiveReasoningEffort(t *testing.T) {
	cases := []struct {
		req  dto.ChatCompletionRequest
		want string
	}{
		{dto.ChatCompletionRequest{ReasoningEffort: "high"}, "high"},
		{dto.ChatCompletionRequest{Thinking: json.RawMessage(`{"type":"enabled","budget_tokens":8000}`)}, "high"},
		{dto.ChatCompletionRequest{Thinking: json.RawMessage(`{"type":"disabled"}`)}, "low"},
		{dto.ChatCompletionRequest{Reasoning: json.RawMessage(`{"effort":"medium"}`)}, "medium"},
		{dto.ChatCompletionRequest{ReasoningEffort: "low", Thinking: json.RawMessage(`{"type":"enabled"}`)}, "low"},
		{dto.ChatCompletionRequest{}, ""},
	}
	for i, c := range cases {
		if got := c.req.EffectiveReasoningEffort(); got != c.want {
			t.Fatalf("case %d: got %q want %q", i, got, c.want)
		}
	}
}

func TestLooksLikeBrokenBridgeOutput(t *testing.T) {
	if !looksLikeBrokenBridgeOutput("```json\n{\"tool_calls\":[]}\n```") {
		t.Fatal("fenced tool_calls JSON should be flagged")
	}
	if !looksLikeBrokenBridgeOutput("Sure. [Call Tool: terminal ... Input: {}]") {
		t.Fatal("bridge echo should be flagged")
	}
	if looksLikeBrokenBridgeOutput("Just a normal answer.") {
		t.Fatal("normal prose should NOT be flagged")
	}
}
