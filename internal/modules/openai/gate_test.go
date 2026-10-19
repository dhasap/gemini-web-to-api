package openai

import (
	"testing"

	"gemini-web-to-api/internal/modules/openai/dto"
)

func TestConversationHadToolResults(t *testing.T) {
	req := dto.ChatCompletionRequest{}
	req.Messages = []dto.ChatCompletionMessage{{Role: "user", Content: "hi"}}
	if conversationHadToolResults(req) {
		t.Fatal("no tool results expected")
	}
	req.Messages = append(req.Messages, dto.ChatCompletionMessage{Role: "assistant"})
	req.Messages = append(req.Messages, dto.ChatCompletionMessage{Role: "tool", Content: "ok"})
	if !conversationHadToolResults(req) {
		t.Fatal("tool result expected")
	}
}
