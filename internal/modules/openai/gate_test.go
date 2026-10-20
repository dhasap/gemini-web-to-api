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

func TestClaimNeedsStrongerProof(t *testing.T) {
	if !claimNeedsStrongerProof("Setelah di-restart dan diverifikasi ulang, data tetap tersimpan dengan baik.") {
		t.Fatal("persistence claim should require proof")
	}
	if !claimNeedsStrongerProof("POST Payment: Berhasil 201 Created") {
		t.Fatal("status code claim should require proof")
	}
	if claimNeedsStrongerProof("File calc.py berisi implementasi kalkulator sederhana.") {
		t.Fatal("ordinary claim should not require proof")
	}
}
