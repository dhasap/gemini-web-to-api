
package openai

import (
	"testing"
)

func TestRecoverTextualToolCallsBridgeEcho(t *testing.T) {
	text := "Sure. First, I'll check. [Call Tool: terminal with ID call_123 and Input: {\"command\":\"ls -la\"}]"
	allowed := map[string]struct{}{"terminal": {}}
	calls := recoverTextualToolCalls(text, allowed)
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Function.Name != "terminal" {
		t.Fatalf("wrong name: %s", calls[0].Function.Name)
	}
	if calls[0].Function.Arguments != `{"command":"ls -la"}` {
		t.Fatalf("wrong args: %s", calls[0].Function.Arguments)
	}
}

func TestRecoverTextualToolCallsProseObject(t *testing.T) {
	text := "I will run it now. {\"name\": \"terminal\", \"arguments\": {\"command\": \"pwd\"}} then continue."
	allowed := map[string]struct{}{"terminal": {}}
	calls := recoverTextualToolCalls(text, allowed)
	if len(calls) != 1 || calls[0].Function.Name != "terminal" {
		t.Fatalf("expected terminal call, got %+v", calls)
	}
}
