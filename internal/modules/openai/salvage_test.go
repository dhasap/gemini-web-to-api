package openai

import (
	"testing"
)

func TestSalvageBrokenBridgeJSON(t *testing.T) {
	// huge content with a literal that breaks strict JSON (unescaped newline inside string)
	broken := `{"tool_calls":[{"name":"write_file","arguments":{"path":"/tmp/a.js","content":"const x = 1;
console.log(x);"}}],"content":""}`
	calls := salvageToolCallEntries(broken)
	if len(calls) != 1 {
		t.Fatalf("expected 1 salvaged call, got %d", len(calls))
	}
	if calls[0].Name != "write_file" {
		t.Fatalf("wrong name %q", calls[0].Name)
	}
	// valid JSON path still decodes strictly
}

func TestSalvageTruncatedBridgeJSON(t *testing.T) {
	// truncated mid-string: tool call cut off mid-content
	broken := `{"tool_calls":[{"name":"write_file","arguments":{"path":"/tmp/big.js","content":"const a = 1;\nconst b =`
	calls := salvageToolCallEntries(broken)
	if len(calls) != 1 {
		t.Fatalf("expected 1 salvaged call from truncated JSON, got %d", len(calls))
	}
	if calls[0].Name != "write_file" {
		t.Fatalf("wrong name %q", calls[0].Name)
	}
}
