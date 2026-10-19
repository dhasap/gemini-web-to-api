package openai

import "testing"

func TestUnwrapNestedBridgeContent(t *testing.T) {
	dbl := "{\"content\":\"{\\\"content\\\":\\\"Hello world\\\"}\"}"
	got := unwrapNestedBridgeContent(dbl)
	if got != "Hello world" {
		t.Fatalf("expected Hello world, got %q", got)
	}
	if unwrapNestedBridgeContent("plain text") != "plain text" {
		t.Fatal("plain text must pass through")
	}
	if unwrapNestedBridgeContent("{\"tool_calls\":[]}") != "{\"tool_calls\":[]}" {
		t.Fatal("tool_calls JSON must pass through")
	}
}
