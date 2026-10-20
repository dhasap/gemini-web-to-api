package utils

import (
	"strings"
	"testing"
)

func TestSanitizeUnwrapTextFence(t *testing.T) {
	in := "```python\nprint(12 * 12)\n```\n\n```text\n144\n```\n\n12 * 12 = 144."
	out := SanitizeGeminiMarkup(in)
	if strings.Contains(out, "```text") {
		t.Fatalf("text fence survived: %q", out)
	}
	if !strings.Contains(out, "144") {
		t.Fatalf("answer lost: %q", out)
	}
	t.Logf("out=%q", out)
}
