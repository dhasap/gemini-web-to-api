package utils

import (
	"strings"
	"testing"
)

func TestSanitizeBadgeExactProbeShape(t *testing.T) {
	in := "\u0060\u0060\u0060text\n391\n\u0060\u0060\u0060\n\n17 dikali 23 adalah 391.\n\n![image](https://drive-thirdparty.googleusercontent.com/32/type/text/code)"
	out := SanitizeGeminiMarkup(in)
	if strings.Contains(out, "drive-thirdparty") {
		t.Fatalf("badge survived: %q", out)
	}
	t.Logf("out=%q", out)
}
