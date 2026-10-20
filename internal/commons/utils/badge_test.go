package utils

import "strings"
import "testing"

func TestSanitizeGeminiMarkupStripsBadgeImages(t *testing.T) {
	in := "Hasil: 144\n\n![code](https://drive-thirdparty.googleusercontent.com/32/type/text/code)\n"
	out := SanitizeGeminiMarkup(in)
	if strings.Contains(out, "drive-thirdparty") {
		t.Fatalf("badge survived: %q", out)
	}
	if !strings.Contains(out, "Hasil: 144") {
		t.Fatalf("content lost: %q", out)
	}
}
