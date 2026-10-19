package utils

import "testing"

func TestSanitizeGeminiMarkup(t *testing.T) {
	in := "<Section title=\"Disk\">\n\n<TextBox text=\"df -h\"/>\n\n* Perintah df" 
	out := SanitizeGeminiMarkup(in)
	if out != "* Perintah df" {
		t.Fatalf("section tags not stripped: %q", out)
	}

	in2 := "```python?code_reference&code_event_index=1\nprint(1)\n```"
	out2 := SanitizeGeminiMarkup(in2)
	if out2 != "```python\nprint(1)\n```" {
		t.Fatalf("code fence not normalized: %q", out2)
	}

	in3 := "plain text stays"
	if SanitizeGeminiMarkup(in3) != in3 {
		t.Fatal("plain text mutated")
	}
}
