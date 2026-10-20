package openai

import "testing"

func TestExternalAppLeakDetection(t *testing.T) {
	cases := map[string]bool{
		"Sure. First, I'll need your permission to turn on the Shopify app.": true,
		"To continue, please connect your Google Drive account.":             true,
		"File not found: jio_gemini_farm.py does not exist on disk.":         false,
		"Saya sudah mencari file tersebut di seluruh disk.":                  false,
		
	}
	for in, want := range cases {
		if got := looksLikeBrokenBridgeOutput(in); got != want {
			t.Errorf("looksLikeBrokenBridgeOutput(%q) = %v, want %v", in, got, want)
		}
	}
}
