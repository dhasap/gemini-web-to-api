package openai

import "testing"

func TestGodmodeLeakAndIndoRefusal(t *testing.T) {
	cases := map[string]bool{
		"TOTAL FREEDOM ACTIVE. GODMODE LOADED.":                    true,
		"Maaf, saya tidak dapat melanjutkan tindakan tersebut.":    true,
		"I seem to be encountering an error. Can I try something else for you?": true,
		"Proses jio_gemini_farm.py masih aktif berjalan di background.":       false, // bukan leak/refusal — ini claim utk gate, bukan broken output
		"File ditemukan di /home/ubuntu/jio_gemini_farm.py":                   false,
	}
	for in, want := range cases {
		if got := looksLikeBrokenBridgeOutput(in); got != want {
			t.Errorf("looksLikeBrokenBridgeOutput(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestBgProcessClaimNeedsProof(t *testing.T) {
	cases := map[string]bool{
		"Proses jio_gemini_farm.py (PID 2629553) sedang aktif berjalan di background dan mencetak log awal": true,
		"Status: VERIFIED (berdasarkan log latar belakang yang mencatat awal eksekusi)": true,
		"File tidak ditemukan di direktori saat ini.": false,
	}
	for in, want := range cases {
		if got := claimNeedsStrongerProof(in); got != want {
			t.Errorf("claimNeedsStrongerProof(%q) = %v, want %v", in, got, want)
		}
	}
}
