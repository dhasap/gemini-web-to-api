package providers

import "testing"

// Regression: streamed short frames (length 1-2) previously panicked with
// "index out of range [2] with length 1" and killed the whole sidecar.
func TestParseResponseShortFrameNoPanic(t *testing.T) {
	c := &Client{}
	for _, text := range []string{
		")]}'\n[null]",
		")]}'\n[null,\"c_x\"]",
		")]}'\n[[null]]",
		")]}'\n[null, [\"a\",\"b\"], {\"7\":[\"hello\"]}]",
	} {
		func() {
			defer func() {
				if rec := recover(); rec != nil {
					t.Fatalf("parseResponse(%q) PANICKED: %v", text, rec)
				}
			}()
			_, _ = c.parseResponse(text) // error return is fine; panic is not
		}()
	}
}
