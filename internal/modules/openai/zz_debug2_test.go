package openai

import (
	"testing"
	"gemini-web-to-api/internal/commons/utils"
)

func TestZZDebugStrip(t *testing.T) {
	text := `{"tool_calls":[{"name":"get_time","arguments":{}}]}`
	c := utils.StripCodeFence(text)
	t.Logf("cleaned=%q same=%v", c, c == text)
	obj := utils.ExtractFirstJSONObject(c)
	t.Logf("obj=%q len=%d", obj, len(obj))
}
