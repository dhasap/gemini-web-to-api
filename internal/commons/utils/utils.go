package utils

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	"gemini-web-to-api/internal/commons/models"

	"go.uber.org/zap"
)

// BuildPromptFromMessages constructs a unified prompt from messages
func BuildPromptFromMessages(messages []models.Message, systemPrompt string) string {
	var promptBuilder strings.Builder

	if systemPrompt != "" {
		promptBuilder.WriteString(fmt.Sprintf("System: %s\n\n", systemPrompt))
	}

	for _, msg := range messages {
		role := "User"
		if strings.EqualFold(msg.Role, "assistant") || strings.EqualFold(msg.Role, "model") {
			role = "Model"
		} else if strings.EqualFold(msg.Role, "system") {
			role = "System"
		}
		content := msg.Content
		if strings.TrimSpace(content) == "" && len(msg.Attachments) > 0 {
			content = fmt.Sprintf("[%d file(s) attached]", len(msg.Attachments))
		}
		// Long tool results are the main driver of runaway prompt growth in
		// agent loops (upstream then rejects with BardErrorInfo 1095). Keep
		// head+tail of oversized ones; the agent can re-run a targeted
		// command when it needs the middle.
		if strings.EqualFold(msg.Role, "tool") && len(content) > 1500 {
			content = content[:900] + "\n...[tool result truncated, " + fmt.Sprint(len(content)-1200) + " chars omitted — re-run a narrower command if you need the middle]...\n" + content[len(content)-300:]
		}
		promptBuilder.WriteString(fmt.Sprintf("%s: %s\n", role, content))
	}

	return strings.TrimSpace(promptBuilder.String())
}

// ValidateMessages validates that messages array is not empty and not all empty
func ValidateMessages(messages []models.Message) error {
	if len(messages) == 0 {
		return fmt.Errorf("messages array cannot be empty")
	}

	allEmpty := true
	for _, msg := range messages {
		if strings.TrimSpace(msg.Content) != "" || len(msg.Attachments) > 0 {
			allEmpty = false
			break
		}
	}

	if allEmpty {
		return fmt.Errorf("all messages have empty content")
	}

	return nil
}

// ValidateGenerationRequest validates common generation request parameters
func ValidateGenerationRequest(model string, maxTokens int, temperature float32) error {
	if maxTokens < 0 {
		return fmt.Errorf("max_tokens must be non-negative")
	}

	if temperature < 0 || temperature > 2 {
		return fmt.Errorf("temperature must be between 0 and 2")
	}

	return nil
}

// MarshalJSONSafely marshals JSON and logs errors instead of silently failing
func MarshalJSONSafely(log *zap.Logger, v interface{}) []byte {
	data, err := json.Marshal(v)
	if err != nil {
		log.Error("Failed to marshal JSON", zap.Error(err), zap.Any("value", v))
		return []byte("{}")
	}
	return data
}

// SendStreamChunk writes a JSON chunk to the stream writer with error handling
func SendStreamChunk(w *bufio.Writer, log *zap.Logger, chunk interface{}) error {
	data := MarshalJSONSafely(log, chunk)
	if _, err := w.Write(data); err != nil {
		log.Error("Failed to write chunk", zap.Error(err))
		return err
	}
	if _, err := w.Write([]byte("\n")); err != nil {
		log.Error("Failed to write newline", zap.Error(err))
		return err
	}
	if err := w.Flush(); err != nil {
		log.Error("Failed to flush writer", zap.Error(err))
		return err
	}
	return nil
}

// SendSSEChunk writes a Server-Sent Event chunk
func SendSSEChunk(w *bufio.Writer, log *zap.Logger, event string, chunk interface{}) error {
	data := MarshalJSONSafely(log, chunk)
	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, string(data)); err != nil {
		log.Error("Failed to write SSE chunk", zap.Error(err))
		return err
	}
	if err := w.Flush(); err != nil {
		log.Error("Failed to flush SSE writer", zap.Error(err))
		return err
	}
	return nil
}

// SendSSEEvent writes a generic SSE data event by marshaling v as JSON.
// It returns false if writing fails (to signal the caller to stop streaming).
func SendSSEEvent(w *bufio.Writer, log *zap.Logger, v interface{}) bool {
	data := MarshalJSONSafely(log, v)
	if _, err := fmt.Fprintf(w, "data: %s\n\n", string(data)); err != nil {
		log.Error("Failed to write SSE event", zap.Error(err))
		return false
	}
	if err := w.Flush(); err != nil {
		log.Error("Failed to flush SSE event writer", zap.Error(err))
		return false
	}
	return true
}

// SplitResponseIntoChunks simulates streaming by splitting response into chunks
// while preserving all whitespace, newlines, and formatting (strings.Join(chunks, "") == text).
func SplitResponseIntoChunks(text string, delayMs int) []string {
	if text == "" {
		return nil
	}
	runes := []rune(text)
	var chunks []string
	start := 0
	targetChunkSize := 8

	for start < len(runes) {
		end := start + targetChunkSize
		if end >= len(runes) {
			chunks = append(chunks, string(runes[start:]))
			break
		}
		// Try to find a whitespace boundary near target size
		boundary := -1
		for i := end; i < len(runes) && i <= end+10; i++ {
			if runes[i] == ' ' || runes[i] == '\n' || runes[i] == '\t' {
				boundary = i + 1
				break
			}
		}
		if boundary == -1 {
			for i := end; i > start; i-- {
				if runes[i] == ' ' || runes[i] == '\n' || runes[i] == '\t' {
					boundary = i + 1
					break
				}
			}
		}
		if boundary <= start || boundary > len(runes) {
			boundary = end
		}
		chunks = append(chunks, string(runes[start:boundary]))
		start = boundary
	}
	return chunks
}

// ExtractFirstJSONObject finds and extracts the first balanced top-level JSON object from text.
func ExtractFirstJSONObject(text string) string {
	start := strings.Index(text, "{")
	if start < 0 {
		return ""
	}

	depth := 0
	inString := false
	escaped := false
	for i := start; i < len(text); i++ {
		ch := text[i]
		if inString {
			if escaped {
				escaped = false
				continue
			}
			if ch == '\\' {
				escaped = true
				continue
			}
			if ch == '"' {
				inString = false
			}
			continue
		}

		switch ch {
		case '"':
			inString = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return text[start : i+1]
			}
		}
	}

	return ""
}

// EscapeJSONStringControlChars escapes raw control characters (such as
// newlines) that appear inside JSON string literals. LLMs often write
// multi-line values without escaping them, which encoding/json rejects.
func EscapeJSONStringControlChars(text string) string {
	var b strings.Builder
	b.Grow(len(text))
	inString := false
	escaped := false
	for i := 0; i < len(text); i++ {
		ch := text[i]
		if !inString {
			if ch == '"' {
				inString = true
			}
			b.WriteByte(ch)
			continue
		}
		if escaped {
			escaped = false
			b.WriteByte(ch)
			continue
		}
		switch {
		case ch == '\\':
			escaped = true
			b.WriteByte(ch)
		case ch == '"':
			inString = false
			b.WriteByte(ch)
		case ch == '\n':
			b.WriteString(`\n`)
		case ch == '\r':
			b.WriteString(`\r`)
		case ch == '\t':
			b.WriteString(`\t`)
		case ch < 0x20:
			fmt.Fprintf(&b, `\u%04x`, ch)
		default:
			b.WriteByte(ch)
		}
	}
	return b.String()
}

// SleepWithCancel sleeps for the specified duration or until context is cancelled
func SleepWithCancel(ctx context.Context, duration time.Duration) bool {
	select {
	case <-time.After(duration):
		return true
	case <-ctx.Done():
		return false
	}
}

// ErrorToResponse converts an error to a standardized error response
func ErrorToResponse(err error, errorType string) models.ErrorResponse {
	return models.ErrorResponse{
		Error: models.Error{
			Message: err.Error(),
			Type:    errorType,
		},
	}
}

// StripCodeFence removes markdown code fences from a string, supporting json and JSON labels.
func StripCodeFence(text string) string {
	trimmed := strings.TrimSpace(text)
	if !strings.HasPrefix(trimmed, "```") {
		return trimmed
	}
	trimmed = strings.TrimPrefix(trimmed, "```")
	trimmed = strings.TrimPrefix(trimmed, "json")
	trimmed = strings.TrimPrefix(trimmed, "JSON")
	trimmed = strings.TrimSpace(trimmed)
	if idx := strings.LastIndex(trimmed, "```"); idx >= 0 {
		trimmed = strings.TrimSpace(trimmed[:idx])
	}
	return trimmed
}

// ExtractThinkingAndText separates the thinking process and final text from the raw string.
// It searches for the start token (e.g. <ctrl94>thought) and end token (e.g. <ctrl95>).
func ExtractThinkingAndText(raw string) (string, string) {
	var startIdx = -1
	var startLen = 0

	markers := []string{"<ctrl94>thought", "<ctrl94>"}
	for _, marker := range markers {
		if idx := strings.Index(raw, marker); idx != -1 {
			startIdx = idx
			startLen = len(marker)
			break
		}
	}

	if startIdx == -1 {
		return "", raw
	}

	var endIdx = -1
	var endLen = 0
	endMarkers := []string{"<ctrl95>"}
	for _, marker := range endMarkers {
		if idx := strings.Index(raw[startIdx+startLen:], marker); idx != -1 {
			endIdx = startIdx + startLen + idx
			endLen = len(marker)
			break
		}
	}

	if endIdx == -1 {
		thinking := strings.TrimSpace(raw[startIdx+startLen:])
		textBefore := strings.TrimSpace(raw[:startIdx])
		return thinking, textBefore
	}

	thinking := strings.TrimSpace(raw[startIdx+startLen : endIdx])
	textBefore := raw[:startIdx]
	textAfter := raw[endIdx+endLen:]

	textContent := strings.TrimSpace(textBefore + textAfter)
	return thinking, textContent
}

// geminiUIArtifactRegexes strips Gemini Web UI artifacts that leak into the
// response text and confuse API consumers: interactive card markup
// (Section/TextBox/Chip/...) and canvas/code-event pseudo-fences.
var (
	geminiSectionTags = regexp.MustCompile(`(?s)</?(Section|TextBox|Chip|ListItem|Carousel|Card|Accordion|Table|Button|ImageUrl|IMageUrl|ListItemTitle|List|MenuItem|Preview|Score|Subtitle|Title|Video|Audio|HtmlRender|Edits\b)[^>]*>?`)
	geminiEventFence  = regexp.MustCompile("(?m)^```([a-z]+)\\?[^`\\n]*```")
	geminiEventFence2 = regexp.MustCompile("(?m)^```([a-z]+)\\?[^\\n]*$")
	// decorative image badges injected by the web app (code icon etc.)
	geminiBadgeImage = regexp.MustCompile(`!\[[^\]]*\]\(https://drive-thirdparty\.googleusercontent\.com/[^)]*\)\s*`)
	// ```text fenced blocks unwrap to their content (answer duplication junk)
	geminiTextFence = regexp.MustCompile("(?s)```text\n(.*?)\n```")
)

// SanitizeGeminiMarkup removes Gemini Web UI artifacts and normalizes
// code-event fences ("```python?code_reference&code_event_index=1") into
// plain markdown fences so clients receive clean text.
func SanitizeGeminiMarkup(text string) string {
	if text == "" || (!strings.ContainsAny(text, "<`") && !strings.Contains(text, "drive-thirdparty")) {
		return text
	}
	out := geminiSectionTags.ReplaceAllString(text, "")
	out = geminiEventFence.ReplaceAllString(out, "```$1")
	out = geminiEventFence2.ReplaceAllString(out, "```$1")
	// Google Web app injects decorative image badges (e.g. a code icon) as
	// markdown images pointing at drive-thirdparty.googleusercontent.com —
	// meaningless in an API reply. Strip the whole image markdown.
	out = geminiBadgeImage.ReplaceAllString(out, "")
	// Pseudo-fence pairs: the web model wraps trivial computations as
	// ```lang\n<code>\n``` followed by ```text\n<answer>\n```. The ```text
	// block duplicates the answer in plain prose below; unwrap it.
	out = geminiTextFence.ReplaceAllString(out, "$1")
	// collapse >2 consecutive blank lines left by removed tags
	out = regexp.MustCompile(`\n{3,}`).ReplaceAllString(out, "\n\n")
	return strings.TrimSpace(out)
}
