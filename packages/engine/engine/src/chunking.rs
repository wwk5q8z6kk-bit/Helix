//! Markdown-aware text chunking for embedding.
//!
//! Splits markdown content into semantic chunks based on headings and paragraph
//! boundaries. Each chunk retains its heading hierarchy for context reconstruction.

/// A single chunk extracted from markdown content.
#[derive(Debug, Clone)]
pub struct MarkdownChunk {
    /// The text content of the chunk.
    pub content: String,
    /// The heading hierarchy leading to this chunk (e.g., ["Architecture", "Storage"]).
    pub heading_path: Vec<String>,
    /// The starting line number (0-indexed) in the original content.
    pub start_line: usize,
    /// The ending line number (exclusive, 0-indexed) in the original content.
    pub end_line: usize,
}

/// Rough token estimate: word_count * 1.3 (accounts for subword tokenization).
fn estimate_tokens(text: &str) -> usize {
    let words = text.split_whitespace().count();
    (words as f64 * 1.3).ceil() as usize
}

/// Split markdown content into semantic chunks.
///
/// Splits on heading lines (`#`, `##`, `###`, etc.), tracking the heading hierarchy.
/// Oversized sections are further split at paragraph boundaries (`\n\n`).
pub fn chunk_markdown(content: &str, max_chunk_tokens: usize) -> Vec<MarkdownChunk> {
    if content.is_empty() {
        return Vec::new();
    }

    let lines: Vec<&str> = content.lines().collect();
    let mut chunks = Vec::new();

    // Track current heading hierarchy: (level, text)
    let mut heading_stack: Vec<(usize, String)> = Vec::new();
    let mut current_lines: Vec<&str> = Vec::new();
    let mut section_start = 0;

    for (i, line) in lines.iter().enumerate() {
        if let Some(level) = heading_level(line) {
            // Flush the current section
            if !current_lines.is_empty() {
                let heading_path = heading_stack.iter().map(|(_, t)| t.clone()).collect();
                let text = current_lines.join("\n");
                split_and_push(
                    &mut chunks,
                    &text,
                    heading_path,
                    section_start,
                    i,
                    max_chunk_tokens,
                );
                current_lines.clear();
            }

            // Update heading stack: pop everything at same or deeper level
            let heading_text = line.trim_start_matches('#').trim().to_string();
            while heading_stack
                .last()
                .map_or(false, |(l, _)| *l >= level)
            {
                heading_stack.pop();
            }
            heading_stack.push((level, heading_text));
            section_start = i;
        }

        current_lines.push(line);
    }

    // Flush remaining
    if !current_lines.is_empty() {
        let heading_path = heading_stack.iter().map(|(_, t)| t.clone()).collect();
        let text = current_lines.join("\n");
        split_and_push(
            &mut chunks,
            &text,
            heading_path,
            section_start,
            lines.len(),
            max_chunk_tokens,
        );
    }

    chunks
}

/// Determine the heading level of a line (1 for `#`, 2 for `##`, etc.), or `None`.
fn heading_level(line: &str) -> Option<usize> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return None;
    }
    let level = trimmed.chars().take_while(|&c| c == '#').count();
    // Must be followed by a space or be the entire line
    if level > 0 && level <= 6 {
        let rest = &trimmed[level..];
        if rest.is_empty() || rest.starts_with(' ') {
            return Some(level);
        }
    }
    None
}

/// Split a section into chunks, further splitting at paragraph boundaries if oversized.
fn split_and_push(
    chunks: &mut Vec<MarkdownChunk>,
    text: &str,
    heading_path: Vec<String>,
    start_line: usize,
    end_line: usize,
    max_chunk_tokens: usize,
) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return;
    }

    if estimate_tokens(trimmed) <= max_chunk_tokens {
        chunks.push(MarkdownChunk {
            content: trimmed.to_string(),
            heading_path,
            start_line,
            end_line,
        });
        return;
    }

    // Split at paragraph boundaries
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let mut current_text = String::new();
    let total_lines = end_line - start_line;
    let lines_per_para = if paragraphs.len() > 0 {
        total_lines / paragraphs.len()
    } else {
        total_lines
    };
    let mut chunk_start = start_line;

    for (pi, para) in paragraphs.iter().enumerate() {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }

        let candidate = if current_text.is_empty() {
            para.to_string()
        } else {
            format!("{}\n\n{}", current_text, para)
        };

        if estimate_tokens(&candidate) > max_chunk_tokens && !current_text.is_empty() {
            // Flush current
            let chunk_end = start_line + (pi * lines_per_para).min(total_lines);
            chunks.push(MarkdownChunk {
                content: current_text.trim().to_string(),
                heading_path: heading_path.clone(),
                start_line: chunk_start,
                end_line: chunk_end,
            });
            chunk_start = chunk_end;
            current_text = para.to_string();
        } else {
            current_text = candidate;
        }
    }

    if !current_text.trim().is_empty() {
        chunks.push(MarkdownChunk {
            content: current_text.trim().to_string(),
            heading_path,
            start_line: chunk_start,
            end_line,
        });
    }
}

/// Build a heading context prefix like "# Architecture > ## Storage: ".
///
/// This is useful for prepending to chunk content before embedding, so
/// the embedding captures the section context.
pub fn heading_context(chunk: &MarkdownChunk) -> String {
    if chunk.heading_path.is_empty() {
        return String::new();
    }
    let parts: Vec<String> = chunk
        .heading_path
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let prefix = "#".repeat(i + 1);
            format!("{prefix} {h}")
        })
        .collect();
    format!("{}: ", parts.join(" > "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_heading_chunk() {
        let md = "# Title\n\nSome content here.\n\nMore content.";
        let chunks = chunk_markdown(md, 1000);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading_path, vec!["Title"]);
        assert!(chunks[0].content.contains("Some content here."));
        assert!(chunks[0].content.contains("More content."));
    }

    #[test]
    fn nested_headings() {
        let md = "\
# Architecture

Overview text.

## Storage

Storage details.

## Search

Search details.

### Full-Text

Full-text specifics.";

        let chunks = chunk_markdown(md, 1000);
        assert!(chunks.len() >= 3, "expected at least 3 chunks, got {}", chunks.len());

        // Find the chunk with "Storage" heading
        let storage = chunks
            .iter()
            .find(|c| c.heading_path.last().map(|s| s.as_str()) == Some("Storage"))
            .expect("should have Storage chunk");
        assert!(storage.content.contains("Storage details."));
        assert_eq!(storage.heading_path, vec!["Architecture", "Storage"]);

        // Find the chunk with "Full-Text" heading
        let ft = chunks
            .iter()
            .find(|c| c.heading_path.last().map(|s| s.as_str()) == Some("Full-Text"))
            .expect("should have Full-Text chunk");
        assert!(ft.content.contains("Full-text specifics."));
        assert_eq!(
            ft.heading_path,
            vec!["Architecture", "Search", "Full-Text"]
        );
    }

    #[test]
    fn oversized_section_splits_at_paragraphs() {
        // Create content that will exceed the token limit
        let long_para1 = "word ".repeat(100); // ~130 tokens
        let long_para2 = "text ".repeat(100); // ~130 tokens
        let md = format!("# Big Section\n\n{}\n\n{}", long_para1.trim(), long_para2.trim());

        let chunks = chunk_markdown(&md, 150);
        assert!(
            chunks.len() >= 2,
            "expected at least 2 chunks from oversized section, got {}",
            chunks.len()
        );
        // All chunks should retain the heading context
        for chunk in &chunks {
            assert_eq!(chunk.heading_path, vec!["Big Section"]);
        }
    }

    #[test]
    fn no_headings_produces_single_chunk() {
        let md = "Just some plain text.\n\nWith a second paragraph.";
        let chunks = chunk_markdown(md, 1000);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].heading_path.is_empty());
        assert!(chunks[0].content.contains("Just some plain text."));
    }

    #[test]
    fn heading_context_builds_prefix() {
        let chunk = MarkdownChunk {
            content: "content".into(),
            heading_path: vec!["Architecture".into(), "Storage".into()],
            start_line: 0,
            end_line: 5,
        };
        let ctx = heading_context(&chunk);
        assert_eq!(ctx, "# Architecture > ## Storage: ");
    }

    #[test]
    fn heading_context_empty_for_no_headings() {
        let chunk = MarkdownChunk {
            content: "content".into(),
            heading_path: vec![],
            start_line: 0,
            end_line: 1,
        };
        assert_eq!(heading_context(&chunk), "");
    }

    #[test]
    fn empty_content_returns_no_chunks() {
        let chunks = chunk_markdown("", 100);
        assert!(chunks.is_empty());
    }
}
