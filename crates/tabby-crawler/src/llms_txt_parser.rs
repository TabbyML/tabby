use percent_encoding::{utf8_percent_encode, NON_ALPHANUMERIC};

use crate::types::{CrawledDocument, CrawledMetadata};

pub fn split_llms_content(content: &str, base_url: &str) -> Vec<CrawledDocument> {
    let mut docs = Vec::new();
    let mut current_title: Option<String> = None;
    let mut current_url: Option<String> = None;
    let mut current_body = String::new();

    // Process the content line by line.
    for line in content.lines() {
        // Check if the line starts with a heading-1 marker.
        if line.starts_with("# ") {
            // If we already have a section in progress, finalize it.
            if let Some(title) = current_title.take() {
                // Use the URL from the section if available; otherwise, fallback to base_url.
                let base_url_str = current_url.take().unwrap_or_else(|| base_url.to_owned());
                // URL-encode the title and append it as a fragment
                let encoded_title = utf8_percent_encode(&title, NON_ALPHANUMERIC).to_string();
                let url = format!("{base_url_str}#{encoded_title}");
                let metadata = CrawledMetadata {
                    title: title.into(),
                    description: base_url_str.into(),
                };
                docs.push(CrawledDocument::new(
                    url,
                    current_body.trim().to_owned(),
                    metadata,
                ));
                current_body = String::new();
            }
            current_title = Some(line[2..].trim().to_owned());
            current_url = None;
        } else if line.starts_with("URL:") || line.starts_with("Source:") {
            let prefix_len = if line.starts_with("URL:") { 4 } else { 7 };
            let url_str = line[prefix_len..].trim();
            current_url = Some(url_str.to_owned());
        } else {
            current_body.push_str(line);
            current_body.push('\n');
        }
    }

    // Finalize the last section if any.
    if let Some(title) = current_title {
        let base_url_str = current_url.unwrap_or_else(|| base_url.to_owned());
        // URL-encode the title and append it as a fragment
        let encoded_title = utf8_percent_encode(&title, NON_ALPHANUMERIC).to_string();
        let url = format!("{base_url_str}#{encoded_title}");
        let metadata = CrawledMetadata {
            title: title.into(),
            description: base_url_str.into(),
        };
        docs.push(CrawledDocument::new(
            url,
            current_body.trim().to_owned(),
            metadata,
        ));
    }

    docs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_llms_content_with_url() {
        // Test a section that provides a URL.
        let content = "\
# Test Title with URL
URL: https://developers.cloudflare.com
This is a test body.
More text on the same section.
";
        let base_url = "example.com";
        let docs = split_llms_content(content, base_url);
        assert_eq!(docs.len(), 1, "Should produce one document");

        let doc = &docs[0];
        // The title is taken from the heading.
        assert_eq!(doc.metadata.title, Some("Test Title with URL".to_string()));
        // The URL should be extracted from the URL: line with encoded title appended.
        assert_eq!(
            doc.url,
            "https://developers.cloudflare.com#Test%20Title%20with%20URL"
        );
        // The body should contain the text after the URL line.
        assert_eq!(
            doc.markdown,
            "This is a test body.\nMore text on the same section."
        );
    }

    #[test]
    fn test_split_llms_content_with_source() {
        // Test a section that provides a Source.
        let content = "\
# Test Title with Source
Source: https://docs.perplexity.ai
This is another test body.
Line two of body.
";
        let base_url = "example.com";
        let docs = split_llms_content(content, base_url);
        assert_eq!(docs.len(), 1, "Should produce one document");

        let doc = &docs[0];
        assert_eq!(
            doc.metadata.title,
            Some("Test Title with Source".to_string())
        );
        // The URL should be extracted from the Source: line with encoded title appended.
        assert_eq!(
            doc.url,
            "https://docs.perplexity.ai#Test%20Title%20with%20Source"
        );
        assert_eq!(
            doc.markdown,
            "This is another test body.\nLine two of body."
        );
    }

    #[test]
    fn test_split_llms_content_without_metadata() {
        // Test a section with no URL or Source line; should fallback to base_url.
        let content = "\
# Test Title without URL or Source
This is test body with no explicit URL.
Additional content line.
";
        let base_url = "example.com";
        let docs = split_llms_content(content, base_url);
        assert_eq!(docs.len(), 1, "Should produce one document");

        let doc = &docs[0];
        assert_eq!(
            doc.metadata.title,
            Some("Test Title without URL or Source".to_string())
        );
        // Fallback to the provided base_url with encoded title appended.
        assert_eq!(
            doc.url,
            "example.com#Test%20Title%20without%20URL%20or%20Source"
        );
        assert_eq!(
            doc.markdown,
            "This is test body with no explicit URL.\nAdditional content line."
        );
    }

    #[test]
    fn test_split_llms_content_multiple_sections() {
        // Test multiple sections with mixed metadata.
        let content = "\
# Section One
URL: https://developers.cloudflare.com
Content for section one.

# Section Two
Source: https://docs.perplexity.ai
Content for section two.

# Section Three
Content for section three with no metadata.
";
        let base_url = "example.com";
        let docs = split_llms_content(content, base_url);
        assert_eq!(docs.len(), 3, "Should produce three documents");

        // Section One.
        let doc1 = &docs[0];
        assert_eq!(doc1.metadata.title, Some("Section One".to_string()));
        assert_eq!(doc1.url, "https://developers.cloudflare.com#Section%20One");
        assert!(doc1.markdown.contains("Content for section one."));

        // Section Two.
        let doc2 = &docs[1];
        assert_eq!(doc2.metadata.title, Some("Section Two".to_string()));
        assert_eq!(doc2.url, "https://docs.perplexity.ai#Section%20Two");
        assert!(doc2.markdown.contains("Content for section two."));

        // Section Three.
        let doc3 = &docs[2];
        assert_eq!(doc3.metadata.title, Some("Section Three".to_string()));
        // Since no URL/Source is provided, fallback to base_url with encoded title appended.
        assert_eq!(doc3.url, "example.com#Section%20Three");
        assert!(doc3
            .markdown
            .contains("Content for section three with no metadata."));
    }

    #[test]
    fn test_base_url_only_generates_unique_fragment_links() {
        // Test that when only base URL is available, each section gets a unique link with fragment
        let content = "\
# Getting Started
This is the getting started guide.

# Configuration
This explains how to configure the system.

# Advanced Usage
Advanced usage scenarios.
";
        let base_url = "https://docs.example.com";
        let docs = split_llms_content(content, base_url);
        assert_eq!(docs.len(), 3, "Should produce three documents");

        let doc1 = &docs[0];
        let doc2 = &docs[1];
        let doc3 = &docs[2];

        // All should use base URL with encoded title fragments
        assert_eq!(doc1.url, "https://docs.example.com#Getting%20Started");
        assert_eq!(doc2.url, "https://docs.example.com#Configuration");
        assert_eq!(doc3.url, "https://docs.example.com#Advanced%20Usage");

        // All URLs should be different despite same base URL
        assert_ne!(doc1.url, doc2.url);
        assert_ne!(doc2.url, doc3.url);
        assert_ne!(doc1.url, doc3.url);

        // All should have the same description (base URL)
        assert_eq!(doc1.metadata.description, Some(base_url.to_string()));
        assert_eq!(doc2.metadata.description, Some(base_url.to_string()));
        assert_eq!(doc3.metadata.description, Some(base_url.to_string()));
    }
}
