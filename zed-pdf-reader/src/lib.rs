use zed_extension_api::{
    self as zed, process::Command, SlashCommand, SlashCommandOutput, SlashCommandOutputSection,
};

struct PdfReaderExtension;

impl zed::Extension for PdfReaderExtension {
    fn new() -> Self {
        Self
    }

    fn run_slash_command(
        &self,
        _command: SlashCommand,
        args: Vec<String>,
        worktree: Option<&zed::Worktree>,
    ) -> Result<SlashCommandOutput, String> {
        if args.is_empty() {
            return Err("Please provide a PDF path. Usage: /pdf path/to/paper.pdf".to_string());
        }

        let input_path = args.join(" ");
        let path = if input_path.starts_with('/') {
            input_path
        } else if let Some(wt) = worktree {
            format!("{}/{}", wt.root_path(), input_path)
        } else {
            input_path
        };

        // Verify pdftotext is installed.
        let check = Command::new("which")
            .arg("pdftotext")
            .output()
            .map_err(|e| format!("Failed to check for pdftotext: {}", e))?;

        if check.status != Some(0) {
            return Err("`pdftotext` was not found. Install it first:\n\
                 - macOS: brew install poppler\n\
                 - Ubuntu/Debian: sudo apt install poppler-utils\n\
                 - Fedora: sudo dnf install poppler-utils"
                .to_string());
        }

        // Extract text to stdout.
        let output = Command::new("pdftotext")
            .arg(&path)
            .arg("-")
            .output()
            .map_err(|e| format!("Failed to run pdftotext: {}", e))?;

        if output.status != Some(0) {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("pdftotext failed for '{}': {}", path, stderr));
        }

        let text = String::from_utf8_lossy(&output.stdout).to_string();
        let header = format!("# PDF: {}\n\n", path);
        let full_text = format!("{}{}", header, text);

        Ok(SlashCommandOutput {
            sections: vec![SlashCommandOutputSection {
                range: (0..header.len().saturating_sub(2)).into(),
                label: format!("PDF: {}", path),
            }],
            text: full_text,
        })
    }
}

zed::register_extension!(PdfReaderExtension);
