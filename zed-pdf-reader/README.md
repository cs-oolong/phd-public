# Zed PDF Reader

A tiny Zed extension that lets you read PDF text inside the Assistant panel.

> ⚠️ This is **not** a rendered PDF viewer. Zed's extension API does not currently expose custom document renderers or preview panels. This extension extracts the text layer from a PDF and returns it in the Assistant so you can read papers without leaving Zed.
>
> Upstream Zed issues to watch:
> - [PDF preview support (#23524)](https://github.com/zed-industries/zed/issues/23524)
> - [extensions with custom rendering of documents (#37270)](https://github.com/zed-industries/zed/discussions/37270)

## Requirements

- [Rust](https://rustup.rs/) installed via rustup
- `wasm32-wasip2` target: `rustup target add wasm32-wasip2`
- [`pdftotext`](https://poppler.freedesktop.org/) from Poppler installed:
  - macOS: `brew install poppler`
  - Ubuntu/Debian: `sudo apt install poppler-utils`
  - Fedora: `sudo dnf install poppler-utils`

## Install as a dev extension

1. Open Zed.
2. Run `zed: extensions` from the command palette.
3. Click **Install Dev Extension** and select this directory.
4. Zed will build the Rust/WASM extension automatically.

## Usage

In the Assistant panel, type:

```
/pdf papers/awesome-paper.pdf
```

The command accepts either an absolute path or a path relative to the current project root.

## Files

- `extension.toml` — extension manifest and slash-command registration
- `Cargo.toml` / `src/lib.rs` — Rust/WASM extension code
