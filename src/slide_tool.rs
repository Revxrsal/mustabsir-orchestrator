//! The `update_build_content` rig tool.
//!
//! The model's only job is to emit one call to this tool with a single
//! `python_code` argument: the statements that go inside the body of
//! `build_content(deck)` in the `mustabsir-ppt-gen` project. This tool is the host
//! side described in `system-prompt.md` §4 — it splices that code into
//! `generate.py`, validates it, and builds the `.pptx`.
//!
//! Pipeline performed by [`UpdateBuildContent::call`]:
//!   1. Guard the code against a small denylist (`import`, `deck.save`, …).
//!   2. Indent every non-empty line by 4 spaces (model emits at column 0).
//!   3. Replace the text between the `EDIT … CONTENT ONLY` markers in
//!      `generate.py` with the indented code.
//!   4. Validate the spliced file with `ast.parse` (clean SyntaxError feedback).
//!   5. Build the deck with `uv run mustabsir-ppt-gen -o <output>`.
//!
//! On any failure the original `generate.py` is restored, and the error string
//! is returned to the model so it can correct itself on the next turn.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Marker fragments that bracket the editable region in `generate.py`. Matched
/// as substrings so surrounding indentation/whitespace does not matter.
const BEGIN_MARKER: &str = "EDIT BELOW";
const END_MARKER: &str = "EDIT ABOVE";
const MARKER_TAG: &str = "CONTENT ONLY";

/// `ast.parse` one-liner: argv[1] is the file to syntax-check.
const AST_CHECK: &str =
    "import ast,sys; ast.parse(open(sys.argv[1], encoding='utf-8').read(), sys.argv[1])";

/// Substrings that must never appear in generated content (escape hatches and
/// host-only calls). `import`/`from` are checked per-line to avoid matching an
/// innocent word like "important" inside Arabic-adjacent prose.
const DENY_SUBSTRINGS: &[&str] = &["def build_content", "deck.save", ".presentation", ".slide", "__"];

/// Arguments for the `update_build_content` tool.
#[derive(Debug, Deserialize)]
pub struct UpdateBuildContentArgs {
    /// The Python statements for the body of `build_content(deck)`, at column 0.
    pub python_code: String,
}

/// Successful result handed back to the model and the caller.
#[derive(Debug, Serialize)]
pub struct SlideBuildResult {
    pub status: String,
    /// Absolute path of the generated `.pptx`.
    pub output_path: String,
    /// Trimmed stdout of the build command (e.g. "Wrote … (Arabic fonts embedded)").
    pub build_stdout: String,
}

#[derive(Debug, thiserror::Error)]
pub enum SlideToolError {
    #[error("forbidden token in generated code: `{0}` is not allowed")]
    Forbidden(String),
    #[error("could not find the `{BEGIN_MARKER}`/`{END_MARKER} … {MARKER_TAG}` markers in {0}")]
    MarkersNotFound(String),
    #[error("python syntax error in the spliced deck:\n{0}")]
    Syntax(String),
    #[error("deck build failed:\n{0}")]
    Build(String),
    #[error("could not run `{cmd}`: {source}")]
    Spawn {
        cmd: String,
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// The slide-generation tool. Construct with [`UpdateBuildContent::new`] and
/// register it on a rig agent with `.tool(...)`.
pub struct UpdateBuildContent {
    /// Root of the `mustabsir-ppt-gen` project (where `uv` is run).
    project_dir: PathBuf,
    /// Path to `generate.py` whose `build_content` body is rewritten.
    generate_py: PathBuf,
    /// Output `.pptx` filename, relative to `project_dir`.
    output: String,
    /// Program used to run python / the build script (defaults to `uv`).
    runner: String,
}

impl UpdateBuildContent {
    /// Build the tool from the `mustabsir-ppt-gen` project directory, using the
    /// default `generate.py` location, `mustabsir-output.pptx`, and the `uv` runner.
    pub fn new(project_dir: impl Into<PathBuf>) -> Self {
        let project_dir = project_dir.into();
        let generate_py = project_dir.join("src/ba7ith_ppt_gen/generate.py");
        Self {
            project_dir,
            generate_py,
            output: "mustabsir-output.pptx".to_string(),
            runner: "uv".to_string(),
        }
    }

    /// Override the output `.pptx` filename (relative to the project dir).
    #[allow(dead_code)]
    pub fn output(mut self, output: impl Into<String>) -> Self {
        self.output = output.into();
        self
    }

    /// Override the runner program (defaults to `uv`, i.e. `uv run …`).
    #[allow(dead_code)]
    pub fn runner(mut self, runner: impl Into<String>) -> Self {
        self.runner = runner.into();
        self
    }

    /// Reject code containing forbidden tokens before touching the filesystem.
    fn guard(code: &str) -> Result<(), SlideToolError> {
        for line in code.lines() {
            let stripped = line.trim_start();
            if stripped.starts_with("import ") || stripped.starts_with("from ") {
                return Err(SlideToolError::Forbidden("import".to_string()));
            }
        }
        for needle in DENY_SUBSTRINGS {
            if code.contains(needle) {
                return Err(SlideToolError::Forbidden((*needle).to_string()));
            }
        }
        Ok(())
    }

    /// Indent every non-empty line of `code` by 4 spaces and splice it between
    /// the markers in `original`, returning the new file contents.
    fn splice(original: &str, code: &str) -> Result<String, SlideToolError> {
        let indented = code
            .lines()
            .map(|l| {
                if l.trim().is_empty() {
                    String::new()
                } else {
                    format!("    {l}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        let mut out = String::with_capacity(original.len() + indented.len());
        // 0 = before BEGIN, 1 = inside the editable region, 2 = after END.
        let mut state = 0u8;
        let (mut saw_begin, mut saw_end) = (false, false);

        for line in original.lines() {
            match state {
                0 => {
                    out.push_str(line);
                    out.push('\n');
                    if line.contains(BEGIN_MARKER) && line.contains(MARKER_TAG) {
                        out.push_str(&indented);
                        out.push('\n');
                        saw_begin = true;
                        state = 1;
                    }
                }
                1 => {
                    // Drop the old body; re-emit the END marker line itself.
                    if line.contains(END_MARKER) && line.contains(MARKER_TAG) {
                        out.push_str(line);
                        out.push('\n');
                        saw_end = true;
                        state = 2;
                    }
                }
                _ => {
                    out.push_str(line);
                    out.push('\n');
                }
            }
        }

        if saw_begin && saw_end {
            Ok(out)
        } else {
            Err(SlideToolError::MarkersNotFound(String::new()))
        }
    }

    /// Run `<runner> <args…>` inside the project directory.
    fn run(&self, args: &[&str]) -> Result<Output, SlideToolError> {
        Command::new(&self.runner)
            .args(args)
            .current_dir(&self.project_dir)
            .output()
            .map_err(|source| SlideToolError::Spawn {
                cmd: format!("{} {}", self.runner, args.join(" ")),
                source,
            })
    }
}

impl Tool for UpdateBuildContent {
    const NAME: &'static str = "update_build_content";

    type Error = SlideToolError;
    type Args = UpdateBuildContentArgs;
    type Output = SlideBuildResult;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Replace the body of build_content(deck) with Python statements that \
                build an Arabic-first slide deck using only the allowed ArabicDeck / Slide / \
                TextBlock helpers, then validate and build the .pptx."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "python_code": {
                        "type": "string",
                        "description": "The Python statements for the body of build_content(deck), \
                            at column 0. No function def, no imports, no deck.save()."
                    }
                },
                "required": ["python_code"],
                "additionalProperties": false
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Self::guard(&args.python_code)?;

        let original = std::fs::read_to_string(&self.generate_py)?;
        let spliced = Self::splice(&original, &args.python_code)
            .map_err(|e| match e {
                SlideToolError::MarkersNotFound(_) => {
                    SlideToolError::MarkersNotFound(self.generate_py.display().to_string())
                }
                other => other,
            })?;
        std::fs::write(&self.generate_py, &spliced)?;

        // Helper to put the template back the way we found it on failure.
        let restore = || -> Result<(), SlideToolError> {
            std::fs::write(&self.generate_py, &original)?;
            Ok(())
        };

        // 1. Syntax-check the spliced module for clean error feedback.
        let gen_path = self.generate_py.to_string_lossy().to_string();
        let check = self.run(&["run", "python", "-c", AST_CHECK, &gen_path])?;
        if !check.status.success() {
            restore()?;
            return Err(SlideToolError::Syntax(stderr_or_stdout(&check)));
        }

        // 2. Build the deck.
        let build = self.run(&["run", "ba7ith-ppt-gen", "-o", &self.output])?;
        if !build.status.success() {
            restore()?;
            return Err(SlideToolError::Build(stderr_or_stdout(&build)));
        }

        // The build may have written to a fallback name (e.g. "… (1).pptx") if
        // the requested file was locked (open in PowerPoint). Trust the path the
        // build reported over the requested one so we point at the real file.
        let stdout = String::from_utf8_lossy(&build.stdout);
        let output_path = parse_written_path(&stdout)
            .map(|rel| abs_output(&self.project_dir, &rel))
            .unwrap_or_else(|| abs_output(&self.project_dir, &self.output));

        Ok(SlideBuildResult {
            status: "ok".to_string(),
            output_path,
            build_stdout: stdout.trim().to_string(),
        })
    }
}

/// Prefer stderr (where Python tracebacks land); fall back to stdout.
fn stderr_or_stdout(out: &Output) -> String {
    let err = String::from_utf8_lossy(&out.stderr);
    let trimmed = err.trim();
    if trimmed.is_empty() {
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    } else {
        trimmed.to_string()
    }
}

/// Extract the written path from the build's `Wrote <path> (Arabic fonts
/// embedded)` line, so a locked-file fallback (`… (1).pptx`) is reported
/// correctly. Returns the path relative to the project dir, as printed.
fn parse_written_path(stdout: &str) -> Option<String> {
    for line in stdout.lines() {
        if let Some(rest) = line.trim().strip_prefix("Wrote ") {
            let path = rest
                .strip_suffix(" (Arabic fonts embedded)")
                .unwrap_or(rest)
                .trim();
            if !path.is_empty() {
                return Some(path.to_string());
            }
        }
    }
    None
}

/// Absolute path of the output file, best-effort.
fn abs_output(project_dir: &Path, output: &str) -> String {
    let joined = project_dir.join(output);
    std::fs::canonicalize(&joined)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| joined.display().to_string())
}
