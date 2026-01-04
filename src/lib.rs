//! Async file watcher with glob-based include/exclude patterns.
//!
//! This crate provides an efficient recursive file watcher using Linux's inotify,
//! with built-in support for glob patterns to filter events. Unlike most file
//! watchers that require you to filter events after receiving them, this watcher
//! only sets up watches on directories that could potentially match your patterns,
//! reducing resource usage on large directory trees.
//!
//! # Features
//!
//! - **Selective directory watching**: Only watches directories that could match your include patterns
//! - **Glob patterns**: Supports `*`, `**`, `?`, and character classes like `[a-z]`
//! - **Include/exclude filtering**: Gitignore-style pattern matching with exclude taking precedence
//! - **Pattern files**: Load patterns from `.gitignore`-style files
//! - **Event filtering**: Watch only creates, deletes, updates, or any combination
//! - **Type filtering**: Match only files, only directories, or both
//! - **Debouncing**: Built-in debounce support to batch rapid changes
//! - **Async/await**: Native tokio integration
//!
//! # Platform Support
//!
//! **Linux only** (uses inotify directly). PRs welcome for other platforms.
//!
//! # Quick Start
//!
//! ```no_run
//! use include_exclude_watcher::{Watcher, WatchEvent};
//!
//! #[tokio::main]
//! async fn main() -> std::io::Result<()> {
//!     Watcher::new()
//!         .set_base_dir("./src")
//!         .add_include("**/*.rs")
//!         .add_exclude("**/target/**")
//!         .run(|event, path| {
//!             println!("{:?}: {}", event, path.display());
//!         })
//!         .await
//! }
//! ```
//!
//! # Debounced Watching
//!
//! When files change rapidly (e.g., during a build), you often want to wait
//! for changes to settle before taking action:
//!
//! ```no_run
//! use include_exclude_watcher::Watcher;
//!
//! #[tokio::main]
//! async fn main() -> std::io::Result<()> {
//!     Watcher::new()
//!         .set_base_dir("./src")
//!         .add_include("**/*.rs")
//!         .run_debounced(500, |first_changed_path| {
//!             println!("Files changed! First: {}", first_changed_path.display());
//!         })
//!         .await
//! }
//! ```
//!
//! # Pattern Syntax
//!
//! Patterns use glob syntax similar to `.gitignore`:
//!
//! | Pattern | Matches |
//! |---------|---------|
//! | `*` | Any characters except `/` |
//! | `**` | Any characters including `/` (matches across directories) |
//! | `?` | Any single character except `/` |
//! | `[abc]` | Any character in the set |
//! | `[a-z]` | Any character in the range |
//!
//! ## Pattern Behavior
//!
//! - Patterns **without** `/` match anywhere in the tree (like gitignore).
//!   For example, `*.rs` matches `foo.rs` and `src/bar.rs`.
//! - Patterns **with** `/` are anchored to the base directory.
//!   For example, `src/*.rs` matches `src/main.rs` but not `src/sub/lib.rs`.
//!
//! ## Examples
//!
//! | Pattern | Description |
//! |---------|-------------|
//! | `*.rs` | All Rust files anywhere |
//! | `src/*.rs` | Rust files directly in `src/` |
//! | `**/test_*.rs` | Test files anywhere |
//! | `target/**` | Everything under `target/` |
//! | `*.{rs,toml}` | Rust and TOML files (character class) |
//!
//! # Loading Patterns from Files
//!
//! You can load exclude patterns from gitignore-style files:
//!
//! ```no_run
//! use include_exclude_watcher::Watcher;
//!
//! #[tokio::main]
//! async fn main() -> std::io::Result<()> {
//!     Watcher::new()
//!         .set_base_dir("/project")
//!         .add_include("**/*")
//!         .add_ignore_file(".gitignore")
//!         .add_ignore_file(".watchignore")
//!         .run(|event, path| {
//!             println!("{:?}: {}", event, path.display());
//!         })
//!         .await
//! }
//! ```
//!
//! Pattern file format:
//! - Lines starting with `#` are comments
//! - Empty lines are ignored
//! - All other lines are exclude patterns
//! - **Note**: `!` negation patterns are not supported (excludes always take precedence)
//!
//! # Filtering Events
//!
//! You can filter which events to receive and what types to match:
//!
//! ```no_run
//! use include_exclude_watcher::Watcher;
//!
//! # async fn example() -> std::io::Result<()> {
//! Watcher::new()
//!     .add_include("**/*.rs")
//!     .watch_create(true)   // Receive create events
//!     .watch_delete(true)   // Receive delete events
//!     .watch_update(false)  // Ignore modifications
//!     .match_files(true)    // Match regular files
//!     .match_dirs(false)    // Ignore directories
//!     .run(|event, path| {
//!         // Only file creates and deletes
//!     })
//!     .await
//! # }
//! ```

use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fs;
use std::io::{BufRead, BufReader, Result};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::unix::AsyncFd;

// --- Pattern Parsing ---

/// Simple glob pattern matcher for a single path component.
/// Supports: * (any chars), ? (single char), [abc] (char class), [a-z] (range)
#[derive(Debug, Clone)]
struct GlobPattern {
    pattern: String,
}

impl PartialEq for GlobPattern {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern
    }
}

impl GlobPattern {
    fn new(pattern: &str) -> Self {
        Self {
            pattern: pattern.to_string(),
        }
    }

    fn matches(&self, text: &str) -> bool {
        Self::match_recursive(self.pattern.as_bytes(), text.as_bytes())
    }

    fn match_recursive(pattern: &[u8], text: &[u8]) -> bool {
        let mut p = 0;
        let mut t = 0;

        // For backtracking on '*'
        let mut star_p = None;
        let mut star_t = None;

        while t < text.len() {
            if p < pattern.len() {
                match pattern[p] {
                    b'*' => {
                        // '*' matches zero or more characters
                        star_p = Some(p);
                        star_t = Some(t);
                        p += 1;
                        continue;
                    }
                    b'?' => {
                        // '?' matches exactly one character
                        p += 1;
                        t += 1;
                        continue;
                    }
                    b'[' => {
                        // Character class
                        if let Some((matched, end_pos)) =
                            Self::match_char_class(&pattern[p..], text[t])
                        {
                            if matched {
                                p += end_pos;
                                t += 1;
                                continue;
                            }
                        }
                        // Fall through to backtrack
                    }
                    c => {
                        // Literal character match
                        if c == text[t] {
                            p += 1;
                            t += 1;
                            continue;
                        }
                        // Fall through to backtrack
                    }
                }
            }

            // No match at current position, try backtracking
            if let (Some(sp), Some(st)) = (star_p, star_t) {
                // Backtrack: make '*' match one more character
                p = sp + 1;
                star_t = Some(st + 1);
                t = st + 1;
            } else {
                return false;
            }
        }

        // Consume any trailing '*' in pattern
        while p < pattern.len() && pattern[p] == b'*' {
            p += 1;
        }

        p == pattern.len()
    }

    /// Match a character class like [abc] or [a-z] or [!abc]
    /// Returns (matched, bytes_consumed) if valid class, None if invalid
    fn match_char_class(pattern: &[u8], ch: u8) -> Option<(bool, usize)> {
        if pattern.is_empty() || pattern[0] != b'[' {
            return None;
        }

        let mut i = 1;
        let mut matched = false;
        let negated = i < pattern.len() && (pattern[i] == b'!' || pattern[i] == b'^');
        if negated {
            i += 1;
        }

        while i < pattern.len() {
            if pattern[i] == b']' && i > 1 + (negated as usize) {
                // End of character class
                return Some((matched != negated, i + 1));
            }

            // Check for range: a-z
            if i + 2 < pattern.len() && pattern[i + 1] == b'-' && pattern[i + 2] != b']' {
                let start = pattern[i];
                let end = pattern[i + 2];
                if ch >= start && ch <= end {
                    matched = true;
                }
                i += 3;
            } else {
                // Single character
                if pattern[i] == ch {
                    matched = true;
                }
                i += 1;
            }
        }

        // No closing bracket found
        None
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Segment {
    Exact(String),
    Wildcard(GlobPattern),
    DoubleWildcard, // **
}

#[derive(Debug, Clone)]
struct Pattern {
    segments: Vec<Segment>,
}

impl Pattern {
    fn parse(pattern: &str) -> Self {
        let mut segments = Vec::new();

        // Patterns without / match anywhere in the tree (like gitignore)
        let effective_pattern = if !pattern.contains('/') {
            format!("**/{}", pattern)
        } else {
            pattern.trim_start_matches('/').to_string()
        };

        let normalized = effective_pattern.replace("//", "/");

        for part in normalized.split('/') {
            if part.is_empty() || part == "." {
                continue;
            }

            if part == "**" {
                segments.push(Segment::DoubleWildcard);
            } else if part.contains('*') || part.contains('?') || part.contains('[') {
                segments.push(Segment::Wildcard(GlobPattern::new(part)));
            } else {
                segments.push(Segment::Exact(part.to_string()));
            }
        }

        Pattern { segments }
    }

    fn check(&self, path_segments: &[String], allow_prefix: bool) -> bool {
        let pattern_segments = &self.segments;
        let mut path_index = 0;

        for pattern_index in 0..pattern_segments.len() {
            let pattern_segment = &pattern_segments[pattern_index];

            if path_index >= path_segments.len() {
                // We ran out of path elements
                if pattern_segment == &Segment::DoubleWildcard && pattern_index == pattern_segments.len() - 1
                {
                    // The only pattern segment we still need to match is **. We'll consider that a match for the parent.
                    return true;
                }
                // Something within this path could potentially match.
                return allow_prefix;
            }

            match &pattern_segment {
                Segment::Exact(s) => {
                    if s != &path_segments[path_index] {
                        return false;
                    }
                    path_index += 1;
                }
                Segment::Wildcard(p) => {
                    if !p.matches(&path_segments[path_index]) {
                        return false;
                    }
                    path_index += 1;
                }
                Segment::DoubleWildcard => {
                    if allow_prefix {
                        // If we're matching a **, there can always be some deeply nested dir structure that
                        // will match the rest of our pattern. So for prefix matching, the answer is always true.
                        return true;
                    }

                    let patterns_left = pattern_segments.len() - (pattern_index + 1);
                    let next_path_index = path_segments.len() - patterns_left;
                    if next_path_index < path_index {
                        return false;
                    }
                    path_index = next_path_index;
                }
            }
        }

        // If there are spurious path elements, this is not a match.
        if path_index < path_segments.len() {
            return false;
        }

        // We have an exact match. However when in allow_prefix mode, that means this directory is the target
        // and its contents does not need to be watched.
        return !allow_prefix;
    }
}

// --- Inotify Wrapper ---

struct Inotify {
    fd: AsyncFd<i32>,
}

impl Inotify {
    fn new() -> Result<Self> {
        let fd = unsafe { libc::inotify_init1(libc::IN_NONBLOCK | libc::IN_CLOEXEC) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self {
            fd: AsyncFd::new(fd)?,
        })
    }

    fn add_watch(&self, path: &Path, mask: u32) -> Result<i32> {
        let c_path = CString::new(path.as_os_str().as_bytes())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        let wd = unsafe { libc::inotify_add_watch(self.fd.as_raw_fd(), c_path.as_ptr(), mask) };
        if wd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(wd)
    }

    async fn read_events(&self, buffer: &mut [u8]) -> Result<usize> {
        loop {
            let mut guard = self.fd.readable().await?;
            match guard.try_io(|inner| {
                let res = unsafe {
                    libc::read(
                        inner.as_raw_fd(),
                        buffer.as_mut_ptr() as *mut _,
                        buffer.len(),
                    )
                };
                if res < 0 {
                    Err(std::io::Error::last_os_error())
                } else {
                    Ok(res as usize)
                }
            }) {
                Ok(Ok(len)) => return Ok(len),
                Ok(Err(e)) => {
                    if e.kind() == std::io::ErrorKind::WouldBlock {
                        continue;
                    }
                    return Err(e);
                }
                Err(_) => continue,
            }
        }
    }
}

impl Drop for Inotify {
    fn drop(&mut self) {
        unsafe { libc::close(self.fd.as_raw_fd()) };
    }
}

// --- Helper Functions ---

fn path_to_segments(path: &Path) -> Vec<String> {
    let path_str = path.to_string_lossy();
    let path_str = path_str.replace("//", "/");
    path_str
        .split('/')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

const INOTIFY_MASK: u32 = libc::IN_MODIFY
    | libc::IN_CLOSE_WRITE
    | libc::IN_CREATE
    | libc::IN_DELETE
    | libc::IN_MOVED_FROM
    | libc::IN_MOVED_TO
    | libc::IN_DONT_FOLLOW;


fn parse_inotify_events(buffer: &[u8], len: usize) -> Vec<(i32, u32, String)> {
    let mut events = Vec::new();
    let mut ptr = buffer.as_ptr();
    let end = unsafe { ptr.add(len) };

    while ptr < end {
        let event = unsafe { &*(ptr as *const libc::inotify_event) };
        let name_len = event.len as usize;

        if name_len > 0 {
            let name_ptr = unsafe { ptr.add(std::mem::size_of::<libc::inotify_event>()) };
            let name_slice =
                unsafe { std::slice::from_raw_parts(name_ptr as *const u8, name_len) };
            let name_str = String::from_utf8_lossy(name_slice)
                .trim_matches(char::from(0))
                .to_string();
            events.push((event.wd, event.mask, name_str));
        }

        ptr = unsafe { ptr.add(std::mem::size_of::<libc::inotify_event>() + name_len) };
    }

    events
}

/// Type of file system event.
///
/// These events correspond to inotify events, but are simplified into three
/// categories that cover most use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchEvent {
    /// File or directory was created.
    ///
    /// Also triggered when a file/directory is moved *into* a watched directory.
    Create,
    /// File or directory was deleted.
    ///
    /// Also triggered when a file/directory is moved *out of* a watched directory.
    Delete,
    /// File content was modified.
    ///
    /// Triggered on `IN_MODIFY` (content changed) or `IN_CLOSE_WRITE` (file
    /// opened for writing was closed). Directory content changes (files added/removed)
    /// are reported as [`Create`](WatchEvent::Create)/[`Delete`](WatchEvent::Delete) instead.
    Update,
    /// Initial event for preexisting files/directories.
    ///
    /// Only emitted when [`Watcher::watch_initial`] is enabled. Fired once
    /// for each file or directory that matches the patterns at the time the
    /// watcher starts, before any file system events are processed.
    Initial,
    /// Debug event: a watch was added on this directory.
    ///
    /// Only emitted when [`Watcher::debug_watches`] is enabled. Useful for
    /// understanding which directories are being watched based on your patterns.
    DebugWatch,
}

/// Builder for configuring and running a file watcher.
///
/// Use method chaining to configure the watcher, then call [`run`](Watcher::run)
/// or [`run_debounced`](Watcher::run_debounced) to start watching.
///
/// # Example
///
/// ```no_run
/// use include_exclude_watcher::Watcher;
///
/// # async fn example() -> std::io::Result<()> {
/// Watcher::new()
///     .set_base_dir("/project")
///     .add_include("src/**/*.rs")
///     .add_include("Cargo.toml")
///     .add_exclude("**/target/**")
///     .run(|event, path| {
///         println!("{:?}: {}", event, path.display());
///     })
///     .await
/// # }
/// ```
pub struct Watcher {
    includes: Vec<String>,
    excludes: Vec<String>,
    base_dir: PathBuf,
    watch_create: bool,
    watch_delete: bool,
    watch_update: bool,
    watch_initial: bool,
    match_files: bool,
    match_dirs: bool,
    return_absolute: bool,
    debug_watches_enabled: bool,
}

/// Backwards compatibility alias for [`Watcher`].
#[deprecated(since = "0.1.2", note = "Renamed to Watcher")]
pub type WatchBuilder = Watcher;

impl Default for Watcher {
    fn default() -> Self {
        Self::new()
    }
}

// Runtime state for the watcher
struct WatcherState<F> {
    root: PathBuf,
    inotify: Inotify,
    watches: HashMap<i32, PathBuf>,
    paths: HashSet<PathBuf>,
    include_patterns: Vec<Pattern>,
    exclude_patterns: Vec<Pattern>,
    callback: F,
}

impl Watcher {
    /// Create a new file watcher with default settings.
    ///
    /// Defaults:
    /// - Base directory: current working directory
    /// - Includes: none (must be added, or watches everything)
    /// - Excludes: none
    /// - Event types: create, delete, update all enabled; initial disabled
    /// - Match types: both files and directories
    /// - Path format: relative paths
    pub fn new() -> Self {
        Watcher {
            includes: Vec::new(),
            excludes: Vec::new(),
            base_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")),
            watch_create: true,
            watch_delete: true,
            watch_update: true,
            watch_initial: false,
            match_files: true,
            match_dirs: true,
            return_absolute: false,
            debug_watches_enabled: false,
        }
    }

    /// Enable debug watch events.
    ///
    /// When enabled, [`WatchEvent::DebugWatch`] events will be emitted for each
    /// directory that is watched. Useful for debugging pattern matching.
    pub fn debug_watches(mut self, enabled: bool) -> Self {
        self.debug_watches_enabled = enabled;
        self
    }

    /// Add a single include pattern.
    ///
    /// Patterns use glob syntax:
    /// - `*` matches any sequence of characters except `/`
    /// - `**` matches any sequence of characters including `/`
    /// - `?` matches any single character except `/`
    /// - `[abc]` matches any character in the set
    ///
    /// Patterns without a `/` match anywhere in the tree (like gitignore).
    /// For example, `*.rs` is equivalent to `**/*.rs`.
    pub fn add_include(mut self, pattern: impl Into<String>) -> Self {
        self.includes.push(pattern.into());
        self
    }

    /// Add multiple include patterns.
    pub fn add_includes(mut self, patterns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.includes.extend(patterns.into_iter().map(|p| p.into()));
        self
    }

    /// Add a single exclude pattern.
    ///
    /// Excludes take precedence over includes. Uses the same glob syntax as includes.
    pub fn add_exclude(mut self, pattern: impl Into<String>) -> Self {
        self.excludes.push(pattern.into());
        self
    }

    /// Add multiple exclude patterns.
    pub fn add_excludes(mut self, patterns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.excludes.extend(patterns.into_iter().map(|p| p.into()));
        self
    }

    /// Add patterns from a gitignore-style file.
    ///
    /// Lines starting with `#` are comments. All other non-empty lines are
    /// exclude patterns. Note: `!` negation patterns are not supported (a
    /// warning will be printed) because excludes always take precedence over
    /// includes in this library.
    ///
    /// If the file doesn't exist, this method does nothing (no error).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use include_exclude_watcher::Watcher;
    ///
    /// # async fn example() -> std::io::Result<()> {
    /// Watcher::new()
    ///     .set_base_dir("/project")
    ///     .add_include("*")
    ///     .add_ignore_file(".gitignore")
    ///     .add_ignore_file(".watchignore")
    ///     .run(|event, path| {
    ///         println!("{:?}: {}", event, path.display());
    ///     })
    ///     .await
    /// # }
    /// ```
    pub fn add_ignore_file(mut self, path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();

        // Resolve relative to base_dir
        let full_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.base_dir.join(path)
        };

        if let Ok(file) = fs::File::open(&full_path) {
            let reader = BufReader::new(file);
            let mut has_negation = false;
            for line in reader.lines().map_while(Result::ok) {
                let trimmed = line.trim();

                // Skip empty lines and comments
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    continue;
                }

                // Lines starting with ! are negations - not supported
                if trimmed.starts_with('!') {
                    has_negation = true;
                } else {
                    // Regular lines are exclude patterns
                    self.excludes.push(trimmed.to_string());
                }
            }
            if has_negation {
                println!("Warning: negation patterns (!) in {} are ignored; excludes always take precedence over includes in this library", full_path.display());
            }
        }

        self
    }

    /// Set the base directory for watching.
    ///
    /// All patterns are relative to this directory. Defaults to the current
    /// working directory.
    pub fn set_base_dir(mut self, base_dir: impl Into<PathBuf>) -> Self {
        self.base_dir = base_dir.into();
        self
    }

    /// Set whether to watch for file/directory creation events.
    ///
    /// Default: `true`
    pub fn watch_create(mut self, enabled: bool) -> Self {
        self.watch_create = enabled;
        self
    }

    /// Set whether to watch for file/directory deletion events.
    ///
    /// Default: `true`
    pub fn watch_delete(mut self, enabled: bool) -> Self {
        self.watch_delete = enabled;
        self
    }

    /// Set whether to watch for file modification events.
    ///
    /// Default: `true`
    pub fn watch_update(mut self, enabled: bool) -> Self {
        self.watch_update = enabled;
        self
    }

    /// Set whether to emit initial events for preexisting files/directories.
    ///
    /// When enabled, [`WatchEvent::Initial`] events will be emitted for all
    /// files and directories that match the patterns at startup, before any
    /// file system events are processed. This is useful for building an initial
    /// inventory of matching files.
    ///
    /// Default: `false`
    ///
    /// # Example
    ///
    /// ```no_run
    /// use include_exclude_watcher::Watcher;
    ///
    /// # async fn example() -> std::io::Result<()> {
    /// Watcher::new()
    ///     .add_include("**/*.rs")
    ///     .watch_initial(true)
    ///     .run(|event, path| {
    ///         // First receives Initial events for all existing .rs files,
    ///         // then receives Create/Update/Delete events for changes
    ///     })
    ///     .await
    /// # }
    /// ```
    pub fn watch_initial(mut self, enabled: bool) -> Self {
        self.watch_initial = enabled;
        self
    }

    /// Set whether to match regular files.
    ///
    /// Default: `true`
    pub fn match_files(mut self, enabled: bool) -> Self {
        self.match_files = enabled;
        self
    }

    /// Set whether to match directories.
    ///
    /// Default: `true`
    pub fn match_dirs(mut self, enabled: bool) -> Self {
        self.match_dirs = enabled;
        self
    }

    /// Set whether to return absolute paths.
    ///
    /// When `false` (default), paths passed to the callback are relative to
    /// the base directory. When `true`, paths are absolute.
    pub fn return_absolute(mut self, enabled: bool) -> Self {
        self.return_absolute = enabled;
        self
    }

    /// Run the watcher with the provided callback.
    ///
    /// This method runs forever, calling the callback for each matching event.
    /// The callback receives the event type and the path (relative or absolute
    /// depending on configuration).
    ///
    /// If no include patterns are specified, watches everything.
    pub async fn run<F>(self, callback: F) -> Result<()>
    where
        F: FnMut(WatchEvent, PathBuf),
    {
        self.run_internal(callback, None).await
    }

    /// Run the watcher with debouncing.
    ///
    /// Waits for file changes, then waits until no changes have occurred for
    /// at least `ms` milliseconds before calling the callback. This is useful
    /// for batching rapid changes (like when a build tool writes many files).
    ///
    /// The callback receives the path of the first file that changed.
    pub async fn run_debounced<F>(self, ms: u64, mut callback: F) -> Result<()>
    where
        F: FnMut(PathBuf),
    {
        self.run_internal(|_, path| callback(path), Some(Duration::from_millis(ms))).await
    }

    fn should_watch<F>(&self, state: &WatcherState<F>, relative_path: &Path, is_dir: bool) -> bool {
        let segments = path_to_segments(relative_path);
        
        if state.exclude_patterns.iter().any(|p| p.check(&segments, false)) {
            return false;
        }

        state.include_patterns.iter().any(|p| p.check(&segments, is_dir))
    }

    fn check_event<F>(&self, state: &WatcherState<F>, rel_path: &Path, is_dir: bool) -> bool {
        if if is_dir { !self.match_dirs } else { !self.match_files } {
            return false;
        }
        self.should_watch(state, rel_path, false)
    }

    fn emit_event<F>(
        &self,
        state: &mut WatcherState<F>,
        event: WatchEvent,
        rel_path: &Path,
    ) where
        F: FnMut(WatchEvent, PathBuf),
    {
        let path = if self.return_absolute {
            if rel_path.as_os_str().is_empty() {
                state.root.clone()
            } else {
                state.root.join(rel_path)
            }
        } else {
            rel_path.to_path_buf()
        };
        (state.callback)(event, path);
    }

    fn add_watch_recursive<F>(
        &self,
        state: &mut WatcherState<F>,
        initial_path: PathBuf,
        emit_initial: bool,
    ) where
        F: FnMut(WatchEvent, PathBuf),
    {
        if state.paths.contains(&initial_path) {
            return;
        }

        let mut stack = vec![initial_path];
        while let Some(rel_path) = stack.pop() {
            if !self.should_watch(state, &rel_path, true) {
                continue;
            }

            let full_path = if rel_path.as_os_str().is_empty() {
                state.root.clone()
            } else {
                state.root.join(&rel_path)
            };

            if !full_path.is_dir() {
                continue;
            }

            let wd = match state.inotify.add_watch(&full_path, INOTIFY_MASK) {
                Ok(wd) => wd,
                Err(e) => {
                    eprintln!("Failed to add watch for {:?}: {}", full_path, e);
                    continue;
                }
            };

            state.paths.insert(rel_path.clone());
            state.watches.insert(wd, rel_path.clone());

            if self.debug_watches_enabled {
                (state.callback)(WatchEvent::DebugWatch, rel_path.clone());
            }

            if let Ok(entries) = std::fs::read_dir(&full_path) {
                for entry in entries.flatten() {
                    if let Ok(ft) = entry.file_type() {
                        let child_rel_path = rel_path.join(entry.file_name());
                        let is_dir = ft.is_dir();

                        if emit_initial && self.check_event(state, &child_rel_path, is_dir) {
                            self.emit_event(state, WatchEvent::Initial, &child_rel_path);
                        }

                        if is_dir && !state.paths.contains(&child_rel_path) {
                            stack.push(child_rel_path);
                        }
                    }
                }
            }
        }
    }

    async fn run_internal<F>(self, callback: F, debounce: Option<Duration>) -> Result<()>
    where
        F: FnMut(WatchEvent, PathBuf),
    {
        // If no includes are specified, watch everything; if empty, sleep forever
        let includes = if self.includes.is_empty() {
            vec!["**".to_string()]
        } else {
            self.includes.clone()
        };

        // If no includes are specified, just sleep forever
        if includes.is_empty() {
            loop {
                tokio::time::sleep(Duration::from_secs(3600)).await;
            }
        }

        let root = if self.base_dir.is_absolute() {
            self.base_dir.clone()
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("/"))
                .join(&self.base_dir)
        };

        let mut state = WatcherState {
            root,
            inotify: Inotify::new()?,
            watches: HashMap::new(),
            paths: HashSet::new(),
            include_patterns: includes.iter().map(|p| Pattern::parse(p)).collect(),
            exclude_patterns: self.excludes.iter().map(|p| Pattern::parse(p)).collect(),
            callback,
        };

        // Initial scan for watches (and optionally emit Initial events)
        let emit_initial = self.watch_initial && debounce.is_none();
        self.add_watch_recursive(&mut state, PathBuf::new(), emit_initial);

        // Debouncing state
        let mut debounce_deadline: Option<tokio::time::Instant> = None;
        let mut debounce_first_path: Option<PathBuf> = None;

        // Event loop
        let mut buffer = [0u8; 8192];
        loop {
            // Calculate timeout for debouncing
            let read_future = state.inotify.read_events(&mut buffer);
            
            let read_result = if let Some(deadline) = debounce_deadline {
                let now = tokio::time::Instant::now();
                if deadline <= now {
                    // Timer expired, fire callback and reset
                    debounce_deadline = None;
                    (state.callback)(WatchEvent::Update, debounce_first_path.take().unwrap_or_default());
                    continue;
                }
                // Wait with timeout
                match tokio::time::timeout(deadline - now, read_future).await {
                    Ok(result) => Some(result),
                    Err(_) => {
                        // Timeout expired, fire callback
                        debounce_deadline = None;
                        (state.callback)(WatchEvent::Update, debounce_first_path.take().unwrap_or_default());
                        continue;
                    }
                }
            } else {
                Some(read_future.await)
            };

            let Some(result) = read_result else { continue };
            
            match result {
                Ok(len) => {
                    let events = parse_inotify_events(&buffer, len);
                    let mut first_matching_path: Option<PathBuf> = None;

                    for (wd, mask, name_str) in events {
                        if (mask & libc::IN_IGNORED as u32) != 0 {
                            if let Some(path) = state.watches.remove(&wd) {
                                state.paths.remove(&path);
                            }
                            continue;
                        }

                        let rel_path = if let Some(dir_path) = state.watches.get(&wd) {
                            dir_path.join(&name_str)
                        } else {
                            println!("Warning: received event for unknown watch descriptor {}", wd);
                            continue;
                        };

                        let is_dir = mask & libc::IN_ISDIR as u32 != 0;
                        let is_create = (mask & libc::IN_CREATE as u32) != 0
                            || (mask & libc::IN_MOVED_TO as u32) != 0;
                        let is_delete = (mask & libc::IN_DELETE as u32) != 0
                            || (mask & libc::IN_MOVED_FROM as u32) != 0;
                        let is_update = (mask & libc::IN_MODIFY as u32) != 0
                            || (mask & libc::IN_CLOSE_WRITE as u32) != 0;

                        if is_dir && is_create {
                            // New directory created 
                            self.add_watch_recursive(&mut state, rel_path.clone(), false);
                        }

                        let event_type = if is_create && self.watch_create {
                            WatchEvent::Create
                        } else if is_delete && self.watch_delete {
                            WatchEvent::Delete
                        } else if is_update && self.watch_update {
                            WatchEvent::Update
                        } else {
                            continue
                        };

                        if !self.check_event(&state, &rel_path, is_dir) {
                            continue;
                        }

                        if first_matching_path.is_none() {
                            first_matching_path = Some(rel_path.clone());
                        }

                        // Emit event if not in debounce mode
                        if debounce.is_none() {
                            self.emit_event(&mut state, event_type, &rel_path);
                        }
                    }
                    
                    // If debouncing and we had events, reset the timer
                    if let Some(d) = debounce {
                        if let Some(path) = first_matching_path {
                            if debounce_first_path.is_none() {
                                debounce_first_path = Some(path);
                            }
                            debounce_deadline = Some(tokio::time::Instant::now() + d);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error reading inotify events: {}", e);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};
    use tokio::task::JoinHandle;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    enum EventType {
        Create,
        Delete,
        Update,
        Initial,
        DebugWatch,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct Event {
        path: PathBuf,
        event_type: EventType,
    }

    type EventTracker = Arc<Mutex<Vec<Event>>>;

    struct TestInstance {
        test_dir: PathBuf,
        tracker: EventTracker,
        watcher_handle: Option<JoinHandle<()>>,
    }

    impl TestInstance {
        async fn new<F>(test_name: &str, configure: F) -> Self
        where
            F: FnOnce(Watcher) -> Watcher + Send + 'static,
        {
            let test_dir = std::env::current_dir()
                .unwrap()
                .join(format!(".file-watcher-test-{}", test_name));

            if test_dir.exists() {
                std::fs::remove_dir_all(&test_dir).unwrap();
            }
            std::fs::create_dir(&test_dir).unwrap();

            let tracker = Arc::new(Mutex::new(Vec::new()));

            let tracker_clone = tracker.clone();
            let test_dir_clone = test_dir.clone();

            let watcher_handle = tokio::spawn(async move {
                let watcher = Watcher::new()
                    .set_base_dir(&test_dir_clone)
                    .debug_watches(true);

                let watcher = configure(watcher);

                let _ = watcher
                    .run(move |event_type, path| {
                        tracker_clone.lock().unwrap().push(Event {
                            path: path.clone(),
                            event_type: match event_type {
                                WatchEvent::Create => EventType::Create,
                                WatchEvent::Delete => EventType::Delete,
                                WatchEvent::Update => EventType::Update,
                                WatchEvent::Initial => EventType::Initial,
                                WatchEvent::DebugWatch => EventType::DebugWatch,
                            },
                        });
                    })
                    .await;
            });

            tokio::time::sleep(Duration::from_millis(100)).await;

            let instance = Self {
                test_dir,
                tracker,
                watcher_handle: Some(watcher_handle),
            };

            instance.assert_events(&[], &[], &[], &[""]).await;

            instance
        }

        fn create_dir(&self, path: &str) {
            std::fs::create_dir_all(self.test_dir.join(path)).unwrap();
        }

        fn write_file(&self, path: &str, content: &str) {
            let full_path = self.test_dir.join(path);
            if let Some(parent) = full_path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(full_path, content).unwrap();
        }

        fn remove_file(&self, path: &str) {
            std::fs::remove_file(self.test_dir.join(path)).unwrap();
        }

        fn rename(&self, from: &str, to: &str) {
            std::fs::rename(self.test_dir.join(from), self.test_dir.join(to)).unwrap();
        }

        async fn assert_events(
            &self,
            creates: &[&str],
            deletes: &[&str],
            updates: &[&str],
            watches: &[&str],
        ) {
            tokio::time::sleep(Duration::from_millis(200)).await;

            let events = self.tracker.lock().unwrap().clone();
            let mut expected = HashSet::new();

            for create in creates {
                expected.insert(Event {
                    path: PathBuf::from(create),
                    event_type: EventType::Create,
                });
            }

            for delete in deletes {
                expected.insert(Event {
                    path: PathBuf::from(delete),
                    event_type: EventType::Delete,
                });
            }

            for update in updates {
                expected.insert(Event {
                    path: PathBuf::from(update),
                    event_type: EventType::Update,
                });
            }

            for watch in watches {
                expected.insert(Event {
                    path: PathBuf::from(watch),
                    event_type: EventType::DebugWatch,
                });
            }

            let actual: HashSet<Event> = events.iter().cloned().collect();

            for event in &actual {
                if !expected.contains(event) {
                    panic!("Unexpected event: {:?}", event);
                }
            }

            for event in &expected {
                if !actual.contains(event) {
                    panic!(
                        "Missing expected event: {:?}\nActual events: {:?}",
                        event, actual
                    );
                }
            }

            self.tracker.lock().unwrap().clear();
        }

        async fn assert_no_events(&self) {
            tokio::time::sleep(Duration::from_millis(500)).await;
            let events = self.tracker.lock().unwrap();
            assert_eq!(
                events.len(),
                0,
                "Expected no events, but got: {:?}",
                events
            );
        }
    }

    impl Drop for TestInstance {
        fn drop(&mut self) {
            if let Some(handle) = self.watcher_handle.take() {
                handle.abort();
            }
            if self.test_dir.exists() {
                let _ = std::fs::remove_dir_all(&self.test_dir);
            }
        }
    }

    #[tokio::test]
    async fn test_file_create_update_delete() {
        let test = TestInstance::new("create_update_delete", |b| b.add_include("**/*")).await;

        test.write_file("test.txt", "");
        test.assert_events(&["test.txt"], &[], &["test.txt"], &[])
            .await;

        test.write_file("test.txt", "hello");
        test.assert_events(&[], &[], &["test.txt"], &[]).await;

        test.remove_file("test.txt");
        test.assert_events(&[], &["test.txt"], &[], &[]).await;
    }

    #[tokio::test]
    async fn test_directory_operations() {
        let test = TestInstance::new("directory_operations", |b| b.add_include("**/*")).await;

        test.create_dir("subdir");
        test.assert_events(&["subdir"], &[], &[], &["subdir"]).await;

        test.write_file("subdir/file.txt", "");
        test.assert_events(&["subdir/file.txt"], &[], &["subdir/file.txt"], &[])
            .await;
    }

    #[tokio::test]
    async fn test_move_operations() {
        let test = TestInstance::new("move_operations", |b| b.add_include("**/*")).await;

        test.write_file("old.txt", "content");
        test.assert_events(&["old.txt"], &[], &["old.txt"], &[])
            .await;

        test.rename("old.txt", "new.txt");
        test.assert_events(&["new.txt"], &["old.txt"], &[], &[])
            .await;
    }

    #[tokio::test]
    async fn test_event_filtering() {
        let test = TestInstance::new("event_filtering", |b| {
            b.add_include("**/*")
                .watch_create(true)
                .watch_delete(false)
                .watch_update(false)
        })
        .await;

        test.write_file("test.txt", "");
        test.assert_events(&["test.txt"], &[], &[], &[]).await;

        test.write_file("test.txt", "hello");
        test.assert_no_events().await;

        test.remove_file("test.txt");
        test.assert_no_events().await;
    }

    #[tokio::test]
    async fn test_pattern_matching() {
        let test = TestInstance::new("pattern_matching", |b| b.add_include("**/*.txt")).await;

        test.write_file("test.txt", "");
        test.assert_events(&["test.txt"], &[], &["test.txt"], &[])
            .await;

        test.write_file("test.rs", "");
        test.assert_no_events().await;
    }

    #[tokio::test]
    async fn test_matching_stops_at_depth() {
        let test = TestInstance::new("matching_stops_at_depth", |b| b.add_include("*/xyz/*.*")).await;

        test.write_file("test.txt", "");
        test.assert_no_events().await;

        test.create_dir("abc/xyz");
        test.assert_events(&[], &[], &[], &["abc", "abc/xyz"]).await;

        test.create_dir("abc/hjk/a.b");
        test.assert_no_events().await;

        test.create_dir("abc/xyz/a.b");
        test.assert_events(&["abc/xyz/a.b"], &[], &[], &[]).await; // Should not watch the a.b dir

        test.create_dir("abc/xyz/a.b/x.y");
        test.assert_events(&[], &[], &[], &[]).await;
    }

    #[tokio::test]
    async fn test_exclude_prevents_watching() {
        let test = TestInstance::new("exclude_prevents_watch", |b| {
            b.add_include("**/*").add_exclude("node_modules/**")
        })
        .await;

        test.create_dir("node_modules");
        tokio::time::sleep(Duration::from_millis(200)).await;

        test.write_file("node_modules/package.json", "");
        test.assert_no_events().await;

        test.write_file("test.txt", "");
        test.assert_events(&["test.txt"], &[], &["test.txt"], &[])
            .await;
    }

    #[tokio::test]
    async fn test_pattern_file() {
        // Setup: create test directory manually and write pattern file first
        let test_dir = std::env::current_dir()
            .unwrap()
            .join(".file-watcher-test-pattern_file");

        if test_dir.exists() {
            std::fs::remove_dir_all(&test_dir).unwrap();
        }
        std::fs::create_dir(&test_dir).unwrap();

        // Write pattern file before starting watcher
        std::fs::write(
            test_dir.join(".watchignore"),
            "# Comment line\nignored/**\n",
        )
        .unwrap();

        // Now create watcher with pattern file
        let tracker = Arc::new(Mutex::new(Vec::<Event>::new()));
        let tracker_clone = tracker.clone();
        let test_dir_clone = test_dir.clone();

        let watcher_handle = tokio::spawn(async move {
            let _ = Watcher::new()
                .set_base_dir(&test_dir_clone)
                .debug_watches(true)
                .add_include("**/*")
                .add_ignore_file(".watchignore")
                .run(move |event_type, path| {
                    tracker_clone.lock().unwrap().push(Event {
                        path: path.clone(),
                        event_type: match event_type {
                            WatchEvent::Create => EventType::Create,
                            WatchEvent::Delete => EventType::Delete,
                            WatchEvent::Update => EventType::Update,
                            WatchEvent::Initial => EventType::Initial,
                            WatchEvent::DebugWatch => EventType::DebugWatch,
                        },
                    });
                })
                .await;
        });

        tokio::time::sleep(Duration::from_millis(100)).await;
        tracker.lock().unwrap().clear(); // Clear initial watch event

        // Create ignored directory
        std::fs::create_dir(test_dir.join("ignored")).unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Files in ignored/ should not trigger events (because of exclude)
        std::fs::write(test_dir.join("ignored/test.txt"), "").unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Check no events for ignored files
        {
            let events = tracker.lock().unwrap();
            let has_ignored_events = events.iter().any(|e| {
                e.path.to_string_lossy().contains("ignored")
                    && e.event_type != EventType::DebugWatch
            });
            assert!(
                !has_ignored_events,
                "Expected no events for ignored files, but got: {:?}",
                events
            );
        }
        tracker.lock().unwrap().clear();

        // Normal files should still work
        std::fs::write(test_dir.join("normal.txt"), "").unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;

        {
            let events = tracker.lock().unwrap();
            let has_normal = events
                .iter()
                .any(|e| e.path == PathBuf::from("normal.txt"));
            assert!(has_normal, "Expected event for normal.txt, got: {:?}", events);
        }

        // Cleanup
        watcher_handle.abort();
        let _ = std::fs::remove_dir_all(&test_dir);
    }

    #[tokio::test]
    async fn test_watch_initial() {
        // Setup: create test directory and files before starting watcher
        let test_dir = std::env::current_dir()
            .unwrap()
            .join(".file-watcher-test-watch_initial");

        if test_dir.exists() {
            std::fs::remove_dir_all(&test_dir).unwrap();
        }
        std::fs::create_dir(&test_dir).unwrap();

        // Create some files before starting the watcher
        std::fs::write(test_dir.join("existing1.txt"), "content1").unwrap();
        std::fs::write(test_dir.join("existing2.txt"), "content2").unwrap();
        std::fs::create_dir(test_dir.join("subdir")).unwrap();
        std::fs::write(test_dir.join("subdir/nested.txt"), "nested").unwrap();
        std::fs::write(test_dir.join("ignored.rs"), "should be ignored").unwrap();

        let tracker = Arc::new(Mutex::new(Vec::<Event>::new()));
        let tracker_clone = tracker.clone();
        let test_dir_clone = test_dir.clone();

        let watcher_handle = tokio::spawn(async move {
            let _ = Watcher::new()
                .set_base_dir(&test_dir_clone)
                .add_include("**/*.txt")
                .watch_initial(true)
                .run(move |event_type, path| {
                    tracker_clone.lock().unwrap().push(Event {
                        path: path.clone(),
                        event_type: match event_type {
                            WatchEvent::Create => EventType::Create,
                            WatchEvent::Delete => EventType::Delete,
                            WatchEvent::Update => EventType::Update,
                            WatchEvent::Initial => EventType::Initial,
                            WatchEvent::DebugWatch => EventType::DebugWatch,
                        },
                    });
                })
                .await;
        });

        tokio::time::sleep(Duration::from_millis(200)).await;

        // Check that Initial events were emitted for preexisting .txt files
        {
            let events = tracker.lock().unwrap();
            let initial_events: Vec<_> = events
                .iter()
                .filter(|e| e.event_type == EventType::Initial)
                .collect();

            assert_eq!(
                initial_events.len(),
                3,
                "Expected 3 Initial events, got: {:?}",
                initial_events
            );

            let paths: HashSet<_> = initial_events.iter().map(|e| e.path.clone()).collect();
            assert!(paths.contains(&PathBuf::from("existing1.txt")));
            assert!(paths.contains(&PathBuf::from("existing2.txt")));
            assert!(paths.contains(&PathBuf::from("subdir/nested.txt")));

            // .rs file should not have Initial event
            assert!(!events.iter().any(|e| e.path.to_string_lossy().contains("ignored.rs")));
        }

        tracker.lock().unwrap().clear();

        // Verify normal events still work after initial scan
        std::fs::write(test_dir.join("new.txt"), "new content").unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;

        {
            let events = tracker.lock().unwrap();
            let has_create = events
                .iter()
                .any(|e| e.path == PathBuf::from("new.txt") && e.event_type == EventType::Create);
            assert!(has_create, "Expected Create event for new.txt, got: {:?}", events);
        }

        // Cleanup
        watcher_handle.abort();
        let _ = std::fs::remove_dir_all(&test_dir);
    }

    #[tokio::test]
    async fn test_watch_initial_with_dirs() {
        // Test that watch_initial respects match_dirs setting
        let test_dir = std::env::current_dir()
            .unwrap()
            .join(".file-watcher-test-watch_initial_dirs");

        if test_dir.exists() {
            std::fs::remove_dir_all(&test_dir).unwrap();
        }
        std::fs::create_dir(&test_dir).unwrap();

        // Create files and directories
        std::fs::write(test_dir.join("file.txt"), "content").unwrap();
        std::fs::create_dir(test_dir.join("mydir")).unwrap();

        let tracker = Arc::new(Mutex::new(Vec::<Event>::new()));
        let tracker_clone = tracker.clone();
        let test_dir_clone = test_dir.clone();

        let watcher_handle = tokio::spawn(async move {
            let _ = Watcher::new()
                .set_base_dir(&test_dir_clone)
                .add_include("**/*")
                .watch_initial(true)
                .match_files(true)
                .match_dirs(false)  // Only files, not directories
                .run(move |event_type, path| {
                    tracker_clone.lock().unwrap().push(Event {
                        path: path.clone(),
                        event_type: match event_type {
                            WatchEvent::Create => EventType::Create,
                            WatchEvent::Delete => EventType::Delete,
                            WatchEvent::Update => EventType::Update,
                            WatchEvent::Initial => EventType::Initial,
                            WatchEvent::DebugWatch => EventType::DebugWatch,
                        },
                    });
                })
                .await;
        });

        tokio::time::sleep(Duration::from_millis(200)).await;

        {
            let events = tracker.lock().unwrap();
            let initial_events: Vec<_> = events
                .iter()
                .filter(|e| e.event_type == EventType::Initial)
                .collect();

            // Should only have Initial for file.txt, not mydir
            assert_eq!(
                initial_events.len(),
                1,
                "Expected 1 Initial event (file only), got: {:?}",
                initial_events
            );
            assert_eq!(initial_events[0].path, PathBuf::from("file.txt"));
        }

        // Cleanup
        watcher_handle.abort();
        let _ = std::fs::remove_dir_all(&test_dir);
    }

    #[tokio::test]
    async fn test_watch_initial_disabled_by_default() {
        // Test that watch_initial is disabled by default
        let test_dir = std::env::current_dir()
            .unwrap()
            .join(".file-watcher-test-watch_initial_disabled");

        if test_dir.exists() {
            std::fs::remove_dir_all(&test_dir).unwrap();
        }
        std::fs::create_dir(&test_dir).unwrap();

        // Create a file before starting watcher
        std::fs::write(test_dir.join("existing.txt"), "content").unwrap();

        let tracker = Arc::new(Mutex::new(Vec::<Event>::new()));
        let tracker_clone = tracker.clone();
        let test_dir_clone = test_dir.clone();

        let watcher_handle = tokio::spawn(async move {
            let _ = Watcher::new()
                .set_base_dir(&test_dir_clone)
                .add_include("**/*.txt")
                // watch_initial not enabled (default is false)
                .run(move |event_type, path| {
                    tracker_clone.lock().unwrap().push(Event {
                        path: path.clone(),
                        event_type: match event_type {
                            WatchEvent::Create => EventType::Create,
                            WatchEvent::Delete => EventType::Delete,
                            WatchEvent::Update => EventType::Update,
                            WatchEvent::Initial => EventType::Initial,
                            WatchEvent::DebugWatch => EventType::DebugWatch,
                        },
                    });
                })
                .await;
        });

        tokio::time::sleep(Duration::from_millis(200)).await;

        {
            let events = tracker.lock().unwrap();
            let initial_events: Vec<_> = events
                .iter()
                .filter(|e| e.event_type == EventType::Initial)
                .collect();

            // Should have no Initial events since watch_initial is disabled
            assert_eq!(
                initial_events.len(),
                0,
                "Expected no Initial events when watch_initial is disabled, got: {:?}",
                initial_events
            );
        }

        // Cleanup
        watcher_handle.abort();
        let _ = std::fs::remove_dir_all(&test_dir);
    }
}
