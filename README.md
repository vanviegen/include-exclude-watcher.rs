# include-exclude-watcher

Async file watcher with glob-based include/exclude patterns. Linux only (inotify).

## Why this crate?

Most file watchers (like `notify`) give you all events and let you filter afterwards. This works fine for small directories, but wastes resources on large trees when you only care about specific patterns.

This crate:
- Exposes an easy-to-use tokio async/await API
- Supports gitignore-style patterns out of the box
- Only watches directories that could match your include/exclude patterns
- Has built-in debouncing

Tradeoffs:
- Linux only (uses inotify directly)
- Simpler pattern syntax than full gitignore
- Patterns cannot be modified on a running watcher

## Installation

```toml
[dependencies]
include-exclude-watcher = "0.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

## Usage

### Basic watching

```rust
use include_exclude_watcher::{WatchBuilder, WatchEvent};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    WatchBuilder::new()
        .add_include("**/*.rs")
        .add_exclude("**/target/**")
        .run(|event, path| {
            println!("{:?}: {}", event, path.display());
        })
        .await
}
```

This uses the current working directory as its base directory.

### With debouncing

```rust
use include_exclude_watcher::WatchBuilder;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    WatchBuilder::new()
        .set_base_dir("./src")
        .add_include("**/*.rs")
        .run_debounced(500, || {
            println!("Files changed, rebuilding...");
        })
        .await
}
```

### Loading patterns from files

```rust
use include_exclude_watcher::WatchBuilder;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    WatchBuilder::new()
        .set_base_dir("/project")
        .add_include("**/*")
        .add_ignore_file(".gitignore")
        .add_ignore_file(".watchignore")
        .run(|event, path| {
            println!("{:?}: {}", event, path.display());
        })
        .await
}
```

Pattern files use gitignore syntax:
- Lines starting with `#` are comments
- Other non-empty lines are exclude patterns
- Note: `!` negation patterns are not supported (excludes always take precedence over includes)

## Pattern syntax

- `*` matches any characters except `/`
- `**` matches any characters including `/`
- `?` matches any single character except `/`
- `[abc]` matches any character in the set
- Patterns without `/` match anywhere (like gitignore)

Examples:
- `*.rs` → matches `foo.rs` and `src/bar.rs`
- `src/*.rs` → matches `src/main.rs` but not `src/sub/lib.rs`
- `**/test_*.rs` → matches test files anywhere
- `target/**` → excludes everything under target

## CLI tool

```sh
cargo install include-exclude-watcher --features cli
iow ./src -i "**/*.rs" -e "**/target/**"
```

Options:
- `-i, --include <PATTERN>` — Include pattern (can be repeated)
- `-e, --exclude <PATTERN>` — Exclude pattern (can be repeated)
- `-p, --pattern-file <FILE>` — Load patterns from file
- `-r, --run <COMMAND>` — Run shell command on each event (sets `$IOW_FILE` and `$IOW_EVENT`)
- `-c, --combine <MS>` — Debounce and output "CHANGES" after quiet period
- `-x, --exit` — Exit after first change
- `-f, --format <FORMAT>` — Output format: default, path, silent
- `-q, --quiet` — Suppress status messages

## Platform support

**Linux only** for now. Uses inotify directly. PRs for other platforms welcome.

## Comparison with alternatives

| Feature | include-exclude-watcher | notify | watchexec |
|---------|------------------------|--------|-----------|
| Pattern-aware watching | ✓ | ✗ | ✗ |
| Built-in debouncing | ✓ | separate crate | ✓ |
| Cross-platform | ✗ | ✓ | ✓ |
| Async | ✓ | ✓ | ✓ |
| Gitignore file support | ✓ | ✗ | ✓ |

## License

MIT
