//! iow - Include/exclude file watcher CLI tool
//!
//! Watch files and directories for changes with glob-based filtering.

use clap::Parser;
use include_exclude_watcher::{WatchBuilder, WatchEvent};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "iow")]
#[command(version)]
#[command(about = "Watch files and directories for changes", long_about = None)]
struct Args {
    /// Directory to watch
    #[arg(value_name = "PATH")]
    path: PathBuf,

    /// Include patterns (glob-style)
    #[arg(short = 'i', long = "include", value_name = "PATTERN")]
    includes: Vec<String>,

    /// Exclude patterns (glob-style)
    #[arg(short = 'e', long = "exclude", value_name = "PATTERN")]
    excludes: Vec<String>,

    /// Load patterns from a gitignore-style file
    #[arg(short = 'p', long = "pattern-file", value_name = "FILE")]
    pattern_files: Vec<PathBuf>,

    /// Watch for file/directory creation events
    #[arg(long = "create", default_value = "true", action = clap::ArgAction::Set)]
    watch_create: bool,

    /// Watch for file/directory deletion events
    #[arg(long = "delete", default_value = "true", action = clap::ArgAction::Set)]
    watch_delete: bool,

    /// Watch for file modification events
    #[arg(long = "modify", default_value = "true", action = clap::ArgAction::Set)]
    watch_modify: bool,

    /// Match regular files
    #[arg(long = "files", default_value = "true", action = clap::ArgAction::Set)]
    match_files: bool,

    /// Match directories
    #[arg(long = "dirs", default_value = "true", action = clap::ArgAction::Set)]
    match_dirs: bool,

    /// Output format: 'default' (CREATE/DELETE/UPDATE + path), 'path' (path only), 'silent' (no output)
    #[arg(short = 'f', long = "format", value_name = "FORMAT", default_value = "default")]
    format: String,

    /// Exit on first change detected
    #[arg(short = 'x', long = "exit")]
    exit_on_first: bool,

    /// Combine changes with debouncing (outputs "CHANGES" after quiet period)
    #[arg(short = 'c', long = "combine", value_name = "MS")]
    combine: Option<u64>,

    /// Run shell command on each event (sets $IOW_FILE and $IOW_EVENT)
    #[arg(short = 'r', long = "run", value_name = "COMMAND")]
    run_command: Option<String>,

    /// Quiet mode (suppress initial status messages)
    #[arg(short = 'q', long = "quiet")]
    quiet: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum OutputFormat {
    Default,
    Path,
    Silent,
}

impl OutputFormat {
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "default" => Ok(OutputFormat::Default),
            "path" => Ok(OutputFormat::Path),
            "silent" => Ok(OutputFormat::Silent),
            _ => Err(format!(
                "Invalid format '{}'. Valid options: default, path, silent",
                s
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EventType {
    Create,
    Delete,
    Update,
}

impl EventType {
    fn as_str(&self) -> &'static str {
        match self {
            EventType::Create => "CREATE",
            EventType::Delete => "DELETE",
            EventType::Update => "UPDATE",
        }
    }
}

fn run_shell_command(cmd: &str, file: &str, event: &str) {
    let _ = std::process::Command::new("/bin/sh")
        .arg("-c")
        .arg(cmd)
        .env("IOW_FILE", file)
        .env("IOW_EVENT", event)
        .status();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let format = OutputFormat::from_str(&args.format).map_err(|e| anyhow::anyhow!(e))?;

    let exit_on_first = args.exit_on_first || format == OutputFormat::Silent;

    let includes = if args.includes.is_empty() {
        vec!["**/*".to_string()]
    } else {
        args.includes
    };

    if !args.quiet {
        eprintln!("Watching: {}", args.path.display());
        eprintln!("Includes: {:?}", includes);
        if !args.excludes.is_empty() {
            eprintln!("Excludes: {:?}", args.excludes);
        }
        if !args.pattern_files.is_empty() {
            eprintln!("Pattern files: {:?}", args.pattern_files);
        }
        eprintln!("Format: {:?}", format);
        if exit_on_first {
            eprintln!("Exit on first change: enabled");
        }
        if let Some(ms) = args.combine {
            eprintln!("Combine mode: {}ms debounce", ms);
        }
        if let Some(ref cmd) = args.run_command {
            eprintln!("Run command: {}", cmd);
        }
        eprintln!("---");
    }

    let mut builder = WatchBuilder::new()
        .set_base_dir(&args.path)
        .add_includes(includes)
        .add_excludes(args.excludes)
        .watch_create(args.watch_create)
        .watch_delete(args.watch_delete)
        .watch_update(args.watch_modify)
        .match_files(args.match_files)
        .match_dirs(args.match_dirs);

    for pattern_file in args.pattern_files {
        builder = builder.add_ignore_file(pattern_file);
    }

    let run_command = args.run_command;

    if let Some(debounce_ms) = args.combine {
        let run_cmd = run_command.clone();
        if exit_on_first {
            builder
                .run_debounced(debounce_ms, move || {
                    println!("CHANGES");
                    if let Some(ref cmd) = run_cmd {
                        run_shell_command(cmd, "", "CHANGES");
                    }
                    std::process::exit(0);
                })
                .await?;
        } else {
            builder
                .run_debounced(debounce_ms, move || {
                    println!("CHANGES");
                    if let Some(ref cmd) = run_cmd {
                        run_shell_command(cmd, "", "CHANGES");
                    }
                })
                .await?;
        }
    } else if exit_on_first {
        builder
            .run(move |event, path| {
                let event_type = match event {
                    WatchEvent::Create => EventType::Create,
                    WatchEvent::Delete => EventType::Delete,
                    WatchEvent::Update => EventType::Update,
                    WatchEvent::DebugWatch => return,
                };

                match format {
                    OutputFormat::Default => {
                        println!("{} {}", event_type.as_str(), path.display());
                    }
                    OutputFormat::Path => {
                        println!("{}", path.display());
                    }
                    OutputFormat::Silent => {}
                }

                if let Some(ref cmd) = run_command {
                    run_shell_command(cmd, &path.to_string_lossy(), event_type.as_str());
                }

                std::process::exit(0);
            })
            .await?;
    } else {
        builder
            .run(move |event, path| {
                let event_type = match event {
                    WatchEvent::Create => EventType::Create,
                    WatchEvent::Delete => EventType::Delete,
                    WatchEvent::Update => EventType::Update,
                    WatchEvent::DebugWatch => return,
                };

                match format {
                    OutputFormat::Default => {
                        println!("{} {}", event_type.as_str(), path.display());
                    }
                    OutputFormat::Path => {
                        println!("{}", path.display());
                    }
                    OutputFormat::Silent => {}
                }

                if let Some(ref cmd) = run_command {
                    run_shell_command(cmd, &path.to_string_lossy(), event_type.as_str());
                }
            })
            .await?;
    }

    Ok(())
}
