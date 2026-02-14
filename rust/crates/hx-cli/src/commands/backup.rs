//! Backup and restore commands for Helix CLI.

use anyhow::{bail, Context, Result};
use chrono::Utc;
use std::fs;
use std::io::BufReader;
use std::path::{Component, Path, PathBuf};

use super::{load_config, shellexpand};

#[derive(Debug, Clone, Copy, Default)]
struct BackupValidationReport {
    total_entries: usize,
    file_entries: usize,
    directory_entries: usize,
    total_unpacked_bytes: u64,
}

impl BackupValidationReport {
    fn is_empty(self) -> bool {
        self.total_entries == 0
    }
}

fn validate_restore_entry_path(path: &Path) -> Result<PathBuf> {
    let mut sanitized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => sanitized.push(part),
            Component::CurDir => {}
            Component::ParentDir => {
                bail!(
                    "backup contains parent-directory path traversal: {}",
                    path.display()
                );
            }
            Component::RootDir | Component::Prefix(_) => {
                bail!("backup contains absolute path: {}", path.display());
            }
        }
    }

    if sanitized.as_os_str().is_empty() {
        bail!("backup contains invalid empty entry path");
    }

    Ok(sanitized)
}

fn validate_backup_archive(input_path: &str) -> Result<BackupValidationReport> {
    let file = fs::File::open(input_path)
        .with_context(|| format!("Failed to open backup: {input_path}"))?;
    let decoder = flate2::read::GzDecoder::new(BufReader::new(file));
    let mut archive = tar::Archive::new(decoder);

    let mut report = BackupValidationReport::default();
    for entry in archive
        .entries()
        .context("Failed to read backup archive entries")?
    {
        let entry = entry.context("Failed to read backup archive entry")?;
        let raw_path = entry
            .path()
            .context("Failed to read backup archive entry path")?
            .to_path_buf();
        let sanitized = validate_restore_entry_path(&raw_path)?;
        let entry_type = entry.header().entry_type();

        if entry_type.is_file() {
            report.file_entries += 1;
            report.total_unpacked_bytes = report
                .total_unpacked_bytes
                .saturating_add(entry.header().size().unwrap_or(0));
        } else if entry_type.is_dir() {
            report.directory_entries += 1;
        } else {
            bail!(
                "backup contains unsupported entry type for '{}': {:?}",
                sanitized.display(),
                entry_type
            );
        }
        report.total_entries += 1;
    }

    if report.is_empty() {
        bail!("backup archive contains no entries");
    }

    Ok(report)
}

/// Create a backup of the Helix data.
pub async fn create(output: Option<String>, config_path: &str) -> Result<()> {
    let config = load_config(config_path)?;
    let data_dir = &config.data_dir;

    // Verify data directory exists
    if !Path::new(data_dir).exists() {
        bail!("Data directory does not exist: {data_dir}");
    }

    // Generate backup filename
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
    let backup_name = format!("helix_backup_{timestamp}.tar.gz");
    let output_path = output
        .map(|p| shellexpand(&p))
        .unwrap_or_else(|| format!("{data_dir}/../backups/{backup_name}"));

    // Create backup directory if needed
    if let Some(parent) = Path::new(&output_path).parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create backup directory: {}", parent.display()))?;
    }

    println!("Creating backup of {data_dir}");
    println!("Output: {output_path}");

    // Create tarball
    let file = fs::File::create(&output_path)
        .with_context(|| format!("Failed to create backup file: {output_path}"))?;
    let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    let mut archive = tar::Builder::new(encoder);

    // Add all files in data directory
    let mut file_count = 0;
    for entry in walkdir::WalkDir::new(data_dir)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            let path = entry.path();
            let relative = path.strip_prefix(data_dir).unwrap_or(path);
            archive
                .append_path_with_name(path, relative)
                .with_context(|| format!("Failed to add file to backup: {}", path.display()))?;
            file_count += 1;
        }
    }

    archive
        .finish()
        .context("Failed to finalize backup archive")?;

    let size = fs::metadata(&output_path)?.len();
    let size_str = if size > 1024 * 1024 {
        format!("{:.1} MB", size as f64 / 1024.0 / 1024.0)
    } else if size > 1024 {
        format!("{:.1} KB", size as f64 / 1024.0)
    } else {
        format!("{size} bytes")
    };

    println!();
    println!("Backup complete:");
    println!("  Files:  {file_count}");
    println!("  Size:   {size_str}");
    println!("  Path:   {output_path}");

    Ok(())
}

/// Restore from a backup.
pub async fn restore(input: String, force: bool, config_path: &str) -> Result<()> {
    let config = load_config(config_path)?;
    let data_dir = &config.data_dir;
    let input_path = shellexpand(&input);

    // Verify backup file exists
    if !Path::new(&input_path).exists() {
        bail!("Backup file does not exist: {input_path}");
    }

    let validation_report = validate_backup_archive(&input_path)?;

    // Check if data directory has existing data
    let has_existing_data = Path::new(data_dir).exists()
        && fs::read_dir(data_dir)
            .map(|mut d| d.next().is_some())
            .unwrap_or(false);

    if has_existing_data && !force {
        println!("Data directory already contains files: {data_dir}");
        println!();
        println!("WARNING: Restoring will overwrite existing data.");
        println!("Run with --force to confirm.");
        return Ok(());
    }

    if has_existing_data {
        // Create a pre-restore backup
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let pre_restore_backup = format!("{data_dir}/../backups/pre_restore_{timestamp}.tar.gz");

        println!("Creating pre-restore backup: {pre_restore_backup}");
        create(Some(pre_restore_backup), config_path).await?;
    }

    println!();
    println!("Restoring from: {input_path}");
    println!("Target: {data_dir}");
    println!(
        "Validated archive: {} entries ({} files, {} directories, {} bytes unpacked)",
        validation_report.total_entries,
        validation_report.file_entries,
        validation_report.directory_entries,
        validation_report.total_unpacked_bytes
    );

    // Create data directory
    fs::create_dir_all(data_dir)
        .with_context(|| format!("Failed to create data directory: {data_dir}"))?;

    // Extract tarball
    let file =
        fs::File::open(&input_path).with_context(|| format!("Failed to open: {input_path}"))?;
    let decoder = flate2::read::GzDecoder::new(BufReader::new(file));
    let mut archive = tar::Archive::new(decoder);

    let mut file_count = 0usize;
    let mut dir_count = 0usize;
    for entry in archive.entries().context("Failed to read backup entries")? {
        let mut entry = entry.context("Failed to read backup entry")?;
        let raw_path = entry
            .path()
            .context("Failed to get backup entry path")?
            .to_path_buf();
        let relative_path = validate_restore_entry_path(&raw_path)?;
        let dest = Path::new(data_dir).join(&relative_path);
        let entry_type = entry.header().entry_type();

        if entry_type.is_dir() {
            fs::create_dir_all(&dest).with_context(|| {
                format!("Failed to create destination directory: {}", dest.display())
            })?;
            dir_count += 1;
            continue;
        }
        if !entry_type.is_file() {
            bail!(
                "Encountered unsupported backup entry type during restore for '{}': {:?}",
                relative_path.display(),
                entry_type
            );
        }

        // Create parent directories
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }

        entry
            .unpack(&dest)
            .with_context(|| format!("Failed to extract: {}", dest.display()))?;
        file_count += 1;
    }

    println!();
    println!("Restore complete:");
    println!("  Files restored: {file_count}");
    println!("  Directories created: {dir_count}");
    println!("  Data directory: {data_dir}");

    Ok(())
}

/// List available backups.
pub async fn list(config_path: &str) -> Result<()> {
    let config = load_config(config_path)?;
    let data_dir = &config.data_dir;
    let backup_dir = format!("{data_dir}/../backups");

    if !Path::new(&backup_dir).exists() {
        println!("No backups found.");
        println!("Backup directory: {backup_dir}");
        return Ok(());
    }

    let mut backups: Vec<_> = fs::read_dir(&backup_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|ext| ext == "gz").unwrap_or(false))
        .collect();

    if backups.is_empty() {
        println!("No backups found in: {backup_dir}");
        return Ok(());
    }

    // Sort by modification time (newest first)
    backups.sort_by(|a, b| {
        b.metadata()
            .and_then(|m| m.modified())
            .ok()
            .cmp(&a.metadata().and_then(|m| m.modified()).ok())
    });

    println!("Available backups:");
    println!();

    for entry in backups {
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        let size_str = if size > 1024 * 1024 {
            format!("{:.1} MB", size as f64 / 1024.0 / 1024.0)
        } else if size > 1024 {
            format!("{:.1} KB", size as f64 / 1024.0)
        } else {
            format!("{size} bytes")
        };

        let modified = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .map(|t| {
                let dt: chrono::DateTime<Utc> = t.into();
                dt.format("%Y-%m-%d %H:%M:%S").to_string()
            })
            .unwrap_or_else(|| "unknown".into());

        println!("  {name}");
        println!("    Size: {size_str}  Modified: {modified}");
        println!();
    }

    println!("Backup directory: {backup_dir}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_backup(bytes: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "helix-backup-test-{}.tar.gz",
            uuid::Uuid::now_v7()
        ));
        let mut file = fs::File::create(&path).expect("temp backup should create");
        file.write_all(bytes).expect("temp backup should write");
        path
    }

    fn build_backup_archive(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        let mut builder = tar::Builder::new(encoder);

        for (path, payload) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(payload.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, *path, &payload[..])
                .expect("archive entry should append");
        }

        builder.finish().expect("archive should finish");
        let encoder = builder.into_inner().expect("encoder should return");
        encoder.finish().expect("gzip should finalize")
    }

    #[test]
    fn validate_restore_entry_path_accepts_relative_segments() {
        let path = Path::new("store/nodes.sqlite");
        let sanitized = validate_restore_entry_path(path).expect("path should be valid");
        assert_eq!(sanitized, PathBuf::from("store/nodes.sqlite"));
    }

    #[test]
    fn validate_restore_entry_path_rejects_parent_traversal() {
        let path = Path::new("../escape.txt");
        let err = validate_restore_entry_path(path).expect_err("parent traversal should fail");
        assert!(err.to_string().contains("path traversal"));
    }

    #[test]
    fn validate_restore_entry_path_rejects_absolute_path() {
        let path = Path::new("/etc/passwd");
        let err = validate_restore_entry_path(path).expect_err("absolute path should fail");
        assert!(err.to_string().contains("absolute path"));
    }

    #[test]
    fn validate_backup_archive_accepts_regular_files() {
        let archive_bytes = build_backup_archive(&[
            ("index/meta.json", br#"{"ok":true}"#),
            ("nodes/store.db", b"sqlite-bytes"),
        ]);
        let archive_path = write_temp_backup(&archive_bytes);

        let report = validate_backup_archive(archive_path.to_str().expect("path str"))
            .expect("archive should validate");
        assert_eq!(report.file_entries, 2);
        assert_eq!(report.total_entries, 2);
        assert!(report.total_unpacked_bytes >= 20);

        let _ = fs::remove_file(archive_path);
    }

    #[test]
    fn validate_backup_archive_rejects_empty_archive() {
        let archive_bytes = build_backup_archive(&[]);
        let archive_path = write_temp_backup(&archive_bytes);

        let err = validate_backup_archive(archive_path.to_str().expect("path str"))
            .expect_err("empty archive should fail");
        assert!(err.to_string().contains("no entries"));

        let _ = fs::remove_file(archive_path);
    }
}
