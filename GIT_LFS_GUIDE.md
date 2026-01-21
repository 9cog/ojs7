# Git Large File Storage (LFS) Guide

## Overview

This repository uses Git LFS to handle large files efficiently, preventing repository bloat and improving clone/push performance.

## Problem Statement

Without LFS, large files cause:
- Slow clone/push operations
- Repository size bloat
- GitHub push failures (files >100MB)
- Poor performance for binary files

## Current LFS Configuration

### Tracked File Types

The `.gitattributes` file configures the following patterns for LFS:

#### Essential Large Files
- `audit_results.json` - Production audit reports (can be 10MB+)
- `tree.txt`, `treeu.txt`, `tree.md` - Directory structure files
- `*.json` - All JSON files to prevent large data files

#### Binary Files
- `*.dll`, `*.so`, `*.dylib` - Native libraries
- `*.zip`, `*.tar.gz`, `*.rar` - Archives

#### AI/ML Files
- `*.model`, `*.h5`, `*.pkl` - Machine learning models
- `*.onnx`, `*.bin` - AI inference models
- `*.csv`, `*.parquet` - Large datasets

#### Media Files
- `*.mp4`, `*.mov`, `*.avi` - Videos
- `*.wav`, `*.mp3` - Audio files
- `*.psd`, `*.tiff` - Large images

#### Database & Logs
- `*.db`, `*.sqlite`, `*.sqlite3` - Database files
- `*.log` - Log files that can grow large

## Usage Instructions

### For New Clones

```bash
# Clone the repository
git clone https://github.com/EchoCog/ojs-7.1.git
cd ojs-7.1

# Pull LFS files (happens automatically with modern Git LFS)
git lfs pull
```

### For Developers

#### Adding New Large Files

1. Check if file type is already tracked:
   ```bash
   git lfs track
   ```

2. If not tracked, add the pattern:
   ```bash
   git lfs track "*.newextension"
   ```

3. Commit the `.gitattributes` change:
   ```bash
   git add .gitattributes
   git commit -m "Track *.newextension files with LFS"
   ```

#### Working with LFS Files

```bash
# Check LFS status
git lfs status

# List LFS files
git lfs ls-files

# Check LFS file info
git lfs ls-files --size

# Verify LFS installation
git lfs version
```

### Troubleshooting

#### Push Errors with Large Files

If you get errors like "file too large" during push:

1. **Check if file should be in LFS**:
   ```bash
   ls -lh <large-file>
   git lfs track
   ```

2. **If file type isn't tracked, add it**:
   ```bash
   git lfs track "<filename>"
   git add .gitattributes <filename>
   git commit -m "Track <filename> with LFS"
   ```

3. **If file is already committed to Git normally**:
   - The file might need migration (contact maintainers)
   - Or exclude it via `.gitignore` if it's generated

#### LFS Not Working

1. **Ensure LFS is installed**:
   ```bash
   git lfs install
   ```

2. **Check hooks are installed**:
   ```bash
   ls -la .git/hooks/
   # Should see pre-push and post-checkout hooks
   ```

3. **Verify tracking**:
   ```bash
   git check-attr filter <filename>
   # Should show: <filename>: filter: lfs
   ```

## Best Practices

### Do Track with LFS
- Files larger than 10MB
- Binary files (executables, libraries)
- Generated reports and logs
- AI models and datasets
- Media files

### Don't Track with LFS
- Source code files
- Small configuration files
- Text-based documentation
- Files that change frequently and are small

### Repository Maintenance

#### Cleaning Up
```bash
# Clean local LFS cache (frees disk space)
git lfs prune

# Check what would be pruned
git lfs prune --dry-run
```

#### Monitoring
```bash
# Check repository LFS usage
git lfs ls-files --size

# Check LFS bandwidth usage (on GitHub)
# Visit repository settings > Git LFS
```

## Migration Notes

### Existing Large Files

Large files that were committed before LFS was configured:
- `audit_results.json` (17MB) - Now tracked with LFS
- `tree.txt` files (185KB-231KB) - Now tracked with LFS
- PHP DLL files - Now tracked with LFS

These files remain in Git history but new versions will use LFS.

### Safe Migration Process

For critical files that need migration:

1. **Create backup branch**
2. **Use git-lfs-migrate** (advanced users only)
3. **Test thoroughly before force-pushing**

**Note**: Migration rewrites history and should be coordinated with all contributors.

## Support

For LFS-related issues:
1. Check this guide first
2. Verify installation with `git lfs version`
3. Contact repository maintainers for migration needs
4. See GitHub's LFS documentation for advanced topics

## Links

- [Git LFS Documentation](https://git-lfs.github.io/)
- [GitHub LFS Guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files)
- [Git LFS Tutorial](https://github.com/git-lfs/git-lfs/wiki/Tutorial)