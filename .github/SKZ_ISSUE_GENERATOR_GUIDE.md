# SKZ Integration Issue Generator - Usage Guide

> **⚠️ Important Note**: If you're reading this because of **Issue #21**, please see `SKZ_TASK_GENERATOR_RESOLUTION.md` for details about the resolution of incorrect summary data and current project status.

## Overview

The SKZ Integration Issue Generator is a GitHub Action that automatically creates GitHub issues from the `SKZ_INTEGRATION_STRATEGY.md` file. It parses phases and tasks from the markdown and generates structured epic and task issues with proper linking and automation features.

## Features

- **Automatic Parsing**: Reads `SKZ_INTEGRATION_STRATEGY.md` and extracts phases and incomplete tasks
- **Epic Issues**: Creates parent issues for each phase containing incomplete tasks
- **Task Issues**: Creates detailed sub-issues for each incomplete task
- **Automatic Linking**: Links sub-tasks to parent epics via comments
- **Tensor DoF Annotations**: Includes degrees-of-freedom annotations for ggml integration
- **Dry Run Mode**: Test mode that simulates issue creation without actually creating issues
- **Smart Filtering**: Only processes incomplete tasks (unchecked `- [ ]` items)

## How It Works

### Parsing Logic

The action scans `SKZ_INTEGRATION_STRATEGY.md` for:

1. **Phase Headers**: Lines matching `#### Phase N: Title`
2. **Task Items**: Lines matching `- [ ] Task description` (incomplete tasks only)
3. **Completed Tasks**: Lines matching `- [x] Task description` (skipped)

### Issue Creation

For each phase with incomplete tasks:

1. **Epic Issue**: Created with phase overview, attention weights, and acceptance criteria
2. **Task Issues**: Created for each incomplete task with:
   - Actionable steps
   - Acceptance criteria
   - Tensor DoF annotations
   - Technical notes
   - Link to parent epic

## Usage

### Manual Trigger (Recommended for Testing)

1. Go to Actions tab in GitHub repository
2. Select "SKZ Integration Issue Generator"
3. Click "Run workflow"
4. Choose options:
   - **Use workflow from**: Select branch (usually `main`)
   - **Run in dry mode**: Check for testing, uncheck for actual issue creation

### Automatic Trigger

The workflow automatically runs when:
- `SKZ_INTEGRATION_STRATEGY.md` is modified and pushed to `main` branch

## Expected Output

Based on current `SKZ_INTEGRATION_STRATEGY.md`:

- **5 Epic Issues**: One for each phase with incomplete tasks
- **18 Task Issues**: One for each incomplete task
- **1 Summary Issue**: Overview of all created issues
- **Total: 24 Issues**

### Issue Labels

- **Epic Issues**: `epic`, `skz-integration`, `phase-{n}`
- **Task Issues**: `task`, `skz-integration`, `phase-{n}`, `tensor-dof-{value}`
- **Summary Issues**: `summary`, `skz-integration`, `automation`

## Testing

### Dry Run Mode

Always test with dry run mode first:

```bash
# The action will log what it would create without actually creating issues
# Check the Actions tab logs to see the simulation results
```

### Manual Validation

Use the provided test scripts:

```bash
# Test parsing logic
node /tmp/test-skz-parsing.js

# Validate workflow syntax
node /tmp/simple-validate.js

# Run full simulation
node /tmp/simulate-workflow.js
```

## Maintenance

### Updating Strategy

When updating `SKZ_INTEGRATION_STRATEGY.md`:

1. Mark completed tasks with `[x]` to prevent duplicate issues
2. Add new tasks with `[ ]` to create new issues
3. Push changes to trigger automatic issue generation

### Managing Issues

The generated issues are fully functional GitHub issues:

- Assign to team members
- Add to project boards
- Track progress with checkboxes
- Link to pull requests
- Close when completed

## Troubleshooting

### Common Issues

1. **No Issues Created**: Check that tasks are marked as incomplete `[ ]` not `[x]`
2. **Parse Errors**: Ensure proper markdown formatting for phase headers
3. **Permission Errors**: Verify `GITHUB_TOKEN` has `issues:write` permission

### Debugging

- Use dry run mode to test without creating issues
- Check Actions tab logs for detailed execution information
- Verify `SKZ_INTEGRATION_STRATEGY.md` format matches expected pattern

## Integration with SKZ Framework

### Tensor DoF Annotations

Each task issue includes tensor degrees-of-freedom annotations:

```
tensor_dof: {1-10}
```

These values indicate the complexity/dimensionality for future ggml cognitive kernel integration.

### Phase Attention Weights

Epic issues include attention allocation weights:

```
phase_attention: {0.00-1.00}
```

These support future ECAN (Enhanced Cognitive Attention Network) integration.

## Example Output

### Epic Issue
```
Title: Epic: Phase 1: Foundation Setup (CURRENT)
Labels: epic, skz-integration, phase-1
Body: [Phase overview, sub-tasks count, attention weight, acceptance criteria]
```

### Task Issue
```
Title: Create SKZ plugin framework for OJS
Labels: task, skz-integration, phase-1, tensor-dof-9
Body: [Description, actionable steps, acceptance criteria, DoF annotation, epic link]
```

This automated system ensures consistent, detailed issue creation for effective project management of the SKZ Integration initiative.