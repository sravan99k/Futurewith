# Git Cheatsheet

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Setup and Configuration](#setup-and-configuration)
3. [Basic Git Commands](#basic-git-commands)
4. [Branching and Merging](#branching-and-merging)
5. [Remote Repositories](#remote-repositories)
6. [Advanced Operations](#advanced-operations)
7. [Conflict Resolution](#conflict-resolution)
8. [Troubleshooting](#troubleshooting)
9. [Git Hooks and Automation](#git-hooks-and-automation)
10. [Best Practices](#best-practices)

---

## Quick Reference

### Most Common Commands

```bash
# Check status
git status

# Stage all changes
git add .

# Commit with message
git commit -m "Your message"

# View history
git log --oneline

# Switch branches
git checkout branch-name
git switch branch-name  # newer syntax

# Create and switch to new branch
git checkout -b new-branch
git switch -c new-branch  # newer syntax

# Push to remote
git push origin branch-name

# Pull from remote
git pull origin branch-name

# Clone repository
git clone https://github.com/user/repo.git
```

### Emergency Commands

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Undo last commit (create new commit that undoes changes)
git revert HEAD

# Discard local changes
git checkout -- file-name

# Return to last commit (detached HEAD)
git checkout HEAD~1
```

---

## Setup and Configuration

### Initial Setup

```bash
# Install Git (Ubuntu/Debian)
sudo apt-get install git

# Install Git (macOS)
brew install git

# Windows: Download from git-scm.com
```

### Configuration

```bash
# Set global identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set default branch name
git config --global init.defaultBranch main

# Set default editor
git config --global core.editor "code --wait"  # VS Code
git config --global core.editor "nano"        # Nano
git config --global core.editor "vim"         # Vim

# View configuration
git config --list
git config --list --global

# View specific config
git config user.name
git config user.email

# Unset a config
git config --unset user.name
git config --global --unset user.name

# Set up line ending handling (Windows)
git config --global core.autocrlf true

# Set up line ending handling (macOS/Linux)
git config --global core.autocrlf input
```

### SSH Setup for GitHub

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key to clipboard
# macOS:
pbcopy < ~/.ssh/id_ed25519.pub
# Linux:
xclip -sel clip < ~/.ssh/id_ed25519.pub

# Test connection
ssh -T git@github.com
```

---

## Basic Git Commands

### Repository Management

```bash
# Initialize new repository
git init

# Clone repository
git clone https://github.com/user/repo.git
git clone https://github.com/user/repo.git my-folder

# Clone specific branch
git clone -b develop https://github.com/user/repo.git

# Clone with shallow history
git clone --depth 1 https://github.com/user/repo.git
```

### File Operations

```bash
# Add files to staging
git add filename.txt
git add *.py
git add .                 # All files
git add -A               # All files (including deletions)
git add -p              # Interactive staging

# Remove files
git rm filename.txt
git rm --cached filename.txt  # Remove from Git but keep locally

# Move/rename files
git mv old-name.txt new-name.txt

# Check status
git status
git status -s           # Short format
git status -sb          # Short format with branch info
```

### Commit Operations

```bash
# Basic commit
git commit -m "Commit message"

# Add and commit in one step
git commit -am "Message"  # Only for modified files, not new

# Commit with detailed message
git commit

# Amend last commit
git commit --amend
git commit --amend -m "New message"

# Amend without changing message
git commit --amend --no-edit

# Skip pre-commit hooks
git commit --no-verify -m "Message"

# Add signed-off-by line
git commit -s -m "Message"
```

### Viewing History

```bash
# Basic log
git log

# Compact format
git log --oneline

# With graph
git log --graph --oneline --all

# Show specific number of commits
git log -5
git log --since="2 weeks ago"
git log --since="2023-01-01"
git log --until="2023-12-31"

# Filter by author
git log --author="Sarah"
git log --author="Sarah\|Alex"

# Search in commit messages
git log --grep="fix"
git log --grep="fix" --oneline

# Show file changes over time
git log --follow filename.txt
git log -p filename.txt     # Show diff for file
git log --stat filename.txt # Show statistics

# Show who changed each line
git blame filename.txt
git blame -L 10,20 filename.txt  # Blame lines 10-20

# Pretty formats
git log --pretty=format:"%h - %an, %ar : %s"
git log --pretty=format:"%C(yellow)%h%Creset - %C(auto)%ad%s%Creset"
```

### Comparing Changes

```bash
# Show unstaged changes
git diff

# Show staged changes
git diff --staged
git diff --cached

# Compare branches
git diff branch1 branch2
git diff main..feature-branch

# Show file differences
git diff HEAD~1 HEAD filename.txt

# Show word differences
git diff --word-diff

# Show statistics
git diff --stat
```

---

## Branching and Merging

### Branch Management

```bash
# List branches
git branch              # Local branches
git branch -r          # Remote branches
git branch -a          # All branches

# Create branch
git branch feature-branch

# Create and switch to new branch
git checkout -b feature-branch
git switch -c feature-branch  # newer syntax

# Switch branches
git checkout feature-branch
git switch feature-branch     # newer syntax

# Switch to previous branch
git checkout -
git switch -                  # newer syntax

# Rename branch
git branch -m old-name new-name
git branch -m new-name        # Rename current branch

# Delete branch
git branch -d feature-branch          # Delete local branch
git branch -D feature-branch          # Force delete
git push origin --delete feature-branch # Delete remote branch

# Set up tracking
git branch --set-upstream-to=origin/main main
git branch --set-upstream-to=upstream/main main
```

### Merging

```bash
# Merge branch into current branch
git merge feature-branch

# Merge without fast-forward (creates merge commit)
git merge --no-ff feature-branch

# Merge with strategy option
git merge -s recursive -X ours feature-branch
git merge -s recursive -X theirs feature-branch

# Merge with message
git merge feature-branch -m "Merge feature branch"

# Merge specific commit
git cherry-pick <commit-hash>
```

### Rebasing

```bash
# Rebase current branch on top of main
git rebase main

# Interactive rebase
git rebase -i HEAD~3    # Rebase last 3 commits
git rebase -i main      # Rebase on main

# Interactive rebase options:
# p, pick    = use commit
# r, reword  = use commit, but edit message
# e, edit    = use commit, but stop for amending
# s, squash  = use commit, but meld into previous
# f, fixup   = like "squash", but discard message
# x, exec    = run command (the rest of the line) using shell

# Continue rebase after resolving conflicts
git rebase --continue

# Abort rebase
git rebase --abort

# Skip current commit
git rebase --skip

# Rebase onto different branch
git rebase --onto main feature-branch

# Rebase with autosquash
git rebase -i --autosquash main
```

### Cherry-picking

```bash
# Apply specific commit
git cherry-pick <commit-hash>

# Apply range of commits
git cherry-pick <commit-hash1>..<commit-hash2>

# Apply without making commit
git cherry-pick --no-commit <commit-hash>

# Cherry-pick multiple commits
git cherry-pick <commit-hash1> <commit-hash2>
```

---

## Remote Repositories

### Remote Management

```bash
# Add remote
git remote add origin https://github.com/user/repo.git
git remote add upstream https://github.com/original/repo.git

# View remotes
git remote -v

# Rename remote
git remote rename origin upstream

# Remove remote
git remote remove upstream

# Change remote URL
git remote set-url origin git@github.com:user/repo.git

# Get remote information
git remote show origin
```

### Pushing and Pulling

```bash
# Push to remote branch
git push origin main
git push origin feature-branch

# Set upstream tracking
git push -u origin feature-branch

# Push all branches
git push --all

# Push tags
git push --tags
git push origin v1.0.0

# Push force (be careful!)
git push --force
git push --force-with-lease  # Safer force push

# Delete remote branch
git push origin --delete feature-branch

# Push after rebasing
git push --force-with-origin main
```

### Fetching and Pulling

```bash
# Fetch from remote
git fetch origin
git fetch upstream

# Fetch all remotes
git fetch --all

# Fetch and prune deleted branches
git fetch --prune origin

# Fetch specific branch
git fetch origin feature-branch

# Pull changes
git pull origin main
git pull --rebase origin main  # Pull with rebase

# Pull with rebase instead of merge
git config --global pull.rebase true
```

### Cloning

```bash
# Clone repository
git clone https://github.com/user/repo.git

# Clone to specific directory
git clone https://github.com/user/repo.git my-project

# Clone specific branch
git clone -b develop https://github.com/user/repo.git

# Clone with shallow history
git clone --depth 1 https://github.com/user/repo.git

# Clone and recurse submodules
git clone --recursive https://github.com/user/repo.git
```

---

## Advanced Operations

### Stashing

```bash
# Stash changes
git stash
git stash save "Work in progress"

# List stashes
git stash list

# Apply stash
git stash pop          # Apply and remove from stash
git stash apply        # Apply but keep in stash
git stash apply stash@{2}  # Apply specific stash

# Show stash diff
git stash show
git stash show -p      # Show patch

# Create branch from stash
git stash branch new-feature

# Drop stash
git stash drop
git stash drop stash@{1}

# Clear all stashes
git stash clear
```

### Tagging

```bash
# Create annotated tag (recommended)
git tag -a v1.0.0 -m "Release version 1.0.0"

# Create lightweight tag
git tag v1.0.0

# Tag specific commit
git tag -a v0.9.0 <commit-hash> -m "Beta version"

# List tags
git tag
git tag -l "v1.*"

# Show tag information
git show v1.0.0

# Push tags to remote
git push origin --tags
git push origin v1.0.0

# Delete tag locally
git tag -d v0.9.0

# Delete tag from remote
git push origin --delete v0.9.0
```

### Submodules

```bash
# Add submodule
git submodule add https://github.com/user/repo.git

# Clone repository with submodules
git clone --recursive https://github.com/user/repo.git

# Initialize submodules after cloning
git submodule update --init --recursive

# Update submodules
git submodule update --remote

# Remove submodule
git submodule deinit path/to/submodule
git rm path/to/submodule
```

### Reflog

```bash
# View reflog
git reflog
git reflog show HEAD

# Show reflog for specific branch
git reflog show branch-name

# Recover lost commit
git reflog
git checkout <commit-hash-from-reflog>
git checkout -b rescue-branch

# Clean up reflog
git reflog expire --expire=now --all
```

---

## Conflict Resolution

### Basic Conflict Resolution

```bash
# During merge/rebase, conflicts will be marked in files:
# <<<<<<< HEAD
# Current branch content
# =======
# Incoming branch content
# >>>>>>> feature-branch

# Check conflicted files
git status

# After resolving conflicts:
git add <resolved-files>
git commit  # for merge
git rebase --continue  # for rebase

# Abort merge
git merge --abort

# Abort rebase
git rebase --abort
```

### Conflict Resolution Tools

```bash
# Use merge tool
git mergetool

# Available merge tools:
# vim, emacs, tortoisemerge, kdiff3, opendiff, etc.

# Configure merge tool
git config --global merge.tool vimdiff

# Use diff tool to compare
git difftool
```

### Common Conflict Resolution Strategies

```bash
# Keep our changes (current branch)
git checkout --ours <file>

# Keep their changes (incoming branch)
git checkout --theirs <file>

# Keep both (manual merge required)
git checkout --conflict=merge <file>

# For rebase conflicts
git checkout --ours .    # Keep our changes
git checkout --theirs .  # Keep their changes
```

---

## Troubleshooting

### Detached HEAD

```bash
# You are in 'detached HEAD' state
git status  # Shows "detached HEAD"

# Solutions:
# 1. Go back to a branch
git checkout main

# 2. Create a new branch from current state
git checkout -b new-branch

# 3. Return to previous location
git checkout -
```

### Resetting and Reverting

```bash
# Soft reset (keep changes staged)
git reset --soft HEAD~1

# Mixed reset (keep changes in working directory)
git reset HEAD~1
git reset --mixed HEAD~1

# Hard reset (discard all changes)
git reset --hard HEAD~1

# Revert commit (safe undo)
git revert <commit-hash>

# Revert multiple commits
git revert <commit-hash1>..<commit-hash2>

# Revert without committing
git revert --no-commit <commit-hash>
```

### Cleaning Repository

```bash
# Remove untracked files
git clean -f      # Remove files
git clean -fd     # Remove files and directories

# Remove unignored files
git clean -fx

# Interactive clean
git clean -i

# Remove ignored files
git clean -fX
```

### Large Files

```bash
# Find large files
find . -type f -exec du -h {} + | sort -rh | head -20

# Check file size in Git
git ls-tree -r --long HEAD

# Remove file from history (BFG)
java -jar bfg.jar --delete-files "*.large" repo.git

# Remove file with filter-branch
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch path/to/file' \
  --prune-empty --tag-name-filter cat -- --all
```

### Permission Issues

```bash
# Fix file permissions
git config --global core.filemode false

# Check file mode changes
git diff --summary

# If you have permission errors, reset file modes
git ls-files -s | awk '{print $1, $4}' | while read mode file; do
  echo "$file" | xargs chmod $mode
done
```

### Authentication Issues

```bash
# Clear cached credentials
git config --global --unset credential.helper
git config --global credential.helper cache

# Use personal access token instead of password
git remote set-url origin https://username:token@github.com/username/repo.git

# Switch to SSH
git remote set-url origin git@github.com:username/repo.git
```

---

## Git Hooks and Automation

### Pre-commit Hook

```bash
# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/sh
# Run tests before commit
npm test
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
EOF

chmod +x .git/hooks/pre-commit
```

### Commit-msg Hook

```bash
# Create commit-msg hook
cat > .git/hooks/commit-msg << 'EOF'
#!/bin/sh
# Enforce commit message format
commit_regex='^(feat|fix|docs|style|refactor|test|chore)(\(.+\))?: .{1,50}'

if ! grep -qE "$commit_regex" "$1"; then
    echo "Invalid commit message format!"
    echo "Format: type(scope): description"
    echo "Types: feat, fix, docs, style, refactor, test, chore"
    exit 1
fi
EOF

chmod +x .git/hooks/commit-msg
```

### Pre-push Hook

```bash
# Create pre-push hook
cat > .git/hooks/pre-push << 'EOF'
#!/bin/sh
# Run tests and linting before push
echo "Running tests..."
npm test
if [ $? -ne 0 ]; then
    echo "Tests failed. Push aborted."
    exit 1
fi

echo "Running linter..."
npm run lint
if [ $? -ne 0 ]; then
    echo "Linting failed. Push aborted."
    exit 1
fi
EOF

chmod +x .git/hooks/pre-push
```

---

## Best Practices

### Commit Messages

```bash
# Good commit messages
git commit -m "feat: add user authentication system"
git commit -m "fix: resolve null pointer exception in login"
git commit -m "docs: update API documentation"
git commit -m "refactor: simplify user validation logic"
git commit -m "test: add unit tests for user service"
git commit -m "chore: update dependencies to latest versions"

# Bad commit messages
git commit -m "fixed stuff"
git commit -m "update"
git commit -m "asdfasdf"
```

### Commit Message Format

```
type(scope): subject

body (optional)

footer (optional)

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Code style changes (formatting, etc.)
- refactor: Code refactoring
- test: Adding or updating tests
- chore: Build process or auxiliary tool changes

Examples:
feat(auth): add JWT token validation
fix(api): handle empty user profile data
docs(readme): add installation instructions
```

### .gitignore Examples

```bash
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
*.egg-info/
dist/
build/

# JavaScript
node_modules/
npm-debug.log*
yarn-debug.log*
.yarn-integrity
.env.local
.env.development.local
.env.test.local
.env.production.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Environment
.env
.env.local
.env.development.local
.env.test.local
.env.production.local
```

### Branch Naming Conventions

```bash
# Feature branches
feature/user-authentication
feature/weather-api-integration
feature/dashboard-redesign

# Bug fixes
bugfix/login-crash
bugfix/memory-leak
bugfix/cross-browser-compatibility

# Hotfixes
hotfix/security-vulnerability
hotfix/payment-processing-error
hotfix/database-connection

# Releases
release/v1.0.0
release/v2.1.0
release/2023-q4
```

### Useful Aliases

```bash
# Add useful aliases
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual '!gitk'

# Advanced aliases
git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
git config --global alias.cm "commit -m"
git config --global alias.ca "commit --amend"
git config --global alias.dc "diff --cached"
git config --global alias.ls "log --pretty=format:'%C(yellow)%h%Cred%d\\ %Creset%s%Cblue\\ [%cn]' --decorate"
```

### Daily Git Workflow

```bash
# Start of day
git checkout main
git pull origin main
git checkout -b feature/my-new-feature

# Work on feature
git add .
git commit -m "feat: add new functionality"
git push origin feature/my-new-feature

# Before creating PR
git rebase main  # Keep history clean
git log --oneline  # Review changes

# After PR merge
git checkout main
git pull origin main
git branch -d feature/my-new-feature
```

---

## Quick Command Reference by Task

### When I want to...

```bash
# Start a new project
git init && git add . && git commit -m "Initial commit"

# See what I've changed
git status
git diff
git diff --staged

# Undo my last commit
git reset --soft HEAD~1

# Undo my last commit completely
git reset --hard HEAD~1

# See my commit history
git log --oneline --graph --all

# Create a new branch
git checkout -b new-branch
git switch -c new-branch  # newer syntax

# Switch to another branch
git checkout branch-name
git switch branch-name  # newer syntax

# Merge a branch
git checkout main
git merge branch-name

# Fix a merge conflict
git status
# Edit files to resolve conflicts
git add .
git commit

# See what changed between two branches
git log --oneline branch1..branch2
git diff branch1..branch2

# Get latest changes from remote
git fetch origin
git pull origin main

# Share my changes
git push origin branch-name

# Stash my work temporarily
git stash
git stash save "message"
git stash pop

# Use a previous version of a file
git checkout HEAD~1 -- filename.txt

# See who changed a line
git blame filename.txt

# Find which commit introduced a bug
git bisect start
git bisect bad
git bisect good <commit-hash>
```

---

## Emergency Procedures

### Repository Recovery

```bash
# If everything goes wrong
git reflog  # Shows all your moves
git reset --hard HEAD@{number}  # Go back to previous state

# If you deleted a branch
git reflog
git checkout -b recovered-branch <commit-hash>

# If you lost commits
git fsck --full
git log --walk-reflogs

# Start over completely (careful!)
rm -rf .git
git init
git add .
git commit -m "Start over"
```

### Fixing Mistakes

```bash
# Changed my mind about the last commit
git commit --amend

# Committed to wrong branch
git branch new-branch
git reset --hard HEAD~1
git checkout new-branch

# Need to move commits to another branch
git branch new-feature
git reset --hard HEAD~3
git checkout new-feature

# Want to undo a commit but keep the changes
git reset HEAD~1  # Moves branch pointer back
```

Remember: Git is very forgiving! Most mistakes can be undone with the right commands. The key is to stay calm and use the appropriate Git commands to recover your work.
