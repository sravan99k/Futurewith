# Git Interview Questions

## Table of Contents

1. [Basic Git Concepts](#basic-git-concepts)
2. [Branching and Merging](#branching-and-merging)
3. [Remote Repositories](#remote-repositories)
4. [Advanced Git Operations](#advanced-git-operations)
5. [Conflict Resolution](#conflict-resolution)
6. [Workflows and Best Practices](#workflows-and-best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Scenario-Based Questions](#scenario-based-questions)
9. [System Design with Git](#system-design-with-git)
10. [Practical Coding Questions](#practical-coding-questions)

---

## Basic Git Concepts

### Question 1: What is Git and how is it different from other version control systems?

**Sample Answer:**
Git is a distributed version control system created by Linus Torvalds in 2005. Unlike centralized systems like SVN or CVS, Git is distributed, meaning every developer has a complete copy of the repository history on their local machine. This provides several advantages:

- **No single point of failure**: If the central server crashes, all developers can continue working locally
- **Better performance**: Most operations are local, making them faster
- **Full offline capability**: Developers can work without internet connection
- **Branching efficiency**: Creating and merging branches is fast and easy
- **Data integrity**: Uses SHA-1 checksums to ensure data integrity
- **Non-linear development**: Supports parallel development streams

**Key Points to Mention:**

- Distributed vs centralized architecture
- Local vs remote operations
- Branching and merging capabilities
- Data integrity features

### Question 2: Explain the three main states of files in Git.

**Sample Answer:**
In Git, files exist in three main states:

1. **Working Directory**: This is your local file system where you edit files. When you modify a file, it's in the working directory state. Git doesn't track changes here automatically.

2. **Staging Area (Index)**: This is a virtual area where you prepare changes for a commit. When you use `git add`, you're moving changes from the working directory to the staging area. The staging area acts as a buffer where you can choose exactly what goes into your next commit.

3. **Repository**: This is where Git permanently stores committed changes. When you run `git commit`, changes from the staging area are saved to the repository with a unique hash and become part of the project's permanent history.

**Workflow**: Working Directory → Staging Area → Repository

**Key Points to Mention:**

- The role of each state
- How files move between states
- Why the staging area is important
- How to check the current state using `git status`

### Question 3: What is a commit hash and how is it used?

**Sample Answer:**
A commit hash is a unique 40-character SHA-1 identifier (often displayed as 7 characters for brevity) that Git generates for each commit. It's computed from:

- The commit message
- The author and timestamp
- The parent commit(s)
- The changes made

**How it's used:**

- **Uniqueness**: Each commit has a unique identifier, preventing conflicts
- **Reference**: You can reference commits by their hash for operations like `git show`, `git checkout`, `git revert`
- **History tracking**: The hash allows Git to maintain the complete history and detect any changes
- **Safety**: If anything changes, the hash changes, making tampering obvious

**Example Usage:**

```bash
git show 1234567
git checkout 1234567
git revert 1234567
```

**Key Points to Mention:**

- SHA-1 hash computation
- Uniqueness and integrity
- Practical usage in Git commands
- Short vs long hash format

### Question 4: What is the difference between `git add` and `git commit`?

**Sample Answer:**

- **`git add`**: Moves changes from the working directory to the staging area. You can add specific files (`git add file.txt`), all changes (`git add .`), or interactively choose what to stage (`git add -p`). This allows you to control exactly what goes into your next commit.

- **`git commit`**: Takes all changes from the staging area and creates a permanent snapshot in the repository. The command creates a new commit object with a unique hash, stores it in the repository, and moves the current branch pointer to this new commit.

**Why the separation?**
The staging area allows you to:

- Prepare your commits thoughtfully
- Include only related changes in one commit
- Exclude work-in-progress changes
- Review what you're about to commit

**Key Points to Mention:**

- The role of the staging area
- Granular control over commits
- The commit creation process
- Best practices for committing

### Question 5: How do you check the status of your repository?

**Sample Answer:**
Use `git status` to see the current state of your repository:

```bash
git status
git status -s     # Short format
git status -sb    # Short format with branch info
```

**What it shows:**

- **On branch main**: Which branch you're currently on
- **Changes to be committed**: Files in the staging area (green)
- **Changes not staged for commit**: Modified files not yet staged (red)
- **Untracked files**: New files Git isn't tracking
- **Your branch is up to date**: Status of your branch vs remote

**Example Output:**

```
On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   src/app.js

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
        modified:   src/utils.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        new_feature.py
```

**Key Points to Mention:**

- How to interpret status output
- Color coding (if applicable)
- Different status options
- What each section means

---

## Branching and Merging

### Question 6: What is a Git branch and how do you create and switch between branches?

**Sample Answer:**
A Git branch is a lightweight, movable pointer to a commit. Branches allow you to develop features, fix bugs, or experiment without affecting the main codebase.

**Creating branches:**

```bash
# Create a new branch (doesn't switch to it)
git branch feature-login

# Create and switch to new branch (most common)
git checkout -b feature-login
git switch -c feature-login  # newer syntax
```

**Switching between branches:**

```bash
# Switch to an existing branch
git checkout feature-login
git switch feature-login

# Switch to previous branch
git checkout -
git switch -
```

**How branches work:**

- Each branch is a pointer to a specific commit
- The `HEAD` pointer indicates your current position
- When you make commits, the branch pointer moves forward
- Switching branches updates your working directory to match that branch

**Branch management:**

```bash
git branch        # List local branches
git branch -r     # List remote branches
git branch -a     # List all branches
git branch -d feature-login  # Delete branch
```

**Key Points to Mention:**

- Branch as a pointer concept
- How to create and switch branches
- The HEAD pointer
- Branch management commands

### Question 7: Explain the difference between `git merge` and `git rebase`.

**Sample Answer:**
Both commands integrate changes from one branch into another, but they work differently:

**`git merge`:**

- Creates a new "merge commit" that combines the histories
- Preserves the exact branch history
- Non-destructive operation
- Creates a commit even for fast-forward merges (unless specified)

**Workflow:**

```bash
git checkout main
git merge feature-branch
```

**Result:** Main branch has all commits from main + all commits from feature branch + a merge commit

**`git rebase`:**

- Moves the entire branch to start from the tip of another branch
- Creates a linear, cleaner history
- Rewrites commit history (should not use on public branches)
- No merge commit created

**Workflow:**

```bash
git checkout feature-branch
git rebase main
```

**Result:** Feature branch commits are replayed on top of main, creating a linear history

**When to use:**

- **Use merge when**: You want to preserve branch history, working on team branches
- **Use rebase when**: You want a clean linear history, working on your own feature branch

**Safety rule:** Never rebase public branches that others are working on

**Key Points to Mention:**

- How each command works
- Historical preservation vs rewriting
- Linear vs branched history
- When to use each approach
- Safety considerations

### Question 8: What is a fast-forward merge?

**Sample Answer:**
A fast-forward merge occurs when the branch you're merging into has not moved forward since you branched from it. Git simply moves the branch pointer forward to point to the same commit as the incoming branch.

**Scenario:**

```
main:     A -- B -- C
               \
feature:         D -- E
```

When you merge feature into main:

```
main:     A -- B -- C -- D -- E
                              \
feature:                     D -- E
```

Git just moves the main pointer to point to E, creating no merge commit.

**How to enable/disable:**

```bash
# Explicitly allow fast-forward
git merge --ff feature-branch

# Disable fast-forward (always create merge commit)
git merge --no-ff feature-branch

# Only allow fast-forward (fail if not possible)
git merge --ff-only feature-branch
```

**When it happens:**

- When the target branch hasn't been updated
- When you're merging a branch that branched from the current branch
- When there are no conflicting changes

**Benefits:**

- Cleaner history
- Fewer merge commits
- Faster operation

**Key Points to Mention:**

- Pointer movement concept
- When it occurs
- How to control it
- Benefits of fast-forward merges

### Question 9: How do you resolve merge conflicts?

**Sample Answer:**
Merge conflicts occur when Git cannot automatically merge changes. You need to manually resolve the conflicts and complete the merge.

**Steps to resolve:**

1. **Identify conflicts:**

   ```bash
   git status  # Shows conflicted files
   ```

2. **Open the conflicted file** and look for conflict markers:

   ```git
   <<<<<<< HEAD
   Current branch content
   =======
   Incoming branch content
   >>>>>>> feature-branch
   ```

3. **Choose the correct version** or combine both:
   - Edit the file to keep the desired content
   - Remove the conflict markers (<<<<<<, =======, >>>>>>)

4. **Stage the resolved file:**

   ```bash
   git add resolved-file.txt
   ```

5. **Complete the merge:**
   ```bash
   git commit  # Use default merge message or customize
   ```

**Alternative tools:**

```bash
# Use merge tool
git mergetool

# Abort merge if needed
git merge --abort
```

**Best practices:**

- Review both changes carefully
- Test your resolution
- Don't just choose one version blindly
- Communicate with team members
- Commit the resolution

**Key Points to Mention:**

- How conflicts occur
- Reading conflict markers
- Manual resolution process
- Tools available for assistance
- Best practices

### Question 10: What is a detached HEAD state and how do you recover from it?

**Sample Answer:**
A detached HEAD state occurs when you check out a specific commit (by hash) instead of a branch. In this state, you're not on any branch, and any commits you make can be lost if you switch away.

**How it happens:**

```bash
git checkout 1234567  # Checkout specific commit
```

**How to recognize it:**

```bash
git status
# Shows: "You are in 'detached HEAD' state"
```

**How to recover:**

1. **Create a new branch** from the current state:

   ```bash
   git checkout -b new-branch
   ```

2. **Go back to an existing branch** (losing uncommitted work):

   ```bash
   git checkout main
   ```

3. **Return to previous location:**
   ```bash
   git checkout -
   ```

**When might you want to use it:**

- Examining old code
- Testing specific versions
- Creating branches from historical points
- Debugging specific commits

**Prevention:**
Always check what you're checking out:

```bash
git log --oneline  # See recent commits
git branch --list  # See available branches
```

**Key Points to Mention:**

- What detached HEAD means
- How it occurs
- Recovery options
- When it might be useful
- How to avoid accidentally entering it

---

## Remote Repositories

### Question 11: What is a remote repository and how do you add one?

**Sample Answer:**
A remote repository is a version of your project stored on a server (like GitHub, GitLab, or Bitbucket). It serves as a backup and enables collaboration between developers.

**Adding a remote:**

```bash
# Add remote with name "origin"
git remote add origin https://github.com/username/repo.git

# Add multiple remotes
git remote add upstream https://github.com/original/repo.git
git remote add fork https://github.com/username/fork.git

# Rename remote
git remote rename origin upstream

# Change remote URL
git remote set-url origin git@github.com:username/repo.git

# Remove remote
git remote remove upstream
```

**Viewing remotes:**

```bash
git remote -v  # Shows name and URL
git remote show origin  # Detailed information
```

**Why remotes are important:**

- Backup and redundancy
- Team collaboration
- Code sharing and review
- Deployment integration
- Open source contribution

**Key Points to Mention:**

- Definition of remote repository
- How to add, modify, and remove remotes
- Common remote management commands
- Why remotes are important for collaboration

### Question 12: Explain the difference between `git fetch` and `git pull`.

**Sample Answer:**
Both commands download changes from remote repositories, but they differ in what they do with the downloaded changes:

**`git fetch`:**

- Downloads changes from remote
- Does NOT automatically merge changes into your working directory
- Updates remote-tracking branches (origin/main, origin/feature, etc.)
- Allows you to review changes before merging
- Safer operation as it doesn't modify your current work

**Basic usage:**

```bash
git fetch origin
git fetch --all  # Fetch from all remotes
git fetch --prune  # Remove deleted remote branches
```

**`git pull`:**

- Downloads changes from remote
- Automatically merges changes into your current branch
- Equivalent to: `git fetch` + `git merge`
- May cause conflicts you need to resolve
- Faster but potentially risky

**Basic usage:**

```bash
git pull origin main
git pull --rebase origin main  # Use rebase instead of merge
```

**When to use each:**

- **Use fetch** when you want to review changes before merging
- **Use pull** when you trust the changes and want quick integration
- **Use fetch** for large team projects where you want to review first
- **Use pull** for personal projects or after review

**Best practice:** Many teams prefer `git fetch` followed by `git rebase` or `git merge` to maintain control over the process.

**Key Points to Mention:**

- What each command does
- Automatic vs manual merging
- Safety considerations
- When to use each approach
- Popular workflows

### Question 13: How do you push a local branch to a remote repository?

**Sample Answer:**
To push a local branch to a remote repository, follow these steps:

**Basic push:**

```bash
# Push local branch to remote
git push origin feature-new-feature

# Set up tracking (so future pushes don't need origin/branch)
git push -u origin feature-new-feature
```

**After setting upstream tracking:**

```bash
# Future pushes are simpler
git push  # Pushes current branch to its tracked branch
git pull  # Pulls from tracked branch
```

**Managing remote branches:**

```bash
# Push all local branches
git push --all

# Push tags
git push --tags
git push origin v1.0.0

# Push with force (careful!)
git push --force
git push --force-with-lease  # Safer force push

# Delete remote branch
git push origin --delete feature-old-branch
```

**Common scenarios:**

1. **First time pushing a new branch:**

   ```bash
   git checkout -b feature-login
   # Make changes
   git add .
   git commit -m "Add login feature"
   git push -u origin feature-login
   ```

2. **Pushing existing branch:**
   ```bash
   git checkout main
   git push origin main
   ```

**Troubleshooting:**

```bash
# Check remote configuration
git remote -v

# Check tracking configuration
git branch -vv

# See what would be pushed
git push --dry-run
```

**Key Points to Mention:**

- Basic push command
- Setting up tracking
- When to use force push
- Common scenarios and troubleshooting

### Question 14: What is a pull request and how do you create one?

**Sample Answer:**
A pull request (PR) is a method of submitting contributions to a project. It allows you to propose changes, request code review, and discuss modifications before merging them into the main codebase.

**What a pull request provides:**

- Code review process
- Discussion and feedback
- Automated testing and checks
- Change history tracking
- Permission and access control

**How to create a pull request (using GitHub as example):**

1. **Create a feature branch:**

   ```bash
   git checkout main
   git checkout -b feature-improve-ui
   ```

2. **Make your changes:**

   ```bash
   # Edit files
   git add .
   git commit -m "Improve user interface design"
   git push -u origin feature-improve-ui
   ```

3. **Create PR on GitHub:**
   - Go to your repository on GitHub
   - Click "Compare & pull request" or go to Pull Requests tab
   - Click "New pull request"
   - Select base branch (main) and compare branch (feature-improve-ui)

4. **Fill in PR details:**
   - Title: Clear, descriptive title
   - Description: What changes, why, how to test
   - Reviewers: Request specific reviewers
   - Labels: Categorize the PR
   - Linked issues: Connect to related issues

5. **Example PR description:**

   ```markdown
   ## What this PR does

   Improves the user interface with better color scheme and layout

   ## Changes made

   - Updated color palette for better accessibility
   - Improved button spacing and typography
   - Added responsive design for mobile devices

   ## Testing

   - [ ] Tested on desktop (Chrome, Firefox, Safari)
   - [ ] Tested on mobile (iOS, Android)
   - [ ] Color contrast meets WCAG guidelines

   ## Screenshots

   [Add before/after screenshots]
   ```

6. **Submit and wait for review**

**Review process:**

- Team members review your code
- Leave comments and suggestions
- Request changes if needed
- Approve and merge when ready

**Types of pull requests:**

- **Feature PRs**: New functionality
- **Bug fix PRs**: Fix existing issues
- **Documentation PRs**: Update docs
- **Refactor PRs**: Improve code structure

**Key Points to Mention:**

- What a pull request is
- Why they're important
- Step-by-step creation process
- Best practices for PR descriptions
- The review process

### Question 15: How do you handle synchronization with remote repositories?

**Sample Answer:**
Keeping your local repository synchronized with remote is crucial for collaboration:

**Daily synchronization workflow:**

1. **Start of day - Pull latest changes:**

   ```bash
   git checkout main
   git pull origin main
   ```

2. **Before pushing - Ensure you're up to date:**

   ```bash
   git pull origin main  # Or: git fetch + git rebase
   ```

3. **Push your changes:**
   ```bash
   git push origin feature-branch
   ```

**Advanced synchronization:**

**Fetch and review before merging:**

```bash
# Download changes without merging
git fetch origin

# Review what changed
git log HEAD..origin/main
git diff HEAD..origin/main

# Merge when ready
git merge origin/main
```

**Rebase for clean history:**

```bash
# Fetch and rebase (cleaner history)
git fetch origin
git rebase origin/main
```

**Handle multiple remotes:**

```bash
# Add multiple remotes
git remote add upstream https://github.com/original/repo.git

# Pull from different remotes
git pull origin main      # From your fork
git pull upstream main    # From original

# Fetch from all remotes
git fetch --all
```

**Synchronize with specific branches:**

```bash
# Work on develop branch
git checkout develop
git pull origin develop

# Push feature branch
git checkout feature-branch
git push origin feature-branch
```

**Keep feature branches updated:**

```bash
# While working on feature branch
git checkout feature-branch
git fetch origin
git rebase origin/main  # Update with latest main
```

**Troubleshooting sync issues:**

```bash
# Check remote status
git remote -v
git branch -vv

# Check what would be pulled/pushed
git status

# Force sync (careful!)
git push --force-with-lease origin main
```

**Key Points to Mention:**

- Daily workflow for staying in sync
- Different ways to synchronize
- Handling multiple remotes
- Best practices for clean history
- Common issues and solutions

---

## Advanced Git Operations

### Question 16: What is `git stash` and when would you use it?

**Sample Answer:**
`git stash` temporarily saves changes in your working directory and staging area, allowing you to work on something else and return to these changes later.

**What stash saves:**

- Changes in working directory (modified files)
- Staged changes (staging area)
- Not saved: New untracked files

**Basic usage:**

```bash
# Save current changes
git stash
git stash save "Work in progress on login feature"

# List stashes
git stash list

# Apply most recent stash
git stash pop

# Apply specific stash
git stash apply stash@{2}

# Apply without removing from stash list
git stash apply

# View stash contents
git stash show
git stash show -p  # Show patch
```

**When to use stash:**

- Switch to another branch without committing work-in-progress
- Temporarily save changes to fix a critical bug
- Clear your working directory for testing
- Keep changes while rebasing
- Share changes between machines

**Common scenarios:**

1. **Emergency bug fix:**

   ```bash
   # Working on feature, critical bug found
   git stash save "Login feature in progress"
   git checkout main
   git checkout -b hotfix/critical-bug
   # Fix the bug
   git stash pop  # Return to feature work
   ```

2. **Switch branches:**

   ```bash
   git stash
   git checkout other-branch
   # Do work on other branch
   git checkout original-branch
   git stash pop
   ```

3. **Rebase preparation:**
   ```bash
   git stash
   git rebase main
   git stash pop
   ```

**Advanced stash options:**

```bash
# Include untracked files
git stash -u  # Include untracked
git stash -a  # Include ignored

# Create branch from stash
git stash branch new-feature-from-stash

# Drop a stash
git stash drop stash@{1}

# Clear all stashes
git stash clear
```

**Best practices:**

- Use descriptive stash messages
- Keep stashes temporary
- Don't stash for extended periods
- Clean up old stashes regularly
- Test before applying stashes

**Key Points to Mention:**

- What git stash does
- Common use cases
- Basic and advanced commands
- When stash is appropriate
- Best practices for using stash

### Question 17: Explain what `git rebase` is and the different types of rebase operations.

**Sample Answer:**
Git rebase moves or combines a sequence of commits to a new base commit. It's primarily used to maintain a linear project history.

**Types of rebase operations:**

**1. Regular rebase:**

```bash
# Rebase current branch on top of main
git checkout feature-branch
git rebase main
```

**Effect:** All commits from feature-branch are replayed on top of main, creating a linear history.

**2. Interactive rebase:**

```bash
# Rebase last 3 commits
git rebase -i HEAD~3

# Rebase on specific branch
git rebase -i main
```

**Interactive options:**

- `pick` - Use commit as-is
- `reword` - Use commit, but edit message
- `edit` - Use commit, but stop to amend
- `squash` - Use commit, but meld into previous
- `fixup` - Like squash, discard commit message
- `exec` - Run command (shell command)

**Example interactive rebase:**

```bash
# During interactive rebase, edit file:
pick 1234567 Add login feature
squash 2345678 Update login feature
squash 3456789 Fix login bug
```

**3. Rebase onto different branch:**

```bash
# Rebase feature-branch onto release-branch
git rebase --onto release-branch main feature-branch
```

**4. Rebase with autosquash:**

```bash
# Automatically organize fixup and squash commits
git rebase -i --autosquash main
```

**Benefits of rebase:**

- Cleaner, linear history
- Easier to understand project evolution
- Better for code review
- Eliminates unnecessary merge commits

**When to use rebase:**

- Clean up local history before sharing
- Integrate upstream changes into feature branch
- Combine related commits
- Edit commit messages
- Remove unwanted commits

**When NOT to use rebase:**

- Never rebase public branches that others are using
- Don't rebase shared history
- Avoid on branches others have based work on

**Rebase workflow:**

```bash
# Before creating PR
git checkout feature-branch
git rebase -i main  # Clean up history
git push --force-with-origin feature-branch
```

**Conflict resolution during rebase:**

```bash
# If conflicts occur:
git status
# Resolve conflicts in files
git add resolved-files
git rebase --continue

# Or abort
git rebase --abort
```

**Key Points to Mention:**

- Different types of rebase operations
- Interactive rebase options
- Benefits and use cases
- Safety considerations (public branches)
- Conflict resolution during rebase

### Question 18: What are Git tags and how do you use them?

**Sample Answer:**
Git tags are references that point to specific commits in your repository history. They are commonly used to mark release points, important milestones, or significant versions of your project.

**Types of tags:**

**1. Annotated tags (recommended for releases):**

```bash
git tag -a v1.0.0 -m "Release version 1.0.0"
git tag -a v2.1.0 -m "Add user authentication"
```

- Stored as full objects in Git database
- Include tagger name, email, date
- Have a tagging message
- Can be signed and verified

**2. Lightweight tags:**

```bash
git tag v1.0.0
git tag v1.0.0-beta
```

- Simple pointers to commits
- Don't store additional information
- Quick to create

**Common operations:**

**Create tags:**

```bash
# Annotated tag
git tag -a v1.0.0 -m "Official release 1.0.0"

# Lightweight tag
git tag v1.0.0

# Tag specific commit
git tag -a v0.9.0 <commit-hash> -m "Beta release"

# Tag based on branch
git tag -a v2.0.0 develop -m "Release 2.0 from develop"
```

**View tags:**

```bash
# List all tags
git tag

# List with pattern
git tag -l "v1.*"

# Show tag information
git show v1.0.0

# Show tag with diff
git show v1.0.0 --stat
```

**Push tags to remote:**

```bash
# Push specific tag
git push origin v1.0.0

# Push all tags
git push --tags

# Push tag with force (if already exists)
git push --force origin v1.0.0
```

**Delete tags:**

```bash
# Delete local tag
git tag -d v0.9.0

# Delete remote tag
git push origin --delete v0.9.0

# Delete multiple tags
git push origin --delete v1.0.0 v1.0.1
```

**Use cases:**

- **Release management**: Mark software versions
- **Milestone tracking**: Important project milestones
- **Deployment tags**: Specific deployable versions
- **Rollback points**: Safe points to return to
- **Documentation**: Reference specific points in history

**Best practices:**

- Use semantic versioning (v1.0.0, v1.2.3)
- Always use annotated tags for releases
- Include meaningful tag messages
- Push tags with code
- Document what each tag represents

**Checking out tags:**

```bash
# Check out specific tag (detached HEAD)
git checkout v1.0.0

# Create branch from tag
git checkout -b version-1.0 v1.0.0
```

**Key Points to Mention:**

- What tags are and why they're used
- Difference between annotated and lightweight tags
- How to create, view, and delete tags
- How to push tags to remote
- Best practices for tagging

### Question 19: What is `git cherry-pick` and when would you use it?

**Sample Answer:**
`git cherry-pick` applies the changes introduced by existing commits to your current branch. It's like copying a commit from one branch to another without merging the entire branch.

**Basic usage:**

```bash
# Apply a specific commit to current branch
git cherry-pick <commit-hash>

# Apply multiple commits
git cherry-pick <commit-hash1> <commit-hash2>

# Apply a range of commits
git cherry-pick <commit-hash1>..<commit-hash2>
```

**When to use cherry-pick:**

1. **Backporting fixes**: Apply a bug fix from development to a release branch
2. **Selective feature adoption**: Take specific features from one branch without merging everything
3. **Hotfix application**: Apply urgent fixes to production branches
4. **Selective merging**: When you only want specific commits, not entire branches

**Common scenarios:**

**Scenario 1: Backport a bug fix**

```bash
# You're on release branch
git checkout release/1.0

# Find the fix commit from develop
git log --oneline develop
# Shows: 1234567 Fix critical authentication bug

# Cherry-pick the fix
git cherry-pick 1234567

# Push the fix
git push origin release/1.0
```

**Scenario 2: Apply multiple related commits**

```bash
# Find commits to apply
git log --oneline feature-branch
# Shows:
# 2345678 Implement user profile feature
# 3456789 Add user settings
# 4567890 Update user profile UI

# Apply all three commits
git cherry-pick 2345678 3456789 4567890
```

**Cherry-pick options:**

```bash
# Apply without committing
git cherry-pick --no-commit <commit-hash>

# Edit commit message
git cherry-pick -e <commit-hash>

# Apply with strategy
git cherry-pick -s recursive -X ours <commit-hash>

# Continue after resolving conflicts
git cherry-pick --continue

# Abort cherry-pick
git cherry-pick --abort
```

**Handling conflicts:**

```bash
# If conflicts occur:
git status  # Shows conflicted files

# Resolve conflicts in files
git add resolved-files

# Continue cherry-pick
git cherry-pick --continue

# Or abort
git cherry-pick --abort
```

**Best practices:**

- Only cherry-pick commits that are truly independent
- Document why you're cherry-picking
- Test thoroughly after cherry-picking
- Consider the impact on project history
- Use for backports and hotfixes, not routine merging

**When NOT to use cherry-pick:**

- When you can merge instead (merges preserve history better)
- For complex dependencies between commits
- When the commit has already been merged elsewhere
- As a substitute for proper branching strategy

**Key Points to Mention:**

- What cherry-pick does
- Common use cases
- How to apply single or multiple commits
- Conflict resolution during cherry-pick
- When to use and when to avoid

### Question 20: Explain `git bisect` and how it's used for debugging.

**Sample Answer:**
`git bisect` is a powerful debugging tool that uses binary search to find the commit that introduced a bug. It dramatically reduces the time needed to find problematic commits.

**How it works:**

- You tell Git a "good" commit (no bug) and a "bad" commit (has bug)
- Git checks out the middle commit
- You test and tell Git if it's good or bad
- Git halves the search space each time
- Eventually finds the exact commit that introduced the bug

**Basic usage:**

```bash
# Start bisect
git bisect start

# Mark current commit as bad
git bisect bad

# Mark a known good commit
git bisect good <commit-hash>
# or
git bisect good HEAD~10  # 10 commits ago

# Git will checkout a commit for testing
# Test the current version...
# If bug is present: git bisect bad
# If bug is absent: git bisect good

# Repeat until Git finds the problematic commit

# Exit bisect when done
git bisect reset
```

**Complete example:**

```bash
# Your calculator has started giving wrong results
# You know version 1.0.0 (tag v1.0.0) worked correctly
# Current version has the bug

git bisect start
git bisect bad           # Current commit is bad
git bisect good v1.0.0   # v1.0.0 is good

# Git checks out a middle commit
# Test: python calculator.py
# If bug present: git bisect bad
# If bug absent: git bisect good

# Repeat testing until Git identifies the problematic commit
# Git will output something like:
# 1234567 is the first bad commit

# Exit and return to normal
git bisect reset
```

**Advanced bisect options:**

```bash
# Use script to automate testing
git bisect run npm test

# Use script to check for bug
git bisect run bash -c 'if ./test-program; then exit 0; else exit 1; fi'

# Skip a commit (if it can't be tested)
git bisect skip

# View bisect log
git bisect log

# Replay bisect decisions
git bisect replay <logfile>
```

**Using with automated tests:**

```bash
# Create test script
cat > test_bug.sh << 'EOF'
#!/bin/bash
# Test for the bug
python calculator.py > output.txt
if grep -q "Error: wrong calculation" output.txt; then
    echo "Bug found"
    exit 1  # bad
else
    echo "Bug not found"
    exit 0  # good
fi
EOF

chmod +x test_bug.sh

# Run automated bisect
git bisect start
git bisect bad
git bisect good v1.0.0
git bisect run ./test_bug.sh
```

**Benefits:**

- Dramatically reduces debugging time (O(log n) instead of O(n))
- Systematic approach
- Can be automated with scripts
- Works with any type of bug (logic errors, performance issues, etc.)

**When to use bisect:**

- Finding when a bug was introduced
- Identifying performance regressions
- Locating commit that broke tests
- Debugging unexpected behavior changes

**Best practices:**

- Start with a clear definition of "good" and "bad"
- Create automated tests when possible
- Document findings
- Consider the commit context when you find the bug

**Key Points to Mention:**

- How binary search works in Git bisect
- Basic workflow (start, mark good/bad, test, repeat)
- How to automate with scripts
- Benefits for debugging
- When to use this tool

---

## Conflict Resolution

### Question 21: How do you handle complex merge conflicts with multiple files?

**Sample Answer:**
Complex merge conflicts occur when multiple files have conflicting changes. A systematic approach is essential:

**Step 1: Identify all conflicts**

```bash
# Start merge
git merge feature/complex-feature

# Check status
git status  # Shows all conflicted files

# List only conflicted files
git status --porcelain | grep "^UU"
```

**Step 2: Create a resolution plan**

```bash
# Create a plan document
cat > MERGE_PLAN.md << 'EOF'
# Merge Conflict Resolution Plan

## High Priority (Core functionality)
1. src/main.py - Application entry point
2. src/models.py - Data models
3. config.py - Configuration

## Medium Priority (Features)
4. src/auth.py - Authentication
5. src/api.py - API endpoints
6. templates/base.html - Base template

## Low Priority (Styling)
7. static/css/main.css - Main stylesheet
8. static/js/app.js - JavaScript
EOF
```

**Step 3: Resolve conflicts systematically**

**For each conflicted file:**

```bash
# Open file and look for conflict markers
# <<<<<<< HEAD
# Current branch content
# =======
# Incoming branch content
# >>>>>>> feature-branch

# Choose the correct version or combine:
# - Read both versions carefully
# - Understand what each change does
# - Merge them logically
# - Remove conflict markers
# - Test the result
```

**Step 4: Example resolution process**

**File: src/main.py**

```python
# After resolving:
import os
import logging
from config import settings
from src.auth import authenticate_user
from src.models import User, Project

def main():
    # Combined functionality from both branches
    if settings.DEBUG:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Start application
    authenticate_user()
    load_projects()

# Remove conflict markers after editing
# <<<<<<< HEAD
# =====
# >>>>>>> feature-branch
```

**Step 5: Stage and test as you go**

```bash
# After resolving each file
git add src/main.py
python test_main.py  # Run tests

# Continue with next file
git add src/models.py
python test_models.py

# If something goes wrong
git reset --hard HEAD  # Start over
```

**Step 6: Complete the merge**

```bash
# After resolving all conflicts
git add .
git commit -m "Resolve merge conflicts: combine feature enhancements"

# Verify the merge
git log --oneline --graph --all
```

**Tools for complex conflicts:**

```bash
# Use merge tool
git mergetool

# Compare specific files
git diff HEAD -- path/to/file
git diff feature-branch -- path/to/file

# View three-way diff
git diff --no-index /dev/null path/to/file  # Not directly helpful
git show :1:path/to/file  # Base version
git show :2:path/to/file  # Current branch
git show :3:path/to/file  # Incoming branch
```

**Best practices:**

1. **Don't panic** - Conflicts are normal
2. **Create a plan** - Prioritize by importance
3. **Resolve systematically** - One file at a time
4. **Test frequently** - Ensure each resolution works
5. **Communicate with team** - Discuss complex decisions
6. **Document decisions** - Why you chose specific resolutions
7. **Use tools** - Merge tools can help visualize conflicts

**Preventing complex conflicts:**

- Keep branches short-lived
- Merge frequently
- Communicate with team about major changes
- Use feature flags for large changes
- Implement incremental merging

**Key Points to Mention:**

- Systematic approach to multiple conflicts
- Prioritization strategy
- Step-by-step resolution process
- Tools and techniques available
- Best practices and prevention

### Question 22: What strategies do you use to prevent merge conflicts?

**Sample Answer:**
Preventing merge conflicts is more efficient than resolving them. Here are key strategies:

**1. Keep Branches Short-Lived**

```bash
# Good: Short feature branches
git checkout -b feature/user-login  # Created today
# Complete work in 1-2 days
git checkout main
git merge feature/user-login
git branch -d feature/user-login

# Bad: Long-lived branches
git checkout -b feature/major-refactor  # Created 3 weeks ago
# Still working after many main updates
```

**2. Merge Frequently**

```bash
# Before starting work
git checkout feature-branch
git merge main  # Get latest changes

# During development
git merge main  # Every few days

# Before creating PR
git merge main  # Final integration
```

**3. Work in Small Increments**

```bash
# Good: Small, focused commits
git commit -m "feat: add user login form"
git commit -m "fix: resolve form validation"
git commit -m "feat: add login API endpoint"
git commit -m "test: add login form tests"

# Bad: Large, monolithic commits
git commit -m "Add entire login system"  # Mixed concerns
```

**4. Communicate with Team**

- **Slack/Teams**: "I'm working on refactoring the user module"
- **Jira/Trello**: Claim related tickets
- **GitHub**: Assign related issues to yourself
- **Standup meetings**: Mention what you're working on

**5. Use Feature Flags**

```javascript
// Instead of big changes, use flags
if (featureFlags.newUI) {
  return renderNewUI();
} else {
  return renderOldUI();
}
```

**6. Coordinate File Changes**

```bash
# If you need to modify the same file
# Coordinate with team members:
# Sarah: "I'm updating models.py to add user preferences"
# Alex: "I'll wait for your PR before modifying models.py"
```

**7. Follow Consistent Code Styles**

```bash
# Pre-commit hook to check formatting
#!/bin/sh
npx prettier --write .
git add .
```

**8. Use Branch Protection Rules**

```bash
# On GitHub, set up branch protection:
# - Require pull request reviews
# - Require status checks to pass
# - Require branches to be up to date
```

**9. Implement Code Ownership**

```yaml
# CODEOWNERS file
# Global owners
* @team-leads

# Specific directories
src/models/ @database-team
src/auth/ @security-team
frontend/ @frontend-team
```

**10. Use Automated Conflict Detection**

```bash
# Pre-merge check
#!/bin/bash
# Check if merge will have conflicts
if ! git merge --no-commit --no-ff feature-branch; then
    echo "Merge would create conflicts. Aborting."
    exit 1
fi
```

**Branch Strategy to Minimize Conflicts:**

**GitHub Flow (simple):**

```bash
# 1. Always start from latest main
git checkout main
git pull origin main

# 2. Create small feature branch
git checkout -b feature/small-change

# 3. Complete quickly (1-3 days)
git add .
git commit -m "Add small feature"

# 4. Create PR and merge
git push origin feature/small-change
# Create PR on GitHub, get reviewed and merged
```

**Best practices summary:**

- Small, frequent merges
- Short-lived branches
- Good communication
- Consistent coding standards
- Automated tools
- Clear ownership

**Key Points to Mention:**

- Prevention is better than resolution
- Short-lived, focused branches
- Frequent synchronization
- Team communication
- Technical tools and practices
- Continuous integration

---

## Workflows and Best Practices

### Question 23: Compare and contrast different Git workflows (GitHub Flow, Git Flow, GitLab Flow).

**Sample Answer:**
Different Git workflows suit different project types and team sizes. Here's a comparison:

**GitHub Flow**

**Structure:**

- `main` branch (always deployable)
- Feature branches from main
- Pull requests for code review
- Immediate deployment after merge

**Workflow:**

```bash
# 1. Create branch from main
git checkout main
git pull origin main
git checkout -b feature/new-feature

# 2. Make changes and commit
git add .
git commit -m "Add new feature"

# 3. Push and create PR
git push -u origin feature/new-feature
# Create PR on GitHub

# 4. After review, merge to main
git checkout main
git pull origin main

# 5. Deploy immediately
```

**When to use:**

- Web applications with frequent deployments
- Teams familiar with GitHub
- Continuous deployment pipelines
- Smaller, agile teams

**Benefits:**

- Simple and easy to understand
- Great for web development
- Forces good practices (code review)
- Fast iteration

**Limitations:**

- No release management
- No long-term maintenance branches
- Not suitable for multiple environments

---

**Git Flow**

**Structure:**

- `main` branch (production)
- `develop` branch (integration)
- Feature branches
- Release branches
- Hotfix branches

**Workflow:**

```bash
# 1. Feature development
git checkout develop
git checkout -b feature/new-feature
# Work and commit
git checkout develop
git merge feature/new-feature

# 2. Release preparation
git checkout -b release/v1.1.0
# Final testing and version bumps
git checkout main
git merge release/v1.1.0
git tag -a v1.1.0 -m "Release v1.1.0"
git checkout develop
git merge release/v1.1.0

# 3. Hotfix (critical bug in production)
git checkout -b hotfix/critical-bug main
# Fix and test
git checkout main
git merge hotfix/critical-bug
git tag -a v1.1.1 -m "Hotfix v1.1.1"
git checkout develop
git merge hotfix/critical-bug
```

**When to use:**

- Complex projects with releases
- Projects requiring maintenance versions
- Teams with established processes
- Software with specific version requirements

**Benefits:**

- Clear release management
- Separate development and production
- Good for maintenance and hotfixes
- Supports multiple versions

**Limitations:**

- More complex to understand
- More branches to manage
- Can be overkill for simple projects
- Requires discipline

---

**GitLab Flow**

**Structure:**

- `main` branch
- Environment-specific branches (staging, production)
- Feature branches
- Automatic deployments to staging

**Workflow:**

```bash
# 1. Develop on feature branch
git checkout -b feature/api-improvement
# Work and commit
git push origin feature/api-improvement

# 2. Merge to main
# After PR review and approval
git checkout main
git merge feature/api-improvement

# 3. Deploy to staging (automatic)
# GitLab CI/CD deploys to staging environment

# 4. After testing, deploy to production
git checkout production
git merge main
```

**When to use:**

- Projects with multiple environments
- CI/CD automation
- Regulated industries
- Projects requiring testing environments

**Benefits:**

- Environment-based branch strategy
- Good CI/CD integration
- Clear promotion process
- Production stability

**Limitations:**

- More complex than GitHub Flow
- Requires CI/CD setup
- Multiple production branches to manage

---

**Comparison Table:**

| Aspect             | GitHub Flow | Git Flow         | GitLab Flow        |
| ------------------ | ----------- | ---------------- | ------------------ |
| Complexity         | Low         | High             | Medium             |
| Branches           | 2-3         | 5-6              | 4-5                |
| Deployment         | Immediate   | Scheduled        | Environment-based  |
| Best for           | Web apps    | Complex software | Multi-env projects |
| Learning curve     | Easy        | Steep            | Medium             |
| Release management | None        | Excellent        | Good               |
| Hotfix handling    | Manual      | Structured       | Environment-based  |

**Choosing the right workflow:**

**Use GitHub Flow when:**

- Building web applications
- Deploying frequently (daily/continuous)
- Team is new to Git workflows
- You want simplicity

**Use Git Flow when:**

- Managing software versions
- Need release management
- Working on complex features
- Supporting multiple versions
- Have established processes

**Use GitLab Flow when:**

- Multiple deployment environments
- Regulated industry (testing required)
- Strong CI/CD needs
- Need production stability

**Key Points to Mention:**

- Each workflow's structure and process
- When to use each workflow
- Benefits and limitations of each
- Real-world examples
- How to transition between workflows

### Question 24: What are some best practices for writing commit messages?

**Sample Answer:**
Good commit messages are crucial for project maintainability and team collaboration. Here are best practices:

**Structure of a Good Commit Message:**

```
type(scope): subject

body (optional)

footer (optional)
```

**Example:**

```
feat(auth): add JWT token validation

Implement JWT-based authentication for API endpoints.
- Add token validation middleware
- Update user session management
- Include token expiration handling

Closes #123
```

**1. Use the Imperative Mood**

```bash
# Good
git commit -m "fix: resolve memory leak in user session"

# Bad
git commit -m "fixed memory leak in user session"
git commit -m "fixes memory leak in user session"
```

**2. Use Meaningful Types**

```bash
# feat: A new feature
git commit -m "feat: add user profile management"

# fix: A bug fix
git commit -m "fix: resolve null pointer exception in login"

# docs: Documentation changes
git commit -m "docs: update API documentation"

# style: Code style changes (formatting, etc.)
git commit -m "style: format code with prettier"

# refactor: Code refactoring
git commit -m "refactor: simplify user validation logic"

# test: Adding or updating tests
git commit -m "test: add unit tests for user service"

# chore: Build process or auxiliary tool changes
git commit -m "chore: update dependencies to latest versions"

# perf: Performance improvements
git commit -m "perf: optimize database queries"

# ci: CI/CD configuration changes
git commit -m "ci: add GitHub Actions workflow"

# build: Build system changes
git commit -m "build: update webpack configuration"
```

**3. Keep Subject Line Short (50 characters max)**

```bash
# Good
git commit -m "feat: add user authentication system"

# Bad - too long
git commit -m "feat: add comprehensive user authentication system with JWT tokens and session management"
```

**4. Use the Body to Explain What and Why, Not How**

```bash
git commit -m "fix: resolve race condition in payment processing

The payment system was processing duplicate transactions due to
a race condition when multiple requests hit the same endpoint
simultaneously.

This commit:
- Adds transaction locking mechanism
- Implements idempotency keys
- Updates error handling to prevent duplicate processing

Fixes #456
```

**5. Use Footer for References**

```bash
# Reference issues
git commit -m "feat: add dark mode toggle

Implements user preference for dark mode
across all application pages.

Closes #789
Refs #456

# Breaking changes
git commit -m "feat: remove deprecated API endpoints

BREAKING CHANGE: Removed /api/v1/users endpoint.
Use /api/v2/users instead.

See migration guide: docs/migration-v2.md
Closes #123"
```

**6. Avoid Generic Messages**

```bash
# Bad
git commit -m "fix stuff"
git commit -m "update"
git commit -m "asdfasdf"
git commit -m "changes"

# Good
git commit -m "fix: resolve incorrect calculation in tax module"
git commit -m "docs: add installation instructions"
```

**7. Test Your Commit Messages**

```bash
# Review your commit message
git log -1  # Shows your last commit

# If you're not satisfied
git commit --amend
# Edit the message and save
```

**8. Use Commit Message Templates**

```bash
# Add to .gitmessage
cat > ~/.gitmessage << 'EOF'
# 50-character subject line
#
# 72-character wrapped body. This should answer:
# * What is the motivation for the change?
# * How does it address the issue?
# * Are there side effects or other consequences?
#
# Type: feat, fix, docs, style, refactor, test, chore
# Scope: (optional) The module affected
# Closes: #issue-number (if applicable)
EOF

git config --global commit.template ~/.gitmessage
```

**Benefits of Good Commit Messages:**

- **Easier code review**: Reviewers understand context
- **Better debugging**: Can find related changes quickly
- **Automated changelogs**: Tools can generate release notes
- **Team collaboration**: Everyone understands project history
- **Professional presentation**: Shows discipline and care

**Commit Message Checklist:**

- [ ] Uses imperative mood
- [ ] 50 characters or less for subject
- [ ] Provides clear context
- [ ] Explains what changed and why
- [ ] Uses appropriate type
- [ ] References issues when relevant
- [ ] Is free of typos and grammar errors

**Key Points to Mention:**

- Structure of commit messages
- Importance of imperative mood
- Different commit types
- Subject line limitations
- Body and footer usage
- Tools and templates available

### Question 25: How do you handle version control in a large team?

**Sample Answer:**
Managing version control in large teams requires clear processes, tools, and discipline:

**1. Establish Clear Branching Strategy**

**Choose appropriate workflow:**

```bash
# GitHub Flow for web teams
# Git Flow for complex projects
# Custom workflow based on team needs
```

**2. Set Up Branch Protection Rules**

```bash
# On GitHub:
# - Require pull request reviews
# - Require status checks to pass
# - Require branches to be up to date
# - Restrict pushes to main branch
# - Require review from code owners
```

**3. Implement Code Ownership**

```yaml
# CODEOWNERS file
# Global
* @team-leads

# Frontend
frontend/ @frontend-team
*.js @frontend-team
*.css @frontend-team

# Backend
backend/ @backend-team
*.py @backend-team
*.go @backend-team

# Infrastructure
deploy/ @devops-team
*.yml @devops-team
Dockerfile @devops-team
```

**4. Define Code Review Process**

```bash
# PR requirements:
# - At least 1 approval from team member
# - At least 1 approval from code owner
# - All CI/CD checks must pass
# - No merge conflicts
# - Tests must pass
# - Documentation updated
```

**5. Use CI/CD Pipeline Integration**

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: npm test
      - name: Run linting
        run: npm run lint
      - name: Security scan
        run: npm audit
```

**6. Communication Protocols**

```bash
# Daily standup updates:
# "Working on feature/login, will create PR by EOD"

# Slack notifications:
# Set up GitHub integration for:
# - PR created
# - PR approved
# - PR merged
# - CI/CD status updates
```

**7. Regular Synchronization**

```bash
# Start of day
git checkout main
git pull origin main

# Before creating PR
git checkout feature/my-feature
git rebase main  # Keep history clean
git push -u origin feature/my-feature

# After PR merge
git checkout main
git pull origin main
git branch -d feature/my-feature
```

**8. Conflict Prevention Strategies**

```bash
# Small, frequent merges
# Clear task assignment
# Regular team communication
# Automated conflict detection
```

**9. Documentation and Onboarding**

```bash
# Create CONTRIBUTING.md:
# - Branching strategy explanation
# - Code review process
# - Commit message conventions
# - Testing requirements
# - Deployment process
```

**Example CONTRIBUTING.md:**

```markdown
## Branching Strategy

- Create feature branches from main
- Use descriptive branch names
- Keep branches short-lived (max 1 week)
- Delete branches after merge

## Pull Request Process

- All PRs require 2 approvals
- Include tests and documentation
- Link to related issues
- Squash commits on merge

## Commit Messages

- Follow conventional commits
- 50 character subject line
- Explain what and why, not how
- Reference issues (#123)
```

**10. Tools and Automation**

```bash
# Pre-commit hooks
echo 'npx lint-staged' > .git/hooks/pre-commit

# Automated testing
# Code quality checks
# Security scanning
# Dependency updates
```

**11. Team Size Considerations**

**Small teams (5-10 people):**

- GitHub Flow
- Less strict rules
- Direct communication

**Medium teams (10-50 people):**

- Mix of Git Flow and GitHub Flow
- Code owners
- CI/CD integration

**Large teams (50+ people):**

- Strict Git Flow
- Multiple approvers
- Automated processes
- Clear ownership boundaries

**12. Performance Monitoring**

```bash
# Track metrics:
# - PR review time
# - Merge frequency
# - Build success rate
# - Deployment frequency
```

**13. Handling Emergencies**

```bash
# Hotfix process:
git checkout -b hotfix/critical-issue main
# Make minimal fix
# Fast-track review
# Deploy immediately
# Merge back to develop
```

**Best practices summary:**

- Clear strategy and documentation
- Automated processes
- Good communication
- Code ownership
- Regular synchronization
- Performance monitoring
- Emergency procedures

**Key Points to Mention:**

- Branching strategy selection
- Code ownership and review
- Automation and CI/CD
- Communication protocols
- Documentation importance
- Scalability considerations

---

## Troubleshooting

### Question 26: What do you do when you accidentally committed to the wrong branch?

**Sample Answer:**
Accidentally committing to the wrong branch is common. Here's how to fix it:

**Scenario 1: Realize immediately (before pushing)**

```bash
# You're on wrong-branch and just committed
# Option 1: Move the commit to correct branch
git branch correct-branch  # Create new branch at current position
git reset --hard HEAD~1   # Remove commit from wrong branch
git checkout correct-branch  # Move to correct branch
```

**Scenario 2: Wrong branch but want to keep the commit**

```bash
# After committing to wrong-branch
# Create branch with the commit
git branch feature-new  # Creates branch at current commit

# Switch to correct branch
git checkout main

# Your commit is now on feature-new branch
# wrong-branch no longer has it
```

**Scenario 3: Already pushed to remote**

```bash
# Fix locally first
git branch feature-new  # Create branch with commit
git reset --hard HEAD~1  # Remove from wrong branch

# Force push to fix remote (careful!)
git push --force-with-origin wrong-branch

# Or safer: communicate with team
git push --force-with-origin wrong-branch
# Notify team about forced push
```

**Scenario 4: Complex case with multiple commits**

```bash
# Accidentally made 3 commits on wrong branch
git branch feature-new  # Create branch at current position
git reset --hard HEAD~3  # Remove 3 commits from wrong branch

# Now feature-new has all 3 commits
```

**Scenario 5: Use cherry-pick to move commits**

```bash
# If you want to keep wrong-branch as-is and move to new branch
git checkout -b feature-correct
git cherry-pick <commit-hash>  # Move specific commit
# or
git cherry-pick <hash1> <hash2> <hash3>  # Move multiple commits
```

**Prevention strategies:**

```bash
# Check current branch before committing
git status  # Shows current branch
git branch  # Lists all branches

# Use better workflow
git checkout main
git pull origin main
git checkout -b feature/new-work
# Do work...
git add .
git commit -m "Add new feature"
git push -u origin feature/new-work
```

**Using git reflog for recovery:**

```bash
# If things go wrong
git reflog  # Shows all your moves
# Find the commit you want
git checkout <commit-hash>
git checkout -b rescue-branch
```

**Best practices:**

- **Double-check** which branch you're on before committing
- **Use `git status`** frequently
- **Don't panic** - Git is very forgiving
- **Communicate** with team if you force push
- **Practice** these scenarios beforehand

**Key Points to Mention:**

- Different scenarios for wrong branch commits
- Solutions for each scenario
- When to use each method
- How to prevent this issue
- Using reflog for recovery

### Question 27: How do you recover from a hard reset that you didn't intend to do?

**Sample Answer:**
A hard reset (`git reset --hard`) discards commits and changes. Recovery is possible using Git's safety mechanisms:

**Immediate Recovery Steps**

**1. Use git reflog to find lost commits**

```bash
# Reflog shows all movements of HEAD
git reflog
git reflog --date=iso  # Shows dates

# Example output:
# abcd123 HEAD@{0}: reset: moving to HEAD~1
# efgh456 HEAD@{1}: commit: Add important feature
# ijkl789 HEAD@{2}: commit: Fix critical bug
# mnop012 HEAD@{3}: merge feature/login: Merge made by recursive strategy
```

**2. Identify the commit to restore**
Look for the commit hash you want to recover (e.g., `efgh456`).

**3. Restore the commit**

```bash
# Option 1: Create branch from lost commit
git checkout efgh456
git checkout -b recovered-feature

# Option 2: Reset current branch back
git checkout main
git reset --hard efgh456

# Option 3: Use git show to see lost commit
git show efgh456 > lost-commit.patch
```

**Advanced Recovery Techniques**

**Using fsck to find dangling commits:**

```bash
# Check for unreachable objects
git fsck --full
git fsck --unreachable

# Find commits in unreachable objects
git fsck --unreachable | grep commit
```

**Using git log with different references:**

```bash
# Check all references
git log --all --oneline
git log --all --graph --oneline

# Check specific reflog
git reflog show branch-name
git reflog show HEAD@{2.days.ago}
```

**Recovery in different scenarios:**

**Scenario 1: Reset to older commit**

```bash
# Accidentally did:
git reset --hard HEAD~5

# Recovery:
git reflog
git reset --hard HEAD@{1}
# or
git reset --hard <commit-hash-from-reflog>
```

**Scenario 2: Reset to completely wrong commit**

```bash
# Accidentally did:
git reset --hard wrong-commit-hash

# Recovery:
git reflog
git reset --hard HEAD@{1}
```

**Scenario 3: Reset merged branch**

```bash
# After merge, did:
git reset --hard HEAD~1

# Recovery:
git reflog
# Find the merge commit
git reset --hard <merge-commit-hash>
```

**Prevention Strategies**

**1. Use safer reset options**

```bash
# Safer options (keep changes):
git reset --soft HEAD~1  # Keep changes staged
git reset --mixed HEAD~1  # Keep changes in working directory

# Only use --hard when absolutely sure
git reset --hard HEAD~1
```

**2. Always check before hard reset**

```bash
# Before reset, save current state
git branch backup-branch
git reflog > reflog-backup.txt

# If something goes wrong, restore
git reset --hard backup-branch
```

**3. Use tags for important points**

```bash
# Before risky operations
git tag backup-before-refactor
git reset --hard

# To restore
git reset --hard backup-before-refactor
git tag -d backup-before-refactor
```

**Understanding the Recovery Process**

**What happens in a hard reset:**

- Branch pointer moves to specified commit
- Working directory updated to match
- Changes are discarded
- Commits become "unreachable"

**Why recovery works:**

- Git doesn't immediately delete unreachable commits
- Reflog keeps track of all movements
- Objects stay in database until garbage collected
- Window for recovery: typically 30-90 days

**Time limits for recovery:**

```bash
# Garbage collection runs automatically
# Prunes unreachable objects older than 30 days
# Can configure: git config gc.reflogExpire 30.days
```

**Commands to check for recoverable data:**

```bash
# Show all objects
git fsck --full --cache

# Check dangling blobs
git fsck --unreachable | grep blob

# View reflog entries older than specific time
git reflog expire --expire=30.days --all
```

**Key Points to Mention:**

- Using git reflog for recovery
- Understanding why recovery works
- Different recovery scenarios
- Prevention strategies
- Time limits for recovery

### Question 28: What is the difference between revert, reset, and restore in Git?

**Sample Answer:**
These three commands serve different purposes for undoing changes in Git:

**`git revert` - Safe undo for public history**

**What it does:**

- Creates a new commit that undoes the changes of a previous commit
- Preserves the original commit in history
- Safe to use on shared/public branches
- Adds a new commit to the history

**When to use:**

- Undo a commit that has been shared/pushed
- Fix bugs in production
- Maintain complete history

**Example:**

```bash
# Revert the last commit
git revert HEAD

# Revert specific commit
git revert 1234567

# Revert range of commits
git revert 1234567..7890abc

# Revert without committing (for manual review)
git revert --no-commit HEAD
```

**Result:**

```
A -- B -- C -- D
           |
           +-- Revert D (undoes D's changes)
```

---

**`git reset` - Move branch pointer**

**What it does:**

- Moves the current branch pointer to a different commit
- Can modify staging area and working directory
- Rewrites history (dangerous on shared branches)
- Three modes: soft, mixed, hard

**Modes:**

**Soft reset (`--soft`):**

```bash
git reset --soft HEAD~1
```

- Moves branch pointer
- Keeps changes staged
- Working directory unchanged

````

**Mixed reset (`--mixed` - default):**
```bash
git reset HEAD~1
git reset --mixed HEAD~1
````

- Moves branch pointer
- Unstages changes
- Working directory unchanged

**Hard reset (`--hard`):**

```bash
git reset --hard HEAD~1
```

- Moves branch pointer
- Discards all changes
- Working directory matches new commit

**When to use:**

- Undo local commits before sharing
- Change branch to different point
- Clean up local history

**Example:**

```
A -- B -- C -- D (HEAD)
         |
         +-- Reset to C
```

---

**`git restore` - Working directory changes**

**What it does:**

- Restores working directory files
- Does not affect history
- Unstages changes
- Can restore from staging or previous commits

**When to use:**

- Discard changes in working directory
- Unstage files
- Restore specific files to previous state

**Examples:**

**Restore working directory:**

```bash
# Discard changes in working directory
git restore filename.txt

# Restore all changed files
git restore .
```

**Unstage files:**

```bash
# Unstage specific file
git restore --staged filename.txt

# Unstage all staged files
git restore --staged .
```

**Restore from specific commit:**

```bash
# Restore file to state from last commit
git restore --source HEAD~1 filename.txt

# Restore file to state from specific commit
git restore --source 1234567 filename.txt
```

---

**Comparison Table:**

| Command       | History          | Working Directory | Staging Area | Safe on Shared Branch |
| ------------- | ---------------- | ----------------- | ------------ | --------------------- |
| `git revert`  | Adds new commit  | Not modified      | Not modified | ✅ Yes                |
| `git reset`   | Rewrites history | Can modify        | Can modify   | ❌ No                 |
| `git restore` | No change        | Restores          | Can unstage  | ✅ Yes                |

**Visual Comparison:**

**Original:** `A -- B -- C -- D (HEAD)`

**After `git revert D`:** `A -- B -- C -- D -- Revert(D)`

**After `git reset --hard C`:** `A -- B -- C (HEAD)`

**After `git restore .`:** `A -- B -- C (HEAD) - D's changes discarded`

**Decision Tree:**

**To undo a commit that hasn't been shared:**

- Use `git reset` for local changes
- Consider `git restore` for specific files

**To undo a commit that has been shared:**

- Always use `git revert`
- Never use `git reset` or `git restore` on shared history

**To discard working directory changes:**

- Use `git restore`
- If staged, use `git restore --staged` first

**Key Points to Mention:**

- Each command's purpose and behavior
- When to use each command
- Safety considerations for shared branches
- Visual examples of effects
- Decision tree for choosing the right command

---

## Scenario-Based Questions

### Question 29: Walk me through how you would handle a merge conflict in a real project scenario.

**Sample Answer:**
Here's how I'd handle a merge conflict in a real project:

**Initial Situation:**
I'm working on a team project (e-commerce platform) with 5 developers. We have:

- `main` branch (production-ready code)
- Team members working on different features
- Automated CI/CD pipeline running tests

**Step 1: Detection**

```bash
# I try to merge my feature branch
git checkout main
git pull origin main
git merge feature/shopping-cart

# Git reports conflict:
# Auto-merging src/components/Cart.js
# CONFLICT (content): Merge conflict in src/components/Cart.js
# Automatic merge failed; fix conflicts and then commit the result.
```

**Step 2: Assessment**

```bash
# Check what files are conflicted
git status

# Create a conflict resolution plan
cat > MERGE_CONFLICT_PLAN.md << 'EOF'
# Merge Conflict Resolution Plan

## Conflicted Files:
1. src/components/Cart.js - Shopping cart component
2. src/utils/pricing.js - Price calculation logic
3. package.json - Dependencies (likely npm update)

## Priority:
1. Cart.js (HIGH) - Core functionality
2. Pricing.js (HIGH) - Business logic
3. Package.json (MEDIUM) - Dependency updates

## Approach:
- Review both versions
- Understand the changes
- Merge logically
- Test thoroughly
EOF
```

**Step 3: Examine the Conflicts**

```bash
# Look at the conflicted file
git diff HEAD -- src/components/Cart.js

# Open Cart.js in editor and see:
# <<<<<<< HEAD
# const [items, setItems] = useState([]);
# const [total, setTotal] = useState(0);
# =======
# const [items, setItems] = useState([]);
# const [discount, setDiscount] = useState(0);
# >>>>>>> feature/shopping-cart
```

**Step 4: Understanding the Changes**

**Current branch (main) changes:**

- Removed `useState` for total (moved to pricing utility)
- Added item validation

**Incoming branch (shopping-cart) changes:**

- Added discount functionality
- Kept local state for total
- Added cart persistence

**Step 5: Strategic Resolution**

```javascript
// After understanding both changes, I merge them:

import { calculateTotal, applyDiscount } from "../utils/pricing";
import { useEffect, useState } from "react";

const [items, setItems] = useState([]);
const [discount, setDiscount] = useState(0); // From shopping-cart
const [total, setTotal] = useState(0); // From main

// Combined validation from main
useEffect(() => {
  // Validate items
  if (!Array.isArray(items)) {
    console.error("Items must be an array");
    return;
  }

  // Calculate total using pricing utility
  const calculatedTotal = calculateTotal(items);
  const finalTotal = applyDiscount(calculatedTotal, discount);
  setTotal(finalTotal);
}, [items, discount]);

// Add from main: persist to localStorage
useEffect(() => {
  localStorage.setItem("cart", JSON.stringify(items));
}, [items]);

// Add from shopping-cart: discount application
const applyDiscountCode = (code) => {
  const discountPercent = getDiscountPercent(code);
  setDiscount(discountPercent);
};

// Remove conflict markers
// <<<<<<< HEAD
// =======
// >>>>>>> feature/shopping-cart
```

**Step 6: Resolve Systematically**

```bash
# For each conflicted file:

# 1. Cart.js - Resolved above
git add src/components/Cart.js
npm test -- --testNamePattern="Cart"  # Run cart tests

# 2. Pricing.js - Similar process
# Check what changed in both branches
# Understand the conflict
# Merge logically
# Test the changes
git add src/utils/pricing.js
npm test -- --testNamePattern="Pricing"

# 3. Package.json - Often easier
# Choose one version or merge dependency changes
git add package.json
npm install  # Update dependencies
```

**Step 7: Testing**

```bash
# Run all tests
npm test

# Check linting
npm run lint

# Run integration tests
npm run test:integration

# Manual testing
npm start
# Test the shopping cart functionality manually
```

**Step 8: Commit the Resolution**

```bash
# Check status
git status

# All conflicts resolved
git add .

# Commit with clear message
git commit -m "Merge feature/shopping-cart with conflict resolution

Conflicts resolved:
- src/components/Cart.js: Combined discount and pricing logic
- src/utils/pricing.js: Merged tax calculation changes
- package.json: Updated dependencies from both branches

Testing:
- All unit tests passing
- Integration tests passing
- Manual cart functionality verified
- No breaking changes detected"
```

**Step 9: Push and Notify Team**

```bash
# Push the resolved merge
git push origin main

# Notify team on Slack
# "Merged feature/shopping-cart with minor conflicts.
# All tests passing. Please pull latest main."
```

**Step 10: Document Lessons Learned**

```bash
# Update team documentation
cat >> TEAM_GUIDE.md << 'EOF'

## Merge Conflict Prevention
- Small, frequent merges
- Clear task assignment
- Regular communication about file changes
- Use feature flags for large UI changes
EOF
```

**Key Decisions Made:**

1. **Prioritized by importance**: Cart component first, then pricing, then dependencies
2. **Combined functionality**: Instead of choosing one version, merged the useful parts of both
3. **Tested thoroughly**: Ran tests after each file resolution
4. **Documented the process**: Clear commit message explaining what was done
5. **Communicated with team**: Notified about the merge and potential impacts

**Prevention for Future:**

- Merge main into feature branches daily
- Break large changes into smaller PRs
- Communicate about file ownership
- Use merge tools for complex conflicts

**Key Points to Mention:**

- Systematic approach to conflict resolution
- Prioritization strategy
- Understanding both sides of the conflict
- Testing at each step
- Clear communication and documentation

### Question 30: Describe how you would set up a Git workflow for a team of developers working on a mobile app.

**Sample Answer:**
Here's how I'd design a Git workflow for a mobile app team:

**Team Structure:**

- 6 developers (2 iOS, 2 Android, 1 backend, 1 UI/UX)
- 1 tech lead
- Mobile app with React Native
- Backend API service
- Weekly releases
- QA testing environment

**Chosen Workflow: GitHub Flow + Release Management**

**Branch Structure:**

```bash
# Protected branches
main              # Production code
develop           # Integration branch
release/*         # Release branches
hotfix/*          # Emergency fixes

# Feature branches
feature/*         # New features
bugfix/*          # Bug fixes
refactor/*        # Code improvements
```

**Daily Workflow**

**1. Start of Day**

```bash
# Each developer
git checkout develop
git pull origin develop
git checkout -b feature/mobile-notifications

# Or for small fixes
git checkout -b bugfix/login-crash
```

**2. Development Cycle**

```bash
# Make changes
git add .
git commit -m "feat(mobile): add push notification system

Implement Firebase Cloud Messaging for:
- Order status updates
- Promotional notifications
- Security alerts

Refs: #123"

# Push feature branch
git push -u origin feature/mobile-notifications
```

**3. Before Creating PR**

```bash
# Update with latest develop
git fetch origin
git rebase origin/develop

# Run tests
npm test
npm run lint
npm run build:android
npm run build:ios

# If tests pass, push updates
git push --force-with-origin feature/mobile-notifications
```

**4. Pull Request Process**

```bash
# Create PR on GitHub
# Base: develop <- Compare: feature/mobile-notifications

# PR Template:
# ## Changes Made
# - [x] iOS implementation
# - [x] Android implementation
# - [x] Backend API integration
# - [x] Unit tests
# - [x] Integration tests
# - [x] Documentation updated
#
# ## Testing
# - [x] Manual testing on iOS device
# - [x] Manual testing on Android device
# - [x] Automated tests passing
# - [x] Performance impact assessed
#
# ## Screenshots
# [Add before/after screenshots]
#
# ## Checklist
# - [x] Code review from tech lead
# - [x] Code review from relevant platform developer
# - [x] All tests passing
# - [x] No linting errors
```

**5. Code Review Requirements**

```bash
# Minimum approvals:
# - 1 approval from tech lead
# - 1 approval from platform expert (iOS/Android)
# - 1 approval from backend (if API changes)
# - 1 approval from QA (if user-facing)

# Automated checks must pass:
# - All unit tests
# - Linting
# - Build for iOS
# - Build for Android
# - Security scan
```

**6. Merge to Develop**

```bash
# After approval, merge via GitHub UI
# Use "Squash and merge" for clean history

# Locally, clean up
git checkout develop
git pull origin develop
git branch -d feature/mobile-notifications
```

**7. Release Process (Weekly)**

```bash
# Create release branch
git checkout develop
git checkout -b release/v2.1.0

# Final testing on release branch
# Update version numbers
# Update changelog
# Deploy to QA environment

# After QA approval
git checkout main
git merge release/v2.1.0
git tag -a v2.1.0 -m "Release version 2.1.0"
git push origin main --tags

# Deploy to app stores
# iOS: Submit to App Store
# Android: Deploy to Play Store

# Merge back to develop
git checkout develop
git merge release/v2.1.0
```

**Environment-Specific Branches**

```bash
# For different environments
main              # Production
staging           # Pre-production testing
develop           # Integration
feature/*         # Individual features
```

**Hotfix Process (Emergency Bugs)**

```bash
# Critical bug in production
git checkout main
git checkout -b hotfix/fix-app-crash

# Minimal fix
git add .
git commit -m "hotfix: fix app crash on iOS 15

Temporary fix for crash when opening profile.
Full solution in feature/profile-redesign.

Refs: #789"

# Fast-track review
# Test on device
# Merge to main immediately
git checkout main
git merge hotfix/fix-app-crash
git tag -a v2.0.1 -m "Emergency fix v2.0.1"

# Deploy to app stores
# Merge back to develop
git checkout develop
git merge hotfix/fix-app-crash
```

**Branch Protection Rules**

```bash
# On GitHub:

# Main branch
- Require pull request reviews (2 reviewers)
- Require status checks to pass
- Require branches to be up to date
- Restrict pushes to main branch
- Require review from code owners

# Develop branch
- Require pull request reviews (1 reviewer)
- Require status checks to pass
- Allow administrators to bypass

# Feature branches
- No restrictions
- Can be force pushed
```

**Automated CI/CD Pipeline**

```yaml
# .github/workflows/mobile-ci.yml
name: Mobile CI

on:
  push:
    branches: [develop, main]
  pull_request:
    branches: [develop]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Node.js
        uses: actions/setup-node@v2
        with:
          node-version: "16"

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test

      - name: Run linting
        run: npm run lint

      - name: Build Android
        run: npm run build:android
        env:
          ANDROID_HOME: /opt/android-sdk

      - name: Build iOS
        run: npm run build:ios
        env:
          APPLE_CERTIFICATE: ${{ secrets.APPLE_CERTIFICATE }}
```

**Code Ownership**

```yaml
# CODEOWNERS
# Global owners
* @tech-lead

# iOS specific
ios/* @ios-developer-1 @ios-developer-2
*.swift @ios-developer-1 @ios-developer-2
*.plist @ios-developer-1 @ios-developer-2

# Android specific
android/* @android-developer-1 @android-developer-2
*.kt @android-developer-1 @android-developer-2
*.java @android-developer-1 @android-developer-2

# Backend
backend/* @backend-developer
api/* @backend-developer
*.py @backend-developer

# Shared/UI
shared/* @mobile-team
components/* @mobile-team @ui-ux-designer
```

**Documentation and Onboarding**

```markdown
# CONTRIBUTING.md

## Getting Started

1. Clone the repository
2. Install dependencies: `npm install`
3. Setup development environment
4. Read our coding standards

## Branch Naming

- feature/short-description
- bugfix/issue-number-short-desc
- hotfix/short-description
- refactor/area-short-desc

## Commit Messages

- feat: new feature
- fix: bug fix
- refactor: code improvement
- docs: documentation
- test: adding tests
- chore: maintenance

## Testing Requirements

- Unit tests for all new code
- Manual testing on both platforms
- Integration tests for API changes
```

**Key Success Factors:**

1. **Clear branching strategy** suitable for mobile development
2. **Automated testing** for both iOS and Android
3. **Code ownership** for platform-specific changes
4. **Regular synchronization** to prevent large conflicts
5. **Clear documentation** for team members
6. **Emergency procedures** for production issues
7. **CI/CD integration** for consistent quality

**Key Points to Mention:**

- Workflow design based on team size and needs
- Branch structure and protection rules
- Pull request and code review process
- Release management process
- Hotfix procedures
- Automated testing and CI/CD
- Code ownership and documentation

---

## System Design with Git

### Question 31: How would you design a Git strategy for a microservices architecture?

**Sample Answer:**
For a microservices architecture, I'd implement a multi-repository strategy with clear ownership and coordination:

**Repository Structure**

**Option 1: One Repository Per Service (Recommended)**

```
/user-service/
  .git/
  src/
  tests/
  Dockerfile
  README.md

/order-service/
  .git/
  src/
  tests/
  Docker
  README.md

/payment-service/
  .git/
  src/
  tests/
  Dockerfile
  README.md

/shared-libraries/
  .git/
  common-utils/
  types/
  README.md

/infrastructure/
  .git/
  kubernetes/
  terraform/
  scripts/
  README.md
```

**Option 2: Monorepo with Service Directories**

```
/microservices/
  .git/
  services/
    user-service/
    order-service/
    payment-service/
  libraries/
    shared-utils/
  infrastructure/
```

**Branching Strategy for Each Service**

**Service-Level Workflow:**

```bash
# Each service follows GitHub Flow
main                    # Production code
develop                # Integration branch
feature/*             # New features
bugfix/*              # Bug fixes
hotfix/*              # Emergency fixes
```

**Service-Specific Configurations:**

```bash
# user-service/.git/config
[core]
    repositoryformatversion = 0
    filemode = true
    bare = false
    logallrefupdates = true
[remote "origin"]
    url = https://github.com/company/user-service.git
    fetch = +refs/heads/*:refs/remotes/origin/*
[branch "main"]
    remote = origin
    merge = refs/heads/main
[user]
    name = Developer Name
    email = dev@company.com
```

**Cross-Service Coordination**

**1. Shared Libraries Repository**

```bash
# shared-libraries follows semantic versioning
v1.2.3  # Major.Minor.Patch

# Service usage in package.json
"dependencies": {
  "@company/shared-utils": "^1.2.0",
  "@company/types": "^2.0.0"
}

# Update process
git checkout -b feature/add-validation
# Make changes in shared-libraries
git tag v1.3.0
git push origin v1.3.0

# Services update dependency
npm install @company/shared-utils@^1.3.0
git commit -m "chore: update shared-utils to v1.3.0"
```

**2. Contract Testing**

```bash
# API contract in shared repository
/user-service/
  contracts/
    user-api-schema.yaml
    order-service-contracts.yaml

# Generate types and validate
npm run generate-types
npm run validate-contracts
```

**3. Integration Testing**

```yaml
# .github/workflows/integration.yml
name: Cross-Service Integration

on:
  push:
    branches: [develop]
    paths: ["services/**"]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout all services
        run: |
          git clone https://github.com/company/user-service
          git clone https://github.com/company/order-service
          git clone https://github.com/company/payment-service

      - name: Start test environment
        run: docker-compose -f docker-compose.test.yml up -d

      - name: Run integration tests
        run: |
          npm run test:integration
          npm run test:contracts
```

**Service Ownership and Access Control**

**CODEOWNERS for each service:**

```bash
# user-service/CODEOWNERS
* @team-user-service
*.py @user-service-backend
*.js @user-service-frontend
*.yaml @devops-team
```

**Branch Protection Rules:**

```bash
# Main branch for each service:
- Require 2 approvals
- Require all tests to pass
- Require contract validation
- Require security scan
- Restrict pushes to main branch
```

**Release Coordination**

**1. Independent Releases**

```bash
# Each service versioned independently
user-service: v2.1.0
order-service: v1.3.2
payment-service: v3.0.0

# Deploy in order of dependencies
# 1. Shared libraries
# 2. Payment service (foundational)
# 3. User service
# 4. Order service (depends on both)
```

**2. Coordinated Releases**

```yaml
# infrastructure/release-orchestrator/
name: Coordinated Release

on:
  workflow_dispatch:
    inputs:
      user_service_version:
        description: "User Service Version"
        required: true
      order_service_version:
        description: "Order Service Version"
        required: true

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy services in order
        run: |
          ./deploy-service.sh user-service ${{ github.event.inputs.user_service_version }}
          ./deploy-service.sh order-service ${{ github.event.inputs.order_service_version }}

      - name: Run integration tests
        run: ./test-end-to-end.sh
```

**Communication Between Services**

**1. API Versioning**

```bash
# Each service maintains API compatibility
/user-service/
  src/
    routes/
      v1/
        users.js
      v2/
        users.js  # New API version

# Deprecation policy
# - Support 2 previous API versions
# - 6-month deprecation warning
# - Clear migration guide
```

**2. Event-Driven Communication**

```yaml
# Event contracts in shared repository
# shared-events/
events/
user.created.v1.yaml
order.completed.v1.yaml
payment.processed.v1.yaml
# Services consume and publish events
# Version managed through event schemas
```

**Database Migration Strategy**

**1. Database-Per-Service**

```bash
# Each service owns its database schema
/user-service/
  migrations/
    001_create_users_table.sql
    002_add_user_preferences.sql

# Migration execution
npm run migrate:up    # Production
npm run migrate:test  # Test environment
```

**2. Schema Evolution**

```bash
# Backward compatibility
# Each service ensures:
# - Old API versions still work
# - Database changes don't break existing functionality
# - Clear migration path for consumers

# Example: Adding column
ALTER TABLE users ADD COLUMN preferences JSON;
-- Old code: continues to work
-- New code: can use preferences
```

**Monitoring and Observability**

**1. Service Health Monitoring**

```bash
# Each service exposes health endpoints
GET /health
GET /health/dependencies
GET /metrics

# Git tags include health check version
git tag v2.1.0-health-v1
```

**2. Error Tracking**

```yaml
# .github/workflows/error-tracking.yml
name: Error Monitoring

on:
  push:
    tags: ["v*"]

jobs:
  update-error-tracking:
    runs-on: ubuntu-latest
    steps:
      - name: Update service version in monitoring
        run: |
          curl -X POST https://monitoring.company.com/api/services \
            -d "service=user-service&version=${{ github.ref_name }}"
```

**Benefits of This Strategy:**

1. **Independent Development**: Teams work at their own pace
2. **Technology Diversity**: Each service can use different technologies
3. **Deployment Flexibility**: Deploy services independently
4. **Clear Ownership**: Each service has dedicated team
5. **Scalability**: Easy to add new services
6. **Isolation**: Failures in one service don't affect others

**Challenges and Solutions:**

1. **Coordination Overhead**: Use automated release orchestration
2. **Dependency Management**: Semantic versioning and contract testing
3. **Testing Complexity**: End-to-end testing in CI/CD pipeline
4. **Configuration Management**: Use service mesh and config service

**Key Points to Mention:**

- Multi-repository strategy for microservices
- Independent versioning and releases
- Cross-service coordination mechanisms
- Shared libraries and contracts
- Automated testing and deployment
- Service ownership and access control

---

## Practical Coding Questions

### Question 32: Write a Git command to find all commits that contain a specific file that was later deleted.

**Sample Answer:**
There are several ways to find commits that contained a specific file that was later deleted:

**Method 1: Using `git log` with `--diff-filter`**

```bash
# Find all commits that added, modified, or removed the file
git log --oneline --follow --diff-filter=AMDR -- "*path/to/deleted-file.txt"

# Explanation:
# --follow: Follow file renames
# --diff-filter=AMDR: Show commits that Added, Modified, Deleted, orRenamed
# "*": Use glob pattern for file path
```

**Method 2: Using `git log` with file path**

```bash
# Show history of a deleted file
git log --oneline -- "*path/to/deleted-file.txt"

# With full diffs
git log -p -- "*path/to/deleted-file.txt"

# Show just the commits (not the diffs)
git log --oneline -- "*path/to/deleted-file.txt"
```

**Method 3: Using `git log` with `--all` and `--source`**

```bash
# Search in all branches
git log --all --oneline --source -- "*path/to/deleted-file.txt"

# Combine with --follow for renames
git log --all --follow --oneline -- "*path/to/deleted-file.txt"
```

**Method 4: Using `git log` with `--since` and `--until`**

```bash
# Find commits in specific time range
git log --oneline --since="2023-01-01" --until="2023-12-31" -- "*path/to/deleted-file.txt"
```

**Method 5: Using `git log` with `--author`**

```bash
# Find commits by specific author
git log --author="Sarah" --oneline -- "*path/to/deleted-file.txt"
```

**Practical Examples:**

**Example 1: Find when a config file was deleted**

```bash
# Find all commits involving config.json
git log --oneline --follow config.json

# Output might show:
# abc123 Add new config structure
# def456 Update config file
# ghi789 Delete old config.json
```

**Example 2: Find the last commit that modified a deleted file**

```bash
# Show the last commit that touched the file
git log -1 --oneline -- "*path/to/deleted-file.txt"

# Get the full diff of that commit
git show <commit-hash> -- "*path/to/deleted-file.txt"
```

**Example 3: Find commits that deleted specific pattern**

```bash
# Find commits that deleted test files
git log --diff-filter=D --oneline -- "*test*.py"

# Find commits that deleted any .log file
git log --diff-filter=D --oneline -- "*.log"
```

**Advanced Usage:**

**Get file content from last commit that had it:**

```bash
# Restore the file to your working directory
git checkout <commit-hash>~1 -- path/to/deleted-file.txt

# Or get the content directly
git show <commit-hash>:path/to/deleted-file.txt > restored-file.txt
```

**Find deleted files in a specific directory:**

```bash
# Find all deleted files in src/ directory
git log --diff-filter=D --summary --oneline -- "src/*"

# Find all deleted JavaScript files
git log --diff-filter=D --oneline -- "*.js"
```

**Search in commit messages:**

```bash
# Find commits that mention "delete" in message and touched a file
git log --grep="delete" --oneline -- "*specific-file.txt"
```

**Useful Flags for File History:**

```bash
# --name-status: Show which files were affected
git log --oneline --name-status -- "*path/to/file.txt"

# --stat: Show statistics of changes
git log --oneline --stat -- "*path/to/file.txt"

# --pretty: Custom output format
git log --pretty=format:"%h %ad %s" --date=short -- "*path/to/file.txt"
```

**Script to Find All Deleted Files:**

```bash
#!/bin/bash
# find-deleted-files.sh

echo "Finding all deleted files in repository:"

git log --diff-filter=D --summary --pretty=format:"%h %s" | \
  while read commit msg; do
    echo "Commit: $commit - $msg"
    git diff-tree --no-commit-id --name-only -r $commit | \
      while read file; do
        echo "  Deleted: $file"
      done
  done
```

**When This Is Useful:**

1. **Debugging**: Understanding when a file was removed
2. **Code Recovery**: Finding and restoring accidentally deleted code
3. **Audit Trails**: Tracking what files were removed and why
4. **Code Archaeology**: Understanding project history
5. **Debugging Issues**: Finding when a critical file was removed

**Key Points to Mention:**

- Multiple methods to find deleted file history
- Combining different flags for specific searches
- How to restore content from deleted files
- Practical examples for real-world usage
- When this technique is useful

---

This completes the comprehensive Git interview questions section! The questions cover a wide range of topics from basic concepts to advanced scenarios, system design, and practical applications. Each answer provides detailed explanations with code examples and real-world context.

Remember: The best way to master Git interview questions is to practice these commands in real scenarios and understand the underlying concepts, not just memorize the syntax.
