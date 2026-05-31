# csa-sar-imaging — Claude Instructions

## GitHub Project Management

This repo is tracked in a GitHub Projects v2 board. Claude should manage issues, milestones, and project status as part of normal work.

**Repo:** `jentsechen/csa-sar-imaging`
**Project board:** `https://github.com/users/jentsechen/projects/4`
**Project node ID:** `PVT_kwHODaH-is4BZShn`
**Project owner:** `jentsechen`

### Milestones

| # | Title | Due |
|---|---|---|
| 1 | Phase 1 – Speckle Statistical Analysis | 2026-06-14 |
| 2 | Phase 2 – Data Collection | 2026-07-12 |
| 3 | Phase 3 – Speckle Noise Injection | 2026-06-21 |
| 4 | Phase 4 – Python CSA | 2026-07-19 |
| 5 | Phase 5 – Despeckle Benchmark | 2026-08-09 |
| 6 | Phase 6 – YOLO Object Detection | 2026-09-13 |
| 7 | Phase 7 – Detection Metric Calculation | 2026-09-27 |
| 8 | Phase 8 – Integration & Final Report | 2026-10-11 |

### Issue map (ID → task)

| Issue | Task |
|---|---|
| #1 | [1.1] Literature review: speckle models |
| #2 | [1.2] Measure speckle statistics on Umbra images |
| #3 | [1.3] Fit distribution model and validate goodness-of-fit |
| #4 | [1.4] Determine number-of-looks (L) from real data |
| #5 | [1.5] Document chosen speckle model |
| #6 | [2.1] Identify additional Umbra open-data scenes |
| #7 | [2.2] Download and register additional Umbra GeoTIFFs |
| #8 | [2.3] Download public SAR detection datasets |
| #9 | [2.4] Convert labels to YOLO format; train/val/test split |
| #10 | [2.5] Document dataset inventory |
| #11 | [3.1] Implement multiplicative speckle noise generator |
| #12 | [3.2] Validate injected speckle statistics |
| #13 | [3.3] Save noisy image set; update diagrams |
| #14 | [4.1] Port ImagingPar and SigPar to Python |
| #15 | [4.2] Python CSA: range compression |
| #16 | [4.3] Python CSA: chirp scaling step |
| #17 | [4.4] Python CSA: RCMC |
| #18 | [4.5] Python CSA: azimuth compression |
| #19 | [4.6] Validate Python CSA against C++ output |
| #20 | [4.7] Test Python CSA on multi-point scene |
| #21 | [5.1] Add Frost, Kuan, Gamma-MAP filters |
| #22 | [5.2] Add Non-Local Means (NLM) filter |
| #23 | [5.3] (Optional) DnCNN / SAR-CNN inference |
| #24 | [5.4] Implement despeckle metric suite |
| #25 | [5.5] Comparison table + LaTeX report update |
| #26 | [6.1] Set up YOLOv8/v11 environment |
| #27 | [6.2] YOLO baseline training run |
| #28 | [6.3] Fine-tune YOLO |
| #29 | [6.4] Run detection on all despeckle variants |
| #30 | [7.1] Implement mAP, Precision, Recall, F1 |
| #31 | [7.2] Per-class metric breakdown |
| #32 | [7.3] Confusion matrix and PR curves |
| #33 | [7.4] Statistical significance test |
| #34 | [7.5] Summary table: best despeckle method |
| #35 | [8.1] End-to-end pipeline script |
| #36 | [8.2] Final figures and summary plots |
| #37 | [8.3] Update simulation_report.tex |

### How Claude should manage the project

**When starting a task:** check the open issue, note its number.

**When finishing a task:** close the issue via commit message:
```
git commit -m "Brief description

closes #N"
```
Or close manually: `gh issue close N --repo jentsechen/csa-sar-imaging`

**To set project item status** (Todo → In Progress → Done):
```bash
# 1. Get item ID for an issue
ITEM_ID=$(gh api graphql -f query='
{
  node(id: "PVT_kwHODaH-is4BZShn") {
    ... on ProjectV2 {
      items(first: 100) {
        nodes {
          id
          content { ... on Issue { number } }
        }
      }
    }
  }
}' --jq '.data.node.items.nodes[] | select(.content.number == ISSUE_NUM) | .id')

# 2. Get Status field option ID
# Todo:        f75ad846
# In Progress: 47fc9ee4
# Done:        98236657

# 3. Update status
gh api graphql -f query="
mutation {
  updateProjectV2ItemFieldValue(input: {
    projectId: \"PVT_kwHODaH-is4BZShn\",
    itemId: \"$ITEM_ID\",
    fieldId: \"PVTSSF_lAHODaH-is4BZShnzhUSOTQ\",
    value: { singleSelectOptionId: \"47fc9ee4\" }
  }) { projectV2Item { id } }
}"
```

**To create a new issue:**
```bash
gh issue create --repo jentsechen/csa-sar-imaging \
  --title "Title" \
  --body "Body" \
  --milestone "Phase N – ..." \
  --label "phase-N,<type>"
```

**To check milestone progress:**
```bash
gh api repos/jentsechen/csa-sar-imaging/milestones --jq '.[] | "\(.title): \(.closed_issues)/\(.open_issues + .closed_issues)"'
```

### Prerequisites on a new device

Run `~/claude-config/setup-gh-project.sh` to verify `gh` is authenticated with the `project` scope. If not authenticated, it prints the exact command to run.

### Weekly workflow

1. Append to `PROGRESS.md` each session (2–3 lines: date, what was done, key decision)
2. Commit `PROGRESS.md` alongside code changes
3. Use `closes #N` in commit messages to auto-close issues
