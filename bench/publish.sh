#!/usr/bin/env bash
# Publish benchmark graphs to the orphan `benchmark-assets` branch and post/update a single
# sticky PR comment that embeds them inline. Uses ONLY the stock GITHUB_TOKEN
# (contents:write + pull-requests:write) — no bot PAT or gist. Same-repo PRs only: fork PRs get
# a read-only token, so the push/comment is skipped gracefully.
#
# Required env: GH_TOKEN, GITHUB_REPOSITORY (owner/repo), PR_NUMBER, RUN_ID. Optional: BASE_LABEL.
set -uo pipefail

: "${GH_TOKEN:?}"; : "${GITHUB_REPOSITORY:?}"; : "${PR_NUMBER:?}"; : "${RUN_ID:?}"
BASE_LABEL="${BASE_LABEL:-base}"
RESULTS="bench/results"
ASSET_BRANCH="benchmark-assets"
ASSET_DIR="pr-${PR_NUMBER}"
MARKER="<!-- conservativeregridding-benchmarks -->"

shopt -s nullglob
pngs=("$RESULTS"/*.png)
if [ ${#pngs[@]} -eq 0 ]; then echo "no graphs to publish"; exit 0; fi

# --- push graphs to the orphan asset branch, under a per-PR subdir ---------------------------
remote="https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
tmp="$(mktemp -d)"
if git clone --quiet --depth 1 --branch "$ASSET_BRANCH" "$remote" "$tmp" 2>/dev/null; then
    echo "cloned existing $ASSET_BRANCH"
else
    git clone --quiet --depth 1 "$remote" "$tmp"
    git -C "$tmp" checkout --orphan "$ASSET_BRANCH"
    git -C "$tmp" rm -rqf . >/dev/null 2>&1 || true
fi
mkdir -p "$tmp/$ASSET_DIR"
cp "$RESULTS"/*.png "$tmp/$ASSET_DIR/"
git -C "$tmp" add -A
git -C "$tmp" -c user.name="github-actions[bot]" \
    -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
    commit --quiet -m "Benchmark graphs for PR #${PR_NUMBER} (run ${RUN_ID})" \
    && git -C "$tmp" push --quiet origin "$ASSET_BRANCH" \
    || echo "nothing to push (no graph changes)"

# --- compose the comment with inline, cache-busted images ------------------------------------
raw="https://raw.githubusercontent.com/${GITHUB_REPOSITORY}/${ASSET_BRANCH}/${ASSET_DIR}"
runurl="https://github.com/${GITHUB_REPOSITORY}/actions/runs/${RUN_ID}"
body="$(mktemp)"
{
    echo "$MARKER"
    echo "## Benchmark results"
    echo
    echo "> Shared CI runners are noisy — treat small differences as noise. [Full-resolution graphs are attached to the run as an artifact.](${runurl})"
    echo
    if [ -f "$RESULTS/pr_vs_master.png" ]; then
        echo "### Regridder construction: this PR vs \`${BASE_LABEL}\`"
        echo "![pr vs base](${raw}/pr_vs_master.png?v=${RUN_ID})"
        echo
    fi
    if [ -f "$RESULTS/scaling.png" ]; then
        echo "### Construction scaling by grid family"
        echo "![scaling](${raw}/scaling.png?v=${RUN_ID})"
        echo
    fi
    if [ -f "$RESULTS/xesmf.png" ]; then
        echo "### ConservativeRegridding vs XESMF (Oceananigans)"
        echo "![xesmf](${raw}/xesmf.png?v=${RUN_ID})"
        echo
    fi
    if [ -f "$RESULTS/summary.md" ]; then
        echo "<details><summary>Construction-time table (PR vs ${BASE_LABEL})</summary>"
        echo
        cat "$RESULTS/summary.md"
        echo
        echo "</details>"
    fi
} > "$body"

# --- upsert the sticky comment (find by hidden marker) ---------------------------------------
cid="$(gh api "repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/comments" --paginate \
    --jq ".[] | select(.body | contains(\"${MARKER}\")) | .id" 2>/dev/null | head -n1)"
if [ -n "${cid:-}" ]; then
    gh api --method PATCH "repos/${GITHUB_REPOSITORY}/issues/comments/${cid}" -F body=@"$body" >/dev/null \
        && echo "updated sticky comment ${cid}"
else
    gh api --method POST "repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/comments" -F body=@"$body" >/dev/null \
        && echo "posted sticky comment"
fi
