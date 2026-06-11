#!/usr/bin/env bash
# Publish benchmark graphs to a SEPARATE assets repo (JuliaGeo/JuliaGeoBenchmarkResults, branch
# `ConservativeRegridding`) and post/update a single sticky PR comment that embeds them inline.
# Hosting the PNG blobs outside the ConservativeRegridding repo keeps its git history free of
# benchmark cruft.
#
# The cross-repo push uses an SSH *deploy key* (secret BENCHMARK_ASSETS_DEPLOY_KEY) scoped with
# write access to the assets repo only — the stock GITHUB_TOKEN is scoped to its own repo and
# cannot push to another. The sticky comment is still posted to the CR repo via the stock GH_TOKEN
# (pull-requests:write).
#
# Fork PRs receive neither the deploy-key secret nor a writable token, so the push and comment
# degrade gracefully — the graphs remain downloadable as the run's uploaded artifact.
#
# Required env: GH_TOKEN, GITHUB_REPOSITORY (the CR repo, for the comment), PR_NUMBER, RUN_ID.
# Optional env: BASE_LABEL, SSH_DEPLOY_KEY (absent on fork PRs → asset push skipped).
set -uo pipefail

: "${GH_TOKEN:?}"; : "${GITHUB_REPOSITORY:?}"; : "${PR_NUMBER:?}"; : "${RUN_ID:?}"
BASE_LABEL="${BASE_LABEL:-base}"
RESULTS="bench/results"
ASSET_REPO="JuliaGeo/JuliaGeoBenchmarkResults"   # public cross-repo image host
ASSET_BRANCH="ConservativeRegridding"            # one branch per package in the shared repo
ASSET_DIR="pr-${PR_NUMBER}"
MARKER="<!-- conservativeregridding-benchmarks -->"

shopt -s nullglob
pngs=("$RESULTS"/*.png)
if [ ${#pngs[@]} -eq 0 ]; then echo "no graphs to publish"; exit 0; fi

# --- push graphs to the assets repo (SSH deploy key), under a per-PR subdir -------------------
PUBLISHED=0
if [ -z "${SSH_DEPLOY_KEY:-}" ]; then
    echo "BENCHMARK_ASSETS_DEPLOY_KEY not present (fork PR?) — skipping asset push"
else
    keyfile="$(mktemp)"; printf '%s\n' "$SSH_DEPLOY_KEY" > "$keyfile"; chmod 600 "$keyfile"
    export GIT_SSH_COMMAND="ssh -i $keyfile -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
    remote="git@github.com:${ASSET_REPO}.git"
    tmp="$(mktemp -d)"
    if git clone --quiet --depth 1 --branch "$ASSET_BRANCH" "$remote" "$tmp" 2>/dev/null; then
        echo "cloned ${ASSET_REPO}@${ASSET_BRANCH}"
    else
        # Branch missing on a fresh assets repo: start it as an orphan.
        git clone --quiet --depth 1 "$remote" "$tmp"
        git -C "$tmp" checkout --orphan "$ASSET_BRANCH"
        git -C "$tmp" rm -rqf . >/dev/null 2>&1 || true
    fi
    mkdir -p "$tmp/$ASSET_DIR"
    cp "$RESULTS"/*.png "$tmp/$ASSET_DIR/"
    git -C "$tmp" add -A
    git -C "$tmp" -c user.name="github-actions[bot]" \
        -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
        commit --quiet -m "ConservativeRegridding PR #${PR_NUMBER} benchmark graphs (run ${RUN_ID})" \
        || echo "no new graph bytes to commit"
    if git -C "$tmp" push --quiet origin "$ASSET_BRANCH"; then
        PUBLISHED=1
        echo "published to ${ASSET_REPO}@${ASSET_BRANCH}/${ASSET_DIR}"
    else
        echo "WARNING: push to ${ASSET_REPO} failed"
    fi
fi

# --- compose the comment with inline, cache-busted images ------------------------------------
raw="https://raw.githubusercontent.com/${ASSET_REPO}/${ASSET_BRANCH}/${ASSET_DIR}"
runurl="https://github.com/${GITHUB_REPOSITORY}/actions/runs/${RUN_ID}"
body="$(mktemp)"
{
    echo "$MARKER"
    echo "## Benchmark results"
    echo
    echo "> Shared CI runners are noisy — treat small differences as noise. [Full-resolution graphs are attached to the run as an artifact.](${runurl})"
    echo
    if [ "$PUBLISHED" = 1 ]; then
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
    else
        echo "_Graphs were not published to the assets repo on this run (no deploy key — e.g. a fork PR); see the artifact linked above._"
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
