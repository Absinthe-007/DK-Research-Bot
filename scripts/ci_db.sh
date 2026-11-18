#!/usr/bin/env bash
# Database persistence for the upgrade workflows. Same scheme as .github/workflows/pipeline.yml:
# Postgres runs inside the job; between runs the database lives as a pg_dump in the "data" release.
#   scripts/ci_db.sh restore   start Postgres, load the saved dump (if any), create/upgrade tables
#   scripts/ci_db.sh save      dump the database back to the release (overwrites db.dump)
set -euo pipefail
TAG="${DATA_TAG:-data}"
PGURL="${PGURL:-postgresql://dkbot:dkbot@localhost:5432/dkbot}"

case "${1:-}" in
  restore)
    sudo systemctl start postgresql.service
    sudo -u postgres psql -q -c "CREATE USER dkbot WITH PASSWORD 'dkbot' SUPERUSER;" -c "CREATE DATABASE dkbot OWNER dkbot;"
    mkdir -p state /tmp/data
    if gh release download "$TAG" -p db.dump -p state.json -D /tmp/data 2>/dev/null; then
      pg_restore --no-owner --role=dkbot -d "$PGURL" /tmp/data/db.dump || echo "restore had warnings"
      cp /tmp/data/state.json state/ 2>/dev/null || true
    else
      echo "No saved database yet - starting fresh."
    fi
    python -m scripts.init_db
    ;;
  save)
    pg_dump -Fc --no-owner "$PGURL" > /tmp/db.dump
    ls -lh /tmp/db.dump
    gh release view "$TAG" >/dev/null 2>&1 || gh release create "$TAG" --title "Pipeline data" \
      --notes "Database dump + planner state, overwritten by every pipeline run. Not a code release." --latest=false
    FILES="/tmp/db.dump"; [ -f state/state.json ] && FILES="$FILES state/state.json"
    gh release upload "$TAG" $FILES --clobber
    ;;
  *) echo "usage: $0 restore|save" >&2; exit 2 ;;
esac
