#!/usr/bin/env bash
# ping_discord_portable.sh — standalone Discord ping (no ai-ops framework needed).
# Usage: ping_discord_portable.sh "your message"
#
# Token: put it in ~/.config/discord_ping.env as:
#     export DISCORD_BOT_TOKEN="Bot YOUR_TOKEN_HERE"
#   then: chmod 600 ~/.config/discord_ping.env
# (Or just `export DISCORD_BOT_TOKEN="Bot ..."` in the shell before calling.)

MSG="${1:-done}"
CHANNEL_ID="1477468279697051892"
ENV_FILE="${HOME}/.config/discord_ping.env"

[ -f "$ENV_FILE" ] && source "$ENV_FILE"

if [ -z "${DISCORD_BOT_TOKEN:-}" ]; then
  echo "ERROR: DISCORD_BOT_TOKEN not set (put it in $ENV_FILE)" >&2
  exit 1
fi

curl -s -X POST "https://discord.com/api/v10/channels/${CHANNEL_ID}/messages" \
  -H "Authorization: ${DISCORD_BOT_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"content\": $(printf '%s' "$MSG" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}" \
  > /dev/null 2>&1
