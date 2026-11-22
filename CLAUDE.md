# CLAUDE.md — LifeboatCacheTest

## Working on GitHub as the agent account (`interplayteam`)

Per Arul (2026-11-22). Claude Code works on GitHub as **`interplayteam`**, the
agent account. Arul (`arul329`) approves its changes; Brian
(`briansathianathan`) is the backup approver. The account has **Write** access
to AgentOneHub, Lifeboat, LifeboatCache, LifeboatCacheTest and iterate-docs. It
is not an admin and not a code owner, so it can never approve its own work.

**The token** is in the macOS keychain on Arul's build Mac, item
`agentwatch-agent-gh`, account `interplayteam`. Never print it, write it to a
file or commit it. Pass it per command:

```bash
T=$(security find-generic-password -s agentwatch-agent-gh -a interplayteam -w)
git -c user.name=interplayteam -c user.email=interplayteam@iterate.ai commit -m "..."
GH_TOKEN=$T git -c credential.helper= -c 'credential.helper=!gh auth git-credential' \
  push -u origin <branch>
GH_TOKEN=$T gh pr create --base main --head <branch> --title "..." --body-file <f> \
  --reviewer arul329
```

A PR opened with the default `gh` login (`arul329`) is authored by Arul, and
Arul then cannot approve it. Always pass the agent token.

**How a change lands:**
1. Work on a branch. Never push to `main`.
2. Open the PR as `interplayteam` and request review from `arul329`.
3. Send Arul the PR link and ask him to approve it.
4. Merge only after the PR shows `reviewDecision == APPROVED`. Check it with
   `gh pr view <n> --json reviewDecision`. Then merge as the agent:
   `GH_TOKEN=$T gh pr merge <n> --squash`.
   If `main` is not protected, GitHub will not stop a merge without an
   approval, so this check is the control.

This holds for docs- and test-only changes too. SOC 2 (Vanta) samples changes
and asks for each one's approval (CC8.1). It applies to AgentWatch now, and
Lifeboat is expected next.
