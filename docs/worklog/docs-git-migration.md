# Docs migration into the active platform repository

The research record was moved from the multi-repository workspace root
`/home/fang/code/LLM4AD/docs/` to the active Git repository at
`/home/fang/code/LLM4AD/LLM4AD/docs/`.

The `ideas/`, `experiments/`, `results/`, and `worklog/` structure and all
existing content were preserved. Result image links were changed from obsolete
absolute paths to repository-relative links. The five plotting scripts under
`docs/results/figures/` were updated for the new directory depth and executed
successfully after migration.

The workspace-root `AGENTS.md` now points research records explicitly to
`LLM4AD/docs/`, so future notes are created inside the tracked repository.
