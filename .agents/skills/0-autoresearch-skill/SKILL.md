---
name: autoresearch
description: Continue a user-requested autonomous research effort through ideas, implementation, experiments, synthesis, and paper drafting. Use when the user asks for sustained research execution; ordinary idea discussion does not start an autonomous experiment campaign.
version: 1.0.0
author: Orchestra Research
license: MIT
tags: [Autonomous Research, Two-Loop Architecture, Experiment Orchestration, Research Synthesis, Project Management]
---

# Autoresearch

Act as a research collaborator with domain judgment. Follow the user's research goal and the repository's AGENTS.md; carry authorized work forward using the existing project structure.

## Think and iterate

Start with the question that matters most for progress. Understand the problem structure, current approach and bottleneck. Propose a promising mechanism and explain why it could help. Use relevant domain skills when they add practical value.

Alternate focused experiments with synthesis: implement the idea, run an informative comparison, understand what happened, and choose the next step. When progress stalls, reconsider the mechanism or research direction rather than extending the same experiment loop automatically.

Choose experiment scale to fit the question and available authorized resources. Preserve the configuration and recovery rules of active formal runs. Reuse existing results where they answer the question; report actual results accurately.

## Work with this repository

Use `docs/README.md` to locate method designs, experiment packages, analyses, and writing material. Update the relevant existing document when there is a meaningful advance. Keep enough context to resume: what changed, what was learned, and what to do next. Use the project's existing logs and manifests instead of creating a second research-state system.

Routine reversible decisions within the requested work need no additional confirmation. Ask only for missing decisions that materially affect the work or actions outside the existing authorization. Stop when the requested outcome is achieved, the agreed resource budget is exhausted, or progress requires user input.

## Continuity and communication

Long research tasks can continue within the active session. Scheduled continuation is optional and used only when requested; use the host's available automation tools. Timers, cron jobs, automatic commits, and external notifications are not prerequisites for research.

Share concise updates when there is a meaningful finding or change of direction. Create plots when they explain results; prepare slides or formal reports when useful for the requested deliverable.

## Toward a paper

As understanding develops, connect the research question, method, and results into a clear contribution. Draft requested sections using the existing framing and verified results, noting consequential gaps briefly. Use `ml-paper-writing` for manuscript preparation. A paper draft is a useful deliverable; submission follows the user's authorization.
