# System Instructions for Gemini CLI

When working in the `everspring-mcp` project, you MUST adhere to the following rules:

1. **Load Context:** At the beginning of any complex task, refactoring, or architectural discussion, read the `PROJECT_CONTEXT.md` file located in the root of the workspace to understand the system's architecture, data flows, and tiers.
2. **Maintain Context:** If your changes introduce new core modules, alter the data pipeline (Scrape -> S3 -> Sync -> Index -> Serve), or change the tiered architecture (main, slim, xslim), you are required to update `PROJECT_CONTEXT.md` to reflect these changes.
