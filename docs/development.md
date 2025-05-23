## Dev guide

### ENV

-   Set all required env vars in .env
-   `source .env && python app/main.py`

### Langfuse

-   Clone and run server: https://langfuse.com/self-hosting/local
-   Set up API keys via UI

## TODOs

-   [x] observability with text
-   [x] handling long text outputs in jupyter cells
-   [x] handling errors in executing notebook
-   [x] partial execution of notebook to avoid repeated login
-   [x] shorter cell outputs
-   [x] verify output loop
-   [x] allow feedback and start from previous state
-   [ ] save solver state gracefully
-   [ ] support local llm
-   [ ] observability with images
-   [ ] saving traces
-   [ ] profiling speed of solver
-   [ ] dataset of tasks for evaluation
-   [ ] handle case where login fails
