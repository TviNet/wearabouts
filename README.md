# Analysis of data from a wearable using natural language.

## Support:

-   Garmin (In-progress)
-   TODO

## Usage:

```
> source .env
> PYTHONPATH=./ python app/main.py --task "Plot my sleep times and steps daily for last week in the same plot"
```

An executed ipynb notebook will be created in the artifacts directory with the desired task completed.
Examples of the output are in the `examples` directory.

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
-   [ ] support local llm
-   [ ] observability with images
-   [ ] saving traces
-   [ ] profiling speed of solver
-   [ ] dataset of tasks for evaluation
-   [ ] handle case where login fails
