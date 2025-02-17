Goal: Analysis of data from a wearable using natural language.

Support:

- Garmin (In-progress)
- TODO

## Dev guide

### ENV

- Set all required env vars in .env
- `source .env && python app/main.py`

### Langfuse

- Clone and run server: https://langfuse.com/self-hosting/local
- Set up API keys via UI

## TODOs

- [x] observability with text
- [ ] observability with images
- [x] handling long text outputs in jupyter cells
- [ ] handling errors in executing notebook
- [ ] saving traces
- [ ] profiling speed of solver
- [ ] dataset of tasks for evaluation
- [ ] shorter cell outputs
