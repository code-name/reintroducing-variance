# Example: a shadow screen (invented data)

An illustrative data folder for the field test (`benchmark.shadow`). **Everything here is
invented**: twelve made-up product and service ideas, four made-up evaluators who rated each idea
for novelty and usefulness on a 1-5 scale, and the organization's decision in `ideas.csv`
(`advanced`: 1 if the idea went forward, 0 if not). Use it to see how the field test works, not to
estimate anything.

```bash
python -m benchmark.judge run --data-dir examples/screen --backend fake --perturbations P0_baseline \
       --judges qwen3.8-27b,deepseek-v4-pro          # fake offline judges; drop --backend fake to use real ones
python -m benchmark.shadow --data-dir examples/screen
```

The shadow screen reports, for each AI judge and for the panel of all judges, how many of the
advanced ideas the AI judge would have dropped if it had advanced the same number of ideas, and
whether those are the ideas the evaluators disagreed about. `debrief_sheet.csv` is the sheet to
show evaluators after they have decided.
