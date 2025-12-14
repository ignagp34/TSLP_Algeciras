# GA Parameter Sensitivity Analysis - Quick Start Guide

## Overview

This analysis addresses reviewer feedback about the lack of experimental evaluation of GA parameters. It systematically tests different configurations to demonstrate the robustness of the GA design.

## Files Created

1. **`scripts/ga_sensitivity_analysis.py`** - Main analysis script
2. **`scripts/notebook_cells_ga_sensitivity.py`** - Template cells for the notebook
3. **`data/Results/sensitivity/`** - Output directory for results

## How to Use

### Option 1: Run from Command Line

```bash
cd c:\Users\ignag\OneDrive\Documentos\TSLP_algeciras
python scripts/ga_sensitivity_analysis.py
```

This will:
- Run 27 GA experiments (3 population sizes × 3 crossover probs × 3 trials)
- Take approximately 10-15 minutes
- Generate CSV results and PNG visualizations in `data/Results/sensitivity/`

### Option 2: Run from Notebook

1. Open `notebooks/01_experimentos.ipynb`
2. Create a new section titled "GA Parameter Sensitivity Analysis"
3. Copy the cells from `scripts/notebook_cells_ga_sensitivity.py` into the notebook
4. Run the cells sequentially

The notebook cells will:
- Import the analysis function
- Run the sensitivity analysis
- Display summary statistics
- Show comparison plots

## Parameters Tested

| Parameter | Values Tested | Default |
|-----------|---------------|---------|
| Population Size | 30, 50, 100 | 50 |
| Crossover Probability | 0.6, 0.8, 0.9 | 0.8 |
| Random Seeds | 42, 43, 44 | 42 |

**Fixed parameters** (to isolate effects):
- Mutation probability: 0.3
- Tournament size: 2
- Max generations: 30
- Sensor budget: 20

## Outputs

### CSV Results
`data/Results/sensitivity/ga_sensitivity_results.csv` contains:
- Configuration details (pop_size, p_crossover, trial, seed)
- Performance metrics (best_fitness, avg_fitness, convergence_gen, runtime)

### Visualizations

1. **`convergence_comparison.png`**
   - Left: Convergence curves by population size
   - Right: Convergence curves by crossover probability
   - Shows mean ± std across trials

2. **`fitness_boxplots.png`**
   - Left: Final fitness distribution by population size
   - Right: Final fitness distribution by crossover probability
   - Shows variability and outliers

3. **`parameter_heatmaps.png`**
   - Three heatmaps showing parameter interactions:
     - Mean best fitness
     - Mean convergence generation
     - Mean runtime

## Expected Results

Based on GA theory, you should observe:

- **Population size**: Larger populations explore more but take longer
- **Crossover probability**: Higher values promote diversity but may slow convergence
- **Trade-offs**: No single "best" configuration; depends on priorities (quality vs speed)

## Interpreting Results

### Good Signs
✓ Different configurations show measurable performance differences
✓ Results are relatively stable across trials (low std)
✓ Convergence curves show improvement over generations
✓ Trade-offs are visible (e.g., larger pop = better fitness but slower)

### Red Flags
✗ All configurations perform identically (suggests parameter insensitivity)
✗ Very high variance across trials (suggests instability)
✗ No convergence improvement (suggests poor GA design)

## Customization

To test additional parameters, modify `ga_sensitivity_analysis.py`:

```python
# Add more values to test
population_sizes = [20, 30, 50, 100, 200]
crossover_probs = [0.5, 0.6, 0.7, 0.8, 0.9]
mutation_probs = [0.1, 0.3, 0.5]  # Currently fixed at 0.3

# Increase trials for more statistical power
n_trials = 5  # Default is 3
```

## Notes

- The analysis uses the **surrogate fitness** (fast approximation) for all evaluations
- Each run saves its solution to `data/processed/selected/`
- The script is designed to be non-invasive (no modifications to existing .py files)
- Results are reproducible due to fixed random seeds

## Citation

When reporting these results in your paper, you can reference:

> "We conducted a sensitivity analysis of the GA parameters, testing 3 population sizes (30, 50, 100) and 3 crossover probabilities (0.6, 0.8, 0.9) with 3 random seeds each (27 total configurations). Results showed that [describe key findings], validating the robustness of our parameter choices."
