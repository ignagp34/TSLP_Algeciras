# Sensor Error Parameter (α) Implementation

## Summary

Added **sensor error parameter α = 0.999** to the MILP model and GA algorithm, implementing constraints (8) and (9) from the paper.

## Changes Made

### 1. `src/milp_model.py`

**Added parameter:**
- `alpha_sensor_error: float = 0.999` (0.1% sensor error)

**Modified constraints (lines 220-235):**

**Before:**
```python
# r >= f - hat - M (1 - x)
r_var >= f_var - hat - M * (1 - x_var)

# r >= hat - f - M (1 - x)
r_var >= hat - f_var - M * (1 - x_var)
```

**After (with sensor error):**
```python
# Constraint (8): r >= α*f - f_prior - M (1 - x)
r_var >= alpha_sensor_error * f_var - hat - M * (1 - x_var)

# Constraint (9): r >= f_prior - α*f - M (1 - x)
r_var >= hat - alpha_sensor_error * f_var - M * (1 - x_var)
```

### 2. `src/ga_algorithm.py`

**Added parameter to TrafficNetwork class:**
- `alpha_sensor_error: float = 0.999`
- Stored as instance variable: `self.alpha_sensor_error`
- Passed to exact MILP model during evaluation

### 3. `scripts/run_ga.py`

**Updated TrafficNetwork initialization:**
```python
domain = TrafficNetwork(
    ...
    alpha_sensor_error=0.999,
)
```

## Interpretation

### What does α = 0.999 mean?

- **α < 1**: Sensor can **underestimate** flow
- **α = 0.999**: Sensor has **0.1% error** (very accurate)
- The reconstructed flow `f` is compared against `α * f_prior` instead of `f_prior`

### Typical values:
- **α = 1.0**: Perfect sensor (no error) - **previous implementation**
- **α = 0.999**: 0.1% error - **current implementation**
- **α = 0.95**: 5% error (more realistic for real sensors)
- **α = 1.05**: 5% overestimation

## Impact

This change makes the model more realistic by accounting for sensor measurement uncertainty. The residual constraints now allow for a small tolerance (0.1%) between the reconstructed flow and the sensor measurements.

## Next Steps

If you want to test different sensor error levels, you can modify the `alpha_sensor_error` parameter:
- For more accurate sensors: increase α (e.g., 0.9999 = 0.01% error)
- For less accurate sensors: decrease α (e.g., 0.95 = 5% error)
