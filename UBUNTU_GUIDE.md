# Ubuntu validation guide

StochX does not require EViews to run. EViews is used only to obtain reference inputs/outputs for Step 13 validation.

## Important
This repository must never generate placeholder or random replacement data for an EViews benchmark. Missing reference data must cause a clear error or an explicit skipped validation.

## Denmark
The public CRAN ARDL dataset provides the Danish money-demand data used in the Johansen literature:
LRM, LRY, LPY, IBO, IDE.

Use:
\
python3 scripts/prepare_eviews_reference_data.py
\

The resulting validation data stay outside the package source tree.

## Uroot
For the Uroot benchmark, obtain the actual EViews example data or an export containing:
date, CS, GDP

Required sample:
1948Q3–1988Q4

Equation:
CS C GDP CS(-1)

## EViews output
Keep exported EViews output as local validation artifacts:
validation_data/eviews/

Do not commit proprietary EViews workfiles unless redistribution rights are clear.

## Step 13 workflow
1. Prepare public/reference data.
2. Put EViews numeric output in validation_data/eviews/.
3. Run the opt-in parity tests.
4. Investigate specification mismatches before changing tolerances.
