# HBSIR PCA — Household-level Workflow (Private)

This Streamlit app runs PCA **per year** on **household-level** expenditure features built at a chosen commodity classification level. Area and settlement are only filters; PCA is always on households.

## Features
- Year selection (single or multiple).
- Area filter: whole country or selected provinces/counties, plus settlement (Urban/Rural/Both).
- Commodity level picker (1–5), no manual feature picking.
- Multiple preprocessing options: Raw, Budget shares, CLR, Log(1+x), etc.
- Weighted PCA (survey weights) or unweighted.
- Per-year outputs saved to disk (scores+household info, loadings for **all** PCs, variance table).
- Plotting section: Scores / Loadings / 2D–3D Biplot; flexible aggregation (Household / County / Province).

## Prerequisites
- **Python 3.9–3.12**  
- Install dependencies:

### pip
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
