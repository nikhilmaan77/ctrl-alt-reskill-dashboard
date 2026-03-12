# Ctrl+Alt+Reskill — Global Workforce Reskilling Gap Intelligence Dashboard

A WEF/ILO-grade analytical dashboard built on 10,000 synthetic survey responses, designed to present actionable reskilling policy insights to government ministers, industry leaders, and international policymakers. Features dark/light mode toggle.

## Dashboard Structure (8 Tabs)

| Tab | Title | Method | Key Deliverable |
|-----|-------|--------|----------------|
| 1 | Executive Summary | KPI Overview | Headline stats, global choropleth, persona & matrix thumbnails |
| 2 | Who Successfully Reskills? | Classification (Gradient Boosting + SHAP) | Feature importance, What-If Simulator, policy levers |
| 3 | Five Global Workforce Personas | Clustering (K-Means, k=5) | Radar profiles, country × persona heatmap, persona cards |
| 4 | Behavioural Patterns | Association Rule Mining (Apriori) | Network graph, Sankey flow, policy bundles |
| 5 | Drivers of Willingness & Anxiety | Regression (OLS) | Coefficient plots, multi-country comparison, inverted-U insight |
| 6 | Geographic Intelligence | Cross-country Comparison | Choropleth, bubble chart, country profiles with personas |
| 7 | The Gender Gap | Gender Comparison | Distribution plots, barrier gaps, willingness-vs-access |
| 8 | Income-Reskilling Matrix | 2×2 Framework | Quadrant scatter, treemap, segment policy recommendations |

Every tab follows a 3-layer structure: (1) Analytical Visualization → (2) Model Validation → (3) Policy Implications.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Ensure `global_reskilling_gap_survey_10k.csv` is in the same directory as `app.py`.

## Files

| File | Description |
|------|-------------|
| `app.py` | Single-file Streamlit dashboard (all 8 tabs) |
| `global_reskilling_gap_survey_10k.csv` | 10,000-row synthetic dataset with 38 variables |
| `data_dictionary_reskilling_gap.md` | Comprehensive variable documentation |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Embedded Data Patterns (What Models Should Discover)

1. Mid-level roles face highest automation exposure (non-linear)
2. Inverted-U between career anxiety and reskilling action
3. Employer support is strongly industry-dependent
4. Cost barrier dominates developing countries; time dominates developed
5. Younger workers overestimate career confidence
6. **KEY:** Employer support predicts reskilling success far more than education
7. Gender gap in reskilling access vs. equal willingness

## Dataset

- **10,000** synthetic survey respondents across **12 countries**
- **31 raw survey questions** + **6 derived composite variables**
- PPP-adjusted income brackets per country
- Realistic noise: 6% straightlining, 1–3% missing values, central tendency bias on Likert scales
- Seed: 42 (reproducible)

## Target Audience

Government ministers, WEF/ILO board members, industry policymakers — not data scientists. Every insight is translated into actionable policy recommendations.

---

*SP Jain School of Global Management — Consulting Project*
