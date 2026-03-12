# Data Dictionary: Global Workforce Reskilling Gap Survey

**Dataset:** `global_reskilling_gap_survey_10k.csv`
**Records:** 10,000 synthetic survey respondents
**Variables:** 38 (31 raw survey + 1 ID + 6 derived)
**Countries:** 12 (weighted by reskilling discourse prominence)
**Purpose:** Classification, Clustering, Association Rule Mining, Regression analysis for consulting deliverable

---

## 1. Respondent Identifier

| # | Variable | Type | Range/Values | Description |
|---|----------|------|-------------|-------------|
| — | `respondent_id` | String | R00001–R10000 | Unique survey respondent identifier |

---

## 2. Demographic Variables (Q1–Q9, Q30)

| Q# | Variable | Type | Range/Values | Description | Generation Logic |
|----|----------|------|-------------|-------------|-----------------|
| Q1 | `age` | Integer | 20–62 | Respondent age | Country-specific normal distributions (e.g., India μ=32 σ=7, Japan μ=40 σ=10), clipped at 20–62 |
| Q2 | `gender` | Categorical | Male, Female, Non-Binary, Prefer Not to Say | Gender identity | Weighted: 52% M, 40% F, 5% NB, 3% PNTS. ~1% missing injected |
| Q3 | `country` | Categorical | 12 countries | Country of residence | Weighted: India(2025), USA(1658), UK(1103), UAE(889), Germany(676), Brazil(661), Nigeria(552), South Africa(550), Japan(548), Australia(452), Singapore(448), Philippines(438) |
| Q4 | `education_level` | Ordinal | Below High School → PhD | Highest education attained | Country-specific distributions; developed countries skew toward Bachelor's/Master's |
| Q5 | `employment_status` | Categorical | 6 categories | Current employment status | Age-correlated: under-25 more students, over-55 more unemployed |
| Q6 | `industry` | Categorical | 12 industries | Industry of employment/most recent role | Developing countries: higher Agriculture, lower Tech/Finance weighting |
| Q7 | `job_role_level` | Ordinal | Entry-level → Executive/C-suite | Seniority level | Age-correlated with non-linear distribution; students/unemployed default to Entry-level |
| Q8 | `work_experience_years` | Integer | 0–40 | Total work experience | Derived from age minus education completion age, with ±2 year noise |
| Q9 | `income_bracket` | Categorical | 6 PPP-adjusted brackets per country | Annual income range | **Localized PPP brackets** (e.g., India: <₹3L to >₹50L; USA: <$25K to >$200K). Correlated with education, role level, and experience. ~3% missing with bias toward higher earners |
| Q30 | `household_dependents` | Integer | 0–4 | Number of dependents | Age-correlated: under-25 skew toward 0; 35–50 peak at 2–3. ~2% missing |

---

## 3. Automation Exposure Variables (Q10–Q13)

| Q# | Variable | Type | Range/Values | Description | Generation Logic |
|----|----------|------|-------------|-------------|-----------------|
| Q10 | `perceived_automation_risk` | Likert 1–5 | 1 (Very Low) → 5 (Very High) | How at-risk respondent feels from automation | **BIAS 1:** Mid-level roles score highest (non-linear with seniority). Industry-specific base rates (Tech/Manufacturing highest). Central tendency bias applied |
| Q11 | `role_partially_automated` | Binary | Yes / No | Whether role has already been partially automated | Probability increases with Q10 score; correlation ~0.6 with noise |
| Q12 | `pct_tasks_automatable` | Ordinal (bucketed) | 0–20%, 21–40%, 41–60%, 61–80%, 81–100% | Estimated share of daily tasks that could be automated | Distribution shifts with Q10 and Q11; high-risk respondents skew toward higher buckets |
| Q13 | `ai_awareness` | Likert 1–5 | 1 (Not Aware) → 5 (Very Aware) | Awareness of AI/automation trends in respondent's industry | Tech/Finance industries +0.6 boost; higher education +0.2 per level; under-30 gets +0.3 (**BIAS 5** component) |

---

## 4. Reskilling Behavior Variables (Q14–Q21)

| Q# | Variable | Type | Range/Values | Description | Generation Logic |
|----|----------|------|-------------|-------------|-----------------|
| Q14 | `enrolled_reskilling` | Binary | Yes / No | Currently enrolled in any reskilling/upskilling program | Base ~25%; boosted by high automation risk, younger age, student/job-seeking status, developed country |
| Q15 | `learning_platform` | Categorical | 10 options including "None" | Primary learning platform used | Conditional on Q14; developed countries favor Coursera/LinkedIn Learning; developing favor Udemy/YouTube. Non-enrolled get 80% "None" |
| Q16 | `upskilling_hours_per_week` | Float | 0–25 | Weekly hours spent on upskilling activities | Exponential distribution for enrolled (μ≈5); full-time employed get 0.6x multiplier; students/seekers get 1.4x. 15% of non-enrolled have 0.5–3 hours (informal learning) |
| Q17 | `willingness_to_reskill` | Likert 1–5 | 1 (Not Willing) → 5 (Very Willing) | Willingness to completely reskill into a new field | **BIAS 2:** Inverted-U with career anxiety — moderate anxiety (3–4) yields highest willingness; extreme anxiety (5) yields lower (paralysis); low anxiety (1) yields lowest (complacency) |
| Q18 | `biggest_barrier` | Categorical | Cost, Time, Lack of Awareness, Family Responsibilities, No Relevant Programs, Employer Doesn't Support | Single biggest barrier to reskilling | **BIAS 4:** Cost dominates developing countries; Time dominates developed. **BIAS 7:** Female respondents 1.6x more likely to cite Family Responsibilities |
| Q19 | `preferred_learning_mode` | Categorical | 6 modes | Preferred format for learning/training | Weighted: Online Self-paced (30%), Hybrid (18%), Online Instructor-led (15%), On-the-job (15%), In-person (12%), Peer Learning (10%) |
| Q20 | `top_skill_pursued` | Categorical | 10 options including "None" | Primary skill currently being pursued | Industry-correlated (Tech → Data/AI; Healthcare → Soft Skills/Leadership). "None" for non-enrolled with 0 upskilling hours |
| Q21 | `reskilling_awareness_source` | Categorical | 7 sources including "None" | How respondent learned about reskilling opportunities | Age-correlated: under-30 favor Social Media/Peers; over-45 favor Employer/News |

---

## 5. Support Ecosystem Variables (Q22–Q25)

| Q# | Variable | Type | Range/Values | Description | Generation Logic |
|----|----------|------|-------------|-------------|-----------------|
| Q22 | `employer_provides_reskilling` | Binary | Yes / No | Whether employer offers reskilling opportunities | **BIAS 3:** Industry-dependent — Tech (1.6x), Finance (1.4x) vs. Agriculture (0.5x), Construction (0.6x). Country base rates apply. Unemployed/students default to "No" |
| Q23 | `govt_reskilling_subsidies` | Categorical | Yes / No / Unsure | Whether government subsidies for reskilling are available | Country-specific: Singapore highest "Yes" (45%), Nigeria lowest (8%). High "Unsure" rates across all countries |
| Q24 | `satisfaction_employer_ld` | Likert 1–5 | 1 (Very Dissatisfied) → 5 (Very Satisfied) | Satisfaction with employer's Learning & Development investment | Correlated with Q22 (support = +1.3 base shift). Tech/Finance employers score higher. **BIAS 7:** Female respondents −0.2 |
| Q25 | `would_switch_for_reskilling` | Binary | Yes / No | Willingness to change employers for better reskilling support | Driven by Q24 dissatisfaction (+30% if ≤2); younger workers +10%; high dependents −10% |

---

## 6. Psychographic Variables (Q26–Q29)

| Q# | Variable | Type | Range/Values | Description | Generation Logic |
|----|----------|------|-------------|-------------|-----------------|
| Q26 | `career_confidence_5yr` | Likert 1–5 | 1 (Not Confident) → 5 (Very Confident) | Confidence that current career path remains relevant in 5 years | **BIAS 5:** Under-30 gets +0.7 overconfidence boost. High automation risk reduces confidence. Enrolled in reskilling +0.3 |
| Q27 | `career_anxiety` | Likert 1–5 | 1 (Very Low) → 5 (Very High) | General anxiety about career future | Peaks in mid-career (35–50, +0.4). High automation risk +0.8. High dependents +0.3 |
| Q28 | `accept_lower_pay_futureproof` | Binary | Yes / No | Willingness to accept lower pay for a more future-proof role | Driven by combined anxiety + willingness; constrained by dependents count |
| Q29 | `expected_role_change_timeline` | Integer | 1–15 years | Expected years before current role changes significantly | Inversely correlated with automation risk; Tech/Finance −1 year shift |

---

## 7. Target Variable (Q31)

| Q# | Variable | Type | Range/Values | Description | Generation Logic |
|----|----------|------|-------------|-------------|-----------------|
| Q31 | `successful_reskill_transition` | Binary | Yes / No | Successfully transitioned to a new role/skill domain in the past 3 years | **BIAS 6 (KEY):** Logistic model — employer support (β=1.2) and upskilling hours (β=0.12) are strongest predictors. Education has intentionally weak effect (β=0.05). Age < 35 gives +0.3 boost. Distribution: ~34% Yes, ~66% No |

---

## 8. Derived Variables (Computed Post-Survey)

| # | Variable | Type | Range | Formula | Analytical Purpose |
|---|----------|------|-------|---------|-------------------|
| DV1 | `automation_vulnerability_idx` | Float | 0–1 | 0.4×norm(Q10) + 0.3×Q11_binary + 0.3×Q12_midpoint | Composite automation exposure score for regression and clustering segmentation axis |
| DV2 | `reskilling_engagement_score` | Float | 0–1 | 0.25×Q14_binary + 0.30×norm(Q16, max=20) + 0.25×norm(Q17) + 0.20×Q20_binary | Measures active reskilling behavior intensity; key clustering dimension |
| DV3 | `support_ecosystem_score` | Float | 0–1 | 0.35×Q22_binary + 0.30×Q23_mapped(Y=1,U=0.5,N=0) + 0.35×norm(Q24) | Structural support environment; differentiates clusters by external enablement |
| DV4 | `anxiety_to_action_ratio` | Float | 0–10 | norm(Q27) / (DV2 + 0.01) | High ratio = paralyzed (high anxiety, low action). Low ratio = proactive. Consulting persona identifier |
| DV5 | `income_reskilling_gap` | Categorical | 4 segments + Unknown | Cross of income tier (top/bottom 3 brackets) × hours (≥5 = high) | "Complacent High Earner", "Invested High Earner", "Striving Upskiller", "At-Risk Low Earner" — HR consulting segmentation |
| DV6 | `future_readiness_idx` | Float | 0–1 | 0.40×norm(Q26) + 0.35×norm(Q17) + 0.25×inverse_norm(Q29, max=15) | Overall preparedness composite; regression target alternative |

---

## 9. Embedded Biases Summary

| Bias | Description | Variables Involved | Expected Model Finding |
|------|-------------|-------------------|----------------------|
| **Bias 1** | Mid-level roles face highest automation exposure (non-linear) | Q7 → Q10, Q11, Q12 | Classification/regression: job_role_level has non-linear feature importance |
| **Bias 2** | Inverted-U relationship between anxiety and reskilling action | Q27 → Q17, DV2, DV4 | Clustering: "Paralyzed" segment (high anxiety, low engagement) vs. "Motivated" segment |
| **Bias 3** | Employer support is strongly industry-dependent | Q6 → Q22, Q24 | Association rules: {Tech, Employer Support=Yes} → {High Satisfaction} |
| **Bias 4** | Cost is top barrier in developing countries; Time in developed | Q3 tier → Q18 | Association rules: {Developing, Low Income} → {Cost Barrier} |
| **Bias 5** | Younger workers overestimate career confidence despite lower engagement | Q1 → Q26 vs. DV2 | Regression: age-confidence paradox visible in coefficient signs |
| **Bias 6** | Successful reskilling driven by employer support + hours, NOT education | Q22, Q16 → Q31 | **KEY FINDING:** Classification SHAP values show employer_support >> education_level |
| **Bias 7** | Gender gap in reskilling access and barrier type | Q2 → Q18, Q24 | Clustering/association: Female segment shows higher Family Responsibilities barrier but equal willingness |

---

## 10. Noise & Data Quality Features

| Feature | Implementation | Purpose |
|---------|---------------|---------|
| **Straightlining** | ~6% of respondents answer same value (3 or 4) across all Likert scales | Simulates survey satisficing; data quality flag opportunity |
| **Missing values** | Income: ~3% (biased toward high earners); Dependents: ~2%; Gender: ~1% | Realistic survey non-response; preprocessing discussion point |
| **Central tendency** | Likert responses pulled toward center; only ~12% hit extremes (1 or 5) | Realistic survey response behavior |
| **Correlation noise** | All deliberate biases have 15–20% noise overlay | Prevents over-clean patterns; ensures realistic model performance |

---

## 11. Recommended Analytical Approaches

| Analysis | Target/Focus | Key Variables | Expected Insight |
|----------|-------------|---------------|-----------------|
| **Classification** | Q31 (successful transition) | All demographics + DV1, DV2, DV3 | Employer support and hours invested predict success more than education |
| **Clustering** | Respondent segmentation | DV1, DV2, DV3, DV4, DV6 | 4–5 personas: Future-Ready, Anxious-Paralyzed, Structurally Unsupported, Automation-Exposed & Active, Complacent |
| **Association Rules** | Behavioral patterns | Q6, Q18, Q15, Q20, Q22 (categorical) | Industry × barrier × platform × skill combinations |
| **Regression** | Q27 or Q17 as continuous target | Demographics + automation exposure + support | What drives reskilling willingness and career anxiety |

---

*Dataset generated with numpy random seed 42 for reproducibility.*
*All income brackets are PPP-adjusted to local currency for survey realism.*
