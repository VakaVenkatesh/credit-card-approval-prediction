# Credit Card Approval Prediction System

Machine learning-powered web application to predict credit card applicant risk using **Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost** classifiers.

## Team

- Vaka Venkatesh
- Ragi Prakash
- Sontineni Sai Krishna Chowdari
- Yarra Vivek

## Folder Structure

| Folder | Contents |
|---|---|
| `1-4` | Ideation, requirement analysis, design & planning documentation (Problem Statements, Empathy Map, Customer Journey Map, Data Flow Diagram, Solution Requirements, Technology Stack, Solution Architecture) |
| `5.Project Development Phase` | All actual code: dataset, notebook, trained model, Flask app |
| `6-8` | Testing, documentation, and demonstration deliverables |
| `docs` | ER Diagram, Data Flow Diagram, and Architecture references |

## Best Model

**XGBoost (tuned)** — Test ROC-AUC: `0.717`, F1 (high-risk class): `0.27`

## How to Run

```bash
cd "5.Project Development Phase/app"
pip install -r requirements.txt
python app.py
```
