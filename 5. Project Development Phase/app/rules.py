from dataclasses import dataclass, field


@dataclass
class RuleResult:
    hard_decline: bool = False
    reasons: list = field(default_factory=list)   # hard-decline reasons
    soft_flags: list = field(default_factory=list)  # soft warnings, don't auto-decline


MIN_WORKING_AGE = 18
MAX_PLAUSIBLE_AGE = 100
MAX_PLAUSIBLE_YEARS_EMPLOYED = 60
LOW_INCOME_PER_PERSON = 30000  # below the training data's effective floor, per family member


def evaluate(raw_input: dict) -> RuleResult:
    """raw_input uses the same keys app.py already builds: AMT_INCOME_TOTAL,
    AGE_YEARS, YEARS_EMPLOYED, IS_EMPLOYED, CNT_FAM_MEMBERS, CNT_CHILDREN,
    NAME_INCOME_TYPE."""
    result = RuleResult()

    income = raw_input.get("AMT_INCOME_TOTAL", 0)
    age = raw_input.get("AGE_YEARS", 0)
    years_employed = raw_input.get("YEARS_EMPLOYED", 0)
    is_employed = raw_input.get("IS_EMPLOYED", 0)
    income_type = raw_input.get("NAME_INCOME_TYPE", "")
    fam_members = raw_input.get("CNT_FAM_MEMBERS", 1) or 1

    # --- Hard declines: inputs the model was never trained on, no ML call needed ---
    if income <= 0:
        result.hard_decline = True
        result.reasons.append("No verifiable income reported (income must be greater than 0).")

    if age < MIN_WORKING_AGE:
        result.hard_decline = True
        result.reasons.append(f"Applicant age below minimum eligible age ({MIN_WORKING_AGE}).")

    if age > MAX_PLAUSIBLE_AGE:
        result.hard_decline = True
        result.reasons.append("Applicant age outside plausible range.")

    if years_employed < 0 or years_employed > MAX_PLAUSIBLE_YEARS_EMPLOYED:
        result.hard_decline = True
        result.reasons.append("Employment duration outside plausible range.")

    if years_employed > 0 and (age - years_employed) < MIN_WORKING_AGE:
        result.hard_decline = True
        result.reasons.append("Reported employment duration is inconsistent with reported age.")

    # An applicant claiming to be actively employed (income type = "Working"
    # and IS_EMPLOYED=1) but with 0 years employed is a soft inconsistency,
    # not necessarily disqualifying (could be a brand-new hire) - flag it.
    if not result.hard_decline:
        if income_type == "Working" and is_employed and years_employed == 0:
            result.soft_flags.append("Applicant reports active employment with 0 years of tenure.")

        income_per_person = income / fam_members
        if income_per_person < LOW_INCOME_PER_PERSON:
            result.soft_flags.append(
                f"Income per family member (~{income_per_person:,.0f}) is below the typical "
                f"range seen in the training population."
            )

    return result
