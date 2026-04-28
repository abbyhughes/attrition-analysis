import pandas as pd
import pytest
from metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5, 6],
        "department": ["Sales", "Sales", "HR", "HR", "IT", "IT"],
        "attrition": ["Yes", "No", "Yes", "No", "Yes", "Yes"],
        "overtime": ["Yes", "No", "Yes", "No", "Yes", "No"],
        "monthly_income": [3000.0, 6000.0, 4000.0, 7000.0, 5000.0, 8000.0],
        "job_satisfaction": [1, 3, 2, 4, 2, 3],
    })


# --- attrition_rate ---

def test_attrition_rate_partial(sample_df):
    assert attrition_rate(sample_df) == 66.67


def test_attrition_rate_no_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


def test_attrition_rate_all_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


# --- attrition_by_department ---

def test_attrition_by_department_columns(sample_df):
    result = attrition_by_department(sample_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_rates(sample_df):
    result = attrition_by_department(sample_df)
    rates = dict(zip(result["department"], result["attrition_rate"]))
    assert rates["IT"] == 100.0
    assert rates["Sales"] == 50.0
    assert rates["HR"] == 50.0


def test_attrition_by_department_sorted_descending(sample_df):
    result = attrition_by_department(sample_df)
    assert result.iloc[0]["department"] == "IT"
    assert list(result["attrition_rate"]) == sorted(result["attrition_rate"], reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_rates(sample_df):
    result = attrition_by_overtime(sample_df)
    rates = dict(zip(result["overtime"], result["attrition_rate"]))
    assert rates["Yes"] == 100.0   # all 3 overtime workers left
    assert rates["No"] == 33.33    # 1 of 3 non-overtime workers left


# --- average_income_by_attrition ---

def test_average_income_by_attrition_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values(sample_df):
    result = average_income_by_attrition(sample_df)
    income = dict(zip(result["attrition"], result["avg_monthly_income"]))
    assert income["Yes"] == 5000.0   # (3000+4000+5000+8000) / 4
    assert income["No"] == 6500.0    # (6000+7000) / 2


# --- satisfaction_summary ---

def test_satisfaction_summary_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_rates(sample_df):
    result = satisfaction_summary(sample_df)
    rates = dict(zip(result["job_satisfaction"], result["attrition_rate"]))
    assert rates[1] == 100.0   # 1 of 1 left
    assert rates[2] == 100.0   # 2 of 2 left
    assert rates[3] == 50.0    # 1 of 2 left
    assert rates[4] == 0.0     # 0 of 1 left


def test_satisfaction_summary_sorted_ascending(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result["job_satisfaction"]) == sorted(result["job_satisfaction"])
