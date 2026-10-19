"""Unit tests for data parsing and cleaning."""
import numpy as np
import pandas as pd
import pytest

from heartml import config
from heartml.data import clean, parse_mirror, parse_uci


def test_parse_uci_marks_question_marks_as_missing():
    raw = b"63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n" \
          b"67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,?,3.0,2\n"
    df = parse_uci(raw)
    assert list(df.columns) == config.UCI_COLUMNS
    assert np.isnan(df.loc[1, "ca"])
    assert df.loc[1, "num"] == 2


def test_parse_mirror_maps_text_codes_to_uci_codes():
    raw = (b",Age,Sex,ChestPain,RestBP,Chol,Fbs,RestECG,MaxHR,ExAng,Oldpeak,Slope,Ca,Thal,AHD\n"
           b"1,63,1,typical,145,233,1,2,150,0,2.3,3,0,fixed,No\n"
           b"2,67,1,asymptomatic,160,286,0,2,108,1,1.5,2,NA,reversable,Yes\n")
    df = parse_mirror(raw)
    assert df["cp"].tolist() == [1, 4]
    assert df["thal"].tolist() == [6, 7]
    assert df["num"].tolist() == [0, 1]
    assert df["ca"].isna().sum() == 1


def test_clean_binarises_target(raw_df, clean_df):
    assert set(clean_df[config.TARGET].unique()) <= {0, 1}
    assert clean_df[config.TARGET].sum() == (raw_df["num"] > 0).sum()
    assert "num" not in clean_df.columns


def test_clean_keeps_schema_and_order(clean_df):
    assert list(clean_df.columns) == config.FEATURES + [config.TARGET]


def test_clean_out_of_range_values_become_missing(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "chol"] = 0        # impossible cholesterol
    bad.loc[1, "trestbps"] = 999  # impossible blood pressure
    bad.loc[2, "thal"] = 5        # undefined code
    out = clean(bad)
    assert pd.isna(out.loc[0, "chol"])
    assert pd.isna(out.loc[1, "trestbps"])
    assert pd.isna(out.loc[2, "thal"])


def test_clean_drops_duplicates_and_unlabelled_rows(raw_df):
    dup = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)
    dup.loc[len(dup)] = [50, 1, 2, 120, 200, 0, 0, 150, 0, 1.0, 1, 0, 3, np.nan]
    out = clean(dup)
    assert len(out) == len(raw_df.drop_duplicates())


def test_clean_coerces_junk_strings(raw_df):
    junk = raw_df.astype(object).copy()
    junk.loc[0, "age"] = "?"
    out = clean(junk)
    assert pd.isna(out.loc[0, "age"])


def test_clean_rejects_missing_columns(raw_df):
    with pytest.raises(ValueError, match="Missing expected columns"):
        clean(raw_df.drop(columns=["chol"]))
