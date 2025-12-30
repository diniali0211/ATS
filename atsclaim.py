import io, re, calendar
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
from difflib import SequenceMatcher

st.set_page_config(page_title="Sustio Claim Builder", layout="wide")
st.title("📊 ATS Claim Builder")

st.markdown(
    "- Attendance codes: **M, N, M8, N8, RN8, RM8, ON8, PM8, PN8** (and variants)\n"
    "- Claim window can be the default 16th→15th month or a custom date range\n"
    "- **Transportation** and **Shift** are ignored\n"
    "- Matching order: **Worker No (raw & digits)** → **Name (exact)** → **Name (fuzzy)**\n"
    "- **Eligibility cap**: days **outside** [`JOIN_DATE`, `JOIN_DATE+3 months−1 day`] are set to **0**"
)

# ───────────────────────────── Constants ─────────────────────────────
# Keep these for heuristic guesses; actual behavior is driven by the sidebar mapping
# Updated to include OM8, PH variants, and OD(M)/OD(N) forms so heuristic mode treats them as present/8H.
BASE_PRESENT = {
    "M","N","M8","N8","RN8","RM8","ON8","PM8","PN8",
    "PN","PM","RN","RM","MR","NR","NR8","MR8","ON",
    "OM8",         # treat OM8 as present/8H
    "PH","PH(M)","PH(N)",  # make PH variants count as present
    "OD(M)","OD(N)"        # make OD(M)/OD(N) count as present
}
MARK_8H = {
    "M8","N8","RN8","RM8","ON8","PM8","PN8",
    "OM8",            # OM8 flagged 8H
    "PH","PH(M)","PH(N)"  # PH variants considered 8H by default (adjust if needed)
}

# ───────────────────────────── Helpers ─────────────────────────────
def norm_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("\u00a0"," ", regex=False).str.strip()

def clean_code_standard(v) -> str:
    """Upper, stripped code for normalization (used for matching)."""
    if pd.isna(v): return ""
    return str(v).strip().upper().replace(" ", "").replace(".", "")

def make_keys(series: pd.Series):
    raw   = norm_text(series).str.upper()
    digit = raw.str.replace(r"\D","", regex=True).str.lstrip("0")
    return raw, digit

def normalize_name(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.upper()
         .str.replace(r"[^A-Z0-9 ]"," ", regex=True)
         .str.replace(r"\s+"," ", regex=True)
         .str.strip()
    )

def parse_join_date(series_like) -> pd.Series:
    s = pd.Series(series_like).copy()
    s = s.replace({0: np.nan, 0.0: np.nan, "0": np.nan, "0000-00-00": np.nan,
                   "NONE": np.nan, "NaN": np.nan, "nan": np.nan})
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    num = pd.to_numeric(s, errors="coerce")
    serial_mask = dt.isna() & num.notna()
    if serial_mask.any():
        dt.loc[serial_mask] = pd.to_datetime(num.loc[serial_mask], unit="D", origin="1899-12-30")
    bad_mask = dt.notna() & (dt.dt.year < 1990)
    dt.loc[bad_mask] = pd.NaT
    return dt

def add_eligible_end_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "JOIN_DATE" in out.columns:
        eligible = pd.to_datetime(out["JOIN_DATE"], errors="coerce") + pd.offsets.DateOffset(months=3) - pd.Timedelta(days=1)
        eligible = eligible.where(~out["JOIN_DATE"].isna())
        insert_at = out.columns.get_loc("JOIN_DATE") + 1
        out.insert(insert_at, "Eligible End Date", eligible)
    return out

def reorder_day_cols(cols, window_start=None, window_end=None):
    cols = list(dict.fromkeys(cols))
    parsed = {}
    for c in cols:
        try:
            dt = pd.to_datetime(c, errors="coerce")
            parsed[c] = dt.normalize() if pd.notna(dt) else pd.NaT
        except Exception:
            parsed[c] = pd.NaT

    if any(pd.notna(v) for v in parsed.values()):
        items = []
        for c, dt in parsed.items():
            if pd.isna(dt):
                try:
                    iv = int(str(c).strip())
                    if window_start is not None and window_end is not None:
                        rng = pd.date_range(window_start, window_end, freq="D")
                        match = rng[rng.day == iv]
                        items.append((c, match[0]) if len(match) else (c, pd.NaT))
                    else:
                        items.append((c, pd.NaT))
                except Exception:
                    items.append((c, pd.NaT))
            else:
                items.append((c, dt))
        items_sorted = sorted(items, key=lambda x: (pd.isna(x[1]), x[1]))
        return [c for c,_ in items_sorted]
    else:
        nums = []
        for c in cols:
            try:
                n = int(str(c).strip())
                if 1 <= n <= 31: nums.append(n)
            except: pass
        nums = sorted(set(nums))
        return [str(d) for d in range(16,32) if d in nums] + [str(d) for d in range(1,16) if d in nums]

def cap_days_by_window(df: pd.DataFrame, day_cols, cycle_end=None, window_start=None, window_end=None):
    d = df.copy()

    if window_start is None or window_end is None:
        if cycle_end is None:
            raise ValueError("Either cycle_end or window_start/window_end must be provided")
        cycle_end_ts   = pd.Timestamp(cycle_end.year, cycle_end.month, 15)
        cycle_start_dt = cycle_end_ts - pd.offsets.MonthBegin(1)
        prev_month_end = cycle_start_dt - pd.Timedelta(days=1)
        window_start   = pd.Timestamp(prev_month_end.year, prev_month_end.month, 16)
        window_end     = cycle_end_ts
    else:
        window_start = pd.to_datetime(window_start)
        window_end   = pd.to_datetime(window_end)

    join = pd.to_datetime(d["JOIN_DATE"], errors="coerce")
    elig = join + pd.offsets.DateOffset(months=3) - pd.Timedelta(days=1)

    for c in day_cols:
        actual = pd.to_datetime(c, errors="coerce")
        if pd.isna(actual):
            try:
                day = int(str(c))
                rng = pd.date_range(window_start, window_end, freq="D")
                matches = rng[rng.day == day]
                actual = matches[0] if len(matches) else pd.NaT
            except Exception:
                actual = pd.NaT

        if pd.isna(actual):
            if c in d.columns:
                d[c] = 0
            continue

        col = str(c)
        if col not in d.columns:
            continue

        mask_before = join.notna() & (join > actual)
        mask_after  = elig.notna() & (elig < actual)
        d.loc[mask_before | mask_after, col] = 0

    present_day_cols = [c for c in day_cols if c in d.columns]
    if present_day_cols:
        d["Total Working Days"] = d[present_day_cols].sum(axis=1).astype(int)
    else:
        d["Total Working Days"] = 0
    return d

# ───────────────── Attendance parsing ─────────────────
def detect_attendance(att_file):
    raw = pd.read_excel(att_file, sheet_name=0, header=None)
    header_idx = 0; found_emp=False; found_name=False
    for i in range(min(25, len(raw))):
        vals = norm_text(raw.iloc[i]).str.lower().tolist()
        has_emp = any(v in {"emp no","employee no","worker no","emp id","employee id","no pekerja","id"} for v in vals)
        has_nam = any(v in {"name","worker name","employee name","nama"} for v in vals)
        if has_emp and has_nam:
            header_idx = i; found_emp=True; found_name=True; break

    day_row_idx = None
    for j in range(header_idx+1, min(header_idx+8, len(raw))):
        vals = norm_text(raw.iloc[j])
        if vals.str.fullmatch(r"\d{1,2}").sum() >= 6:
            day_row_idx = j; break

    if day_row_idx is not None:
        df = pd.read_excel(att_file, sheet_name=0, header=[header_idx, day_row_idx])
        flat = []
        for t,b in df.columns:
            ts, bs = str(t).strip(), str(b).strip()
            if re.fullmatch(r"\d{1,2}", bs): flat.append(bs)
            elif re.fullmatch(r"\d{1,2}", ts): flat.append(ts)
            else: flat.append(ts if ts and ts.lower()!="nan" else bs)
        df.columns = flat
    else:
        df = pd.read_excel(att_file, sheet_name=0, header=header_idx)

    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = norm_text(pd.Index(df.columns))

    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl in {"emp no","employee no","worker no","emp id","employee id","no pekerja","id"}: ren[c] = "Worker No"
        elif cl in {"name","worker name","employee name","nama"}: ren[c] = "Worker Name"
        elif cl in {"joined date","join date","date join","date joined","tarikh masuk","doj"}: ren[c] = "JOIN_DATE"
        elif cl in {"transportation","transport"}: ren[c] = "_drop_transport"
        elif cl in {"shift"}: ren[c] = "_drop_shift"
    if ren: df = df.rename(columns=ren)
    df = df[[c for c in df.columns if not str(c).startswith("_drop_")]]

    # convert headers that are day numbers OR date-like to standardized names (ISO for dates)
    conv = {}
    for c in df.columns:
        try:
            iv = int(str(c).strip())
            if 1 <= iv <= 31:
                conv[c] = str(iv)
                continue
        except Exception:
            pass

        try:
            ts = pd.to_datetime(c, errors="coerce", dayfirst=False)
            if pd.notna(ts):
                conv[c] = ts.strftime("%Y-%m-%d")
                continue
            ts2 = pd.to_datetime(c, errors="coerce", dayfirst=True)
            if pd.notna(ts2):
                conv[c] = ts2.strftime("%Y-%m-%d")
                continue
        except Exception:
            pass

    if conv:
        df = df.rename(columns=conv)

    for col in ["Worker No","Worker Name","JOIN_DATE"]:
        if col not in df.columns: df[col] = pd.NA

    day_cols = [c for c in df.columns if (str(c).isdigit() or (isinstance(c, str) and re.match(r"\d{4}-\d{2}-\d{2}", c)))]
    return df, day_cols, dict(header_idx=header_idx, day_row_idx=day_row_idx,
                              found_emp=found_emp, found_name=found_name)

# ───────────────── Masterlist normalization ─────────────────
def normalize_masterlist_auto(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = norm_text(pd.Index(d.columns))

    ren = {}
    for c in d.columns:
        cl = c.lower()
        if cl in {
            "emp id","empid","employee id","employeeid",
            "worker id","worker no","emp no","employee no",
            "no pekerja","no.pekerja","id","emp id no","employee id no"
        }:
            ren[c] = "Worker No"
        elif cl in {"name","worker name","employee name","nama","nama pekerja"}:
            ren[c] = "Worker Name"
        elif cl in {"join date","joined date","date joined","date of join","doj","tarikh masuk","date joined"}:
            ren[c] = "JOIN_DATE"
        elif cl in {"recruiter","recruiter name","consultant","agent","pic","pic recruiter","consultant name","recuiter"}:
            ren[c] = "Recruiter"
    d = d.rename(columns=ren)
    return d

def apply_masterlist_mapping(raw_df: pd.DataFrame, map_cols: dict) -> pd.DataFrame:
    d = raw_df.copy()
    for std, sel in map_cols.items():
        d[std] = d[sel] if sel in d.columns else pd.NA

    d["Worker No"]   = norm_text(d["Worker No"]).str.upper()
    d["Worker Name"] = norm_text(d["Worker Name"])
    d["JOIN_DATE"]   = parse_join_date(d["JOIN_DATE"])
    d["Recruiter"]   = norm_text(d["Recruiter"]).replace({"": pd.NA}) if "Recruiter" in d.columns else pd.NA
    d["NAME_KEY"]    = normalize_name(d["Worker Name"])
    d["WORKER_NO_KEY_RAW"], d["WORKER_NO_KEY_DIGIT"] = make_keys(d["Worker No"])

    keep = ["Worker No","Worker Name","JOIN_DATE","Recruiter",
            "NAME_KEY","WORKER_NO_KEY_RAW","WORKER_NO_KEY_DIGIT"]
    for k in keep:
        if k not in d.columns: d[k] = pd.NA
    return d[keep].copy()

# ───────────────── Matching & enrichment (defensive) ─────────────────
def enrich_and_filter(pres_df: pd.DataFrame, master: pd.DataFrame, day_cols,
                      keep_only_master=True, fuzzy_threshold=0.90):
    df = pres_df.copy()

    required_cols = [
        "Worker No","Worker Name","JOIN_DATE","Recruiter",
        "NAME_KEY","WORKER_NO_KEY_RAW","WORKER_NO_KEY_DIGIT",
        "Total Working Days","Worker_Type"
    ]
    for rc in required_cols:
        if rc not in df.columns:
            df[rc] = pd.NA

    valid_raw   = set(master.get("WORKER_NO_KEY_RAW", pd.Series([], dtype=object)).dropna().astype(str))
    valid_digit = set(master.get("WORKER_NO_KEY_DIGIT", pd.Series([], dtype=object)).dropna().astype(str))
    valid_name  = set(master.get("NAME_KEY", pd.Series([], dtype=object)).dropna().astype(str))

    mask = df["WORKER_NO_KEY_RAW"].astype(str).isin(valid_raw) | \
           df["WORKER_NO_KEY_DIGIT"].astype(str).isin(valid_digit) | \
           df["NAME_KEY"].astype(str).isin(valid_name)

    unmatched = df[~mask].copy()
    if keep_only_master:
        df = df[mask].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if "WORKER_NO_KEY_RAW" in master.columns:
        lut_raw_rec = master.drop_duplicates("WORKER_NO_KEY_RAW").set_index("WORKER_NO_KEY_RAW")["Recruiter"]
        lut_raw_jdt = master.drop_duplicates("WORKER_NO_KEY_RAW").set_index("WORKER_NO_KEY_RAW")["JOIN_DATE"]
    else:
        lut_raw_rec = pd.Series(dtype=object)
        lut_raw_jdt = pd.Series(dtype="datetime64[ns]")

    if "WORKER_NO_KEY_DIGIT" in master.columns:
        lut_dig_rec = master.drop_duplicates("WORKER_NO_KEY_DIGIT").set_index("WORKER_NO_KEY_DIGIT")["Recruiter"]
        lut_dig_jdt = master.drop_duplicates("WORKER_NO_KEY_DIGIT").set_index("WORKER_NO_KEY_DIGIT")["JOIN_DATE"]
    else:
        lut_dig_rec = pd.Series(dtype=object)
        lut_dig_jdt = pd.Series(dtype="datetime64[ns]")

    if "NAME_KEY" in master.columns:
        lut_name_rec = master.drop_duplicates("NAME_KEY").set_index("NAME_KEY")["Recruiter"]
        lut_name_jdt = master.drop_duplicates("NAME_KEY").set_index("NAME_KEY")["JOIN_DATE"]
    else:
        lut_name_rec = pd.Series(dtype=object)
        lut_name_jdt = pd.Series(dtype="datetime64[ns]")

    rec = df["WORKER_NO_KEY_RAW"].map(lut_raw_rec)\
            .fillna(df["WORKER_NO_KEY_DIGIT"].map(lut_dig_rec))\
            .fillna(df["NAME_KEY"].map(lut_name_rec))
    jdt = df["WORKER_NO_KEY_RAW"].map(lut_raw_jdt)\
            .fillna(df["WORKER_NO_KEY_DIGIT"].map(lut_dig_jdt))\
            .fillna(df["NAME_KEY"].map(lut_name_jdt))

    need_fuzzy = rec.isna()
    if need_fuzzy.any() and ("NAME_KEY" in master.columns) and len(master):
        ml_names = master["NAME_KEY"].dropna().unique().tolist()
        ml_rec   = master.drop_duplicates("NAME_KEY").set_index("NAME_KEY")["Recruiter"]
        ml_jdt   = master.drop_duplicates("NAME_KEY").set_index("NAME_KEY")["JOIN_DATE"]

        def best_match(name):
            if not isinstance(name, str) or not name: return (None, None)
            best_ratio = 0.0; best_key = None
            for mk in ml_names:
                r = SequenceMatcher(a=name, b=mk).ratio()
                if r > best_ratio:
                    best_ratio = r; best_key = mk
            if best_ratio >= fuzzy_threshold and best_key is not None:
                return (ml_rec.get(best_key, pd.NA), ml_jdt.get(best_key, pd.NaT))
            return (None, None)

        matched = df.loc[need_fuzzy, "NAME_KEY"].apply(best_match)
        rec.loc[need_fuzzy] = [t[0] for t in matched]
        jdt.loc[need_fuzzy] = [t[1] for t in matched]

    df["Recruiter"] = rec
    df["JOIN_DATE"] = jdt

    cols = ["Worker No","Worker Name","JOIN_DATE","Recruiter"] + [c for c in day_cols if c in df.columns] + ["Total Working Days","Worker_Type"]
    cols = [c for c in cols if c in df.columns]
    return df[cols], unmatched

# ───────────────── Presence builder (configurable) ─────────────────
def build_presence(att_df: pd.DataFrame, day_cols, present_codes=None, mark_8h_codes=None):
    if present_codes is None: present_codes = []
    if mark_8h_codes is None: mark_8h_codes = []

    # normalize present/8h sets to the same canonical form as cell normalization
    present_set = set([str(x).strip().upper().replace(" ", "").replace(".", "") for x in present_codes])
    mark8_set = set([str(x).strip().upper().replace(" ", "").replace(".", "") for x in mark_8h_codes])

    d = att_df.copy()
    d["Worker No"]   = norm_text(d.get("Worker No", pd.Series(dtype=object))).str.upper()
    d["Worker Name"] = norm_text(d.get("Worker Name", pd.Series(dtype=object)))
    d["NAME_KEY"]    = normalize_name(d["Worker Name"])
    d["WORKER_NO_KEY_RAW"], d["WORKER_NO_KEY_DIGIT"] = make_keys(d["Worker No"])

    has_8h = pd.Series(False, index=d.index)
    for c in day_cols:
        if c in d.columns:
            # normalize cell to canonical code for matching (strip spaces/dots)
            col_vals = d[c].astype(str).fillna("").str.strip().str.upper().str.replace(" ", "").str.replace(".", "")
            d[c] = col_vals.isin(present_set).astype(int)
            has_8h = has_8h | col_vals.isin(mark8_set)
        else:
            d[c] = 0

    d["Worker_Type"] = np.where(has_8h, "8H", "12H")
    existing = [c for c in day_cols if c in d.columns]
    if existing:
        d["Total Working Days"] = d[existing].sum(axis=1).astype(int)
    else:
        d["Total Working Days"] = 0
    return d

def recruiter_summary(df_all: pd.DataFrame, day_cols, rate=3):
    if df_all.empty:
        return pd.DataFrame([{"Recruiter":"TOTAL","Days":0,"Rate (RM)":rate,"Amount (RM)":0}])
    tmp = df_all.copy()
    tmp["Days"] = tmp[day_cols].sum(axis=1).astype(int)
    tmp["Recruiter"] = tmp["Recruiter"].fillna("Unassigned")
    grp = tmp.groupby("Recruiter", dropna=False)["Days"].sum().reset_index()
    grp["Rate (RM)"] = rate
    grp["Amount (RM)"] = grp["Days"] * rate
    total = pd.DataFrame([{
        "Recruiter":"TOTAL",
        "Days": int(grp["Days"].sum()),
        "Rate (RM)": rate,
        "Amount (RM)": int(grp["Amount (RM)"].sum())
    }])
    return pd.concat([grp.sort_values("Days", ascending=False).reset_index(drop=True), total], ignore_index=True)

# ───────────────────────────── UI ─────────────────────────────
st.sidebar.header("⚙️ Options")
rate_rm = st.sidebar.number_input("Per-day rate (RM)", min_value=0, value=3, step=1)
debug_mode = st.sidebar.checkbox("Debug / show parsing details", value=True)
bypass_master = st.sidebar.checkbox("Bypass Masterlist filtering (show all attendance rows)", value=False)
fuzzy_thr = st.sidebar.slider("Fuzzy name match threshold", min_value=0.70, max_value=0.99, value=0.90, step=0.01)
exclude_unassigned_export = st.sidebar.checkbox("Exclude 'Unassigned' from download (Claim & Summary)", value=True)

def month_year_picker(label="Claim month (ends on 15th)"):
    today = date.today()
    years = [today.year - 1, today.year, today.year + 1]
    months = list(range(1, 13))
    sel_year  = st.sidebar.selectbox(f"{label} — Year", years, index=years.index(today.year))
    sel_month = st.sidebar.selectbox(f"{label} — Month", [calendar.month_abbr[m] for m in months],
                                     index=today.month - 1)
    sel_month_num = months[[calendar.month_abbr[m] for m in months].index(sel_month)]
    return date(sel_year, sel_month_num, 15)

use_custom_window = st.sidebar.checkbox("Use custom claim window (start/end)", value=True)

if use_custom_window:
    default_start = date(2025, 10, 8)
    default_end   = date(2025, 11, 6)
    window_start_date = st.sidebar.date_input("Window start (inclusive)", value=default_start)
    window_end_date   = st.sidebar.date_input("Window end (inclusive)", value=default_end)
    window_start = pd.Timestamp(window_start_date)
    window_end   = pd.Timestamp(window_end_date)
else:
    cycle_end = month_year_picker()
    cycle_end_ts   = pd.Timestamp(cycle_end.year, cycle_end.month, 15)
    cycle_start_dt = cycle_end_ts - pd.offsets.MonthBegin(1)
    prev_month_end = cycle_start_dt - pd.Timedelta(days=1)
    window_start   = pd.Timestamp(prev_month_end.year, prev_month_end.month, 16)
    window_end     = cycle_end_ts

st.info(f"Claim window: **{window_start.date()} → {window_end.date()}**")

att_file = st.file_uploader("📄 Attendance (xlsx/xls)", type=["xlsx","xls"])
ml_file  = st.file_uploader("📇 Masterlist (xlsx/xls)", type=["xlsx","xls"])

if att_file and ml_file:
    with st.spinner("Processing..."):
        # Attendance
        att_df, day_cols, meta = detect_attendance(att_file)

        # --- detect unique codes and show mapping UI in sidebar ---
        detected_codes = []
        if day_cols:
            sample = att_df[day_cols].astype(str).fillna("").applymap(lambda v: v.strip().upper())
            flattened = pd.Series(sample.values.ravel())
            flattened = flattened[flattened != ""]
            detected_codes = sorted(flattened.unique().tolist())

        def norm_code(c):
            return str(c).strip().upper().replace(" ", "").replace(".", "")

        detected_codes_norm = [norm_code(c) for c in detected_codes]

        # Defaults derived from your legend
        # — Make OD(M) and OD(N) count as 1 day.
        # — Make PH and PH variants count as 1 day as well.
        # — Include OM8 as 8H present by default.
        default_present = {
            "NM", "PH", "PH(N)", "PH(M)",
            "N", "M",
            "RD(M)", "RD(N)", "RD(NM)", "RPH(M)", "RPH(N)",
            "OD(M)", "OD(N)",   # ← New: OD variants count as present
            "OM8"               # ← New: OM8 treated as present
        }
        default_8h = {"NM", "PH", "RD(NM)", "RPH(M)", "RPH(N)", "OM8"}

        guessed_present = [c for c in detected_codes_norm if c in default_present]
        guessed_8h = [c for c in guessed_present if c in default_8h]

        # build mapping normalized->originals for UI labels
        norm_to_original = {}
        for orig in detected_codes:
            n = norm_code(orig)
            norm_to_original.setdefault(n, []).append(orig)

        sidebar_options = []
        for n in sorted(set(detected_codes_norm)):
            examples = ", ".join(norm_to_original.get(n, [n])[:3])
            sidebar_options.append((n, f"{n}  ({examples})"))
        st.sidebar.markdown("### Attendance code mapping (choose how to treat codes)")
        if detected_codes:
            labels = [lbl for _, lbl in sidebar_options]
            label_to_norm = {lbl: key for key, lbl in sidebar_options}

            default_labels_present = [lbl for lbl in labels if label_to_norm[lbl] in guessed_present]
            selected_labels = st.sidebar.multiselect("Codes to treat as PRESENT (select all that mean worker present)", options=labels, default=default_labels_present)

            present_codes = [label_to_norm[lbl] for lbl in selected_labels]

            labels_8h_options = [lbl for lbl in labels if label_to_norm[lbl] in present_codes]
            default_labels_8h = [lbl for lbl in labels_8h_options if label_to_norm[lbl] in guessed_8h]
            selected_labels_8h = st.sidebar.multiselect("Of the PRESENT codes, mark which are 8H", options=labels_8h_options, default=default_labels_8h)

            mark_8h_codes = [label_to_norm[lbl] for lbl in selected_labels_8h]
        else:
            st.sidebar.write("No day columns detected yet — upload file first.")
            present_codes = []
            mark_8h_codes = []

        # Reorder day_cols considering the chosen window
        day_cols = reorder_day_cols(day_cols, window_start=window_start, window_end=window_end)

        # Masterlist (auto-normalize + mapping UI)
        xls = pd.ExcelFile(ml_file)
        ml_sheet = st.selectbox("Masterlist sheet", xls.sheet_names, index=0)
        master_raw = pd.read_excel(xls, sheet_name=ml_sheet)
        auto_ml = normalize_masterlist_auto(master_raw)

        st.markdown("**Masterlist column mapping (adjust only if auto is wrong):**")
        cols_list = list(auto_ml.columns) or ["— no columns —"]

        def guess(cands, default=None):
            for c in cands:
                if c in cols_list: return c
            return default if default in cols_list else cols_list[0]

        sel_worker_no = st.selectbox("→ Worker No column", cols_list, index=cols_list.index(guess(["Worker No","Emp id","Emp ID","EMP ID"])) if cols_list else 0)
        sel_worker_nm = st.selectbox("→ Worker Name column", cols_list, index=cols_list.index(guess(["Worker Name","Name"])) if cols_list else 0)
        sel_join_date = st.selectbox("→ JOIN_DATE column", cols_list, index=cols_list.index(guess(["JOIN_DATE","Date joined","Joined Date"])) if cols_list else 0)
        sel_recruiter = st.selectbox("→ Recruiter column", cols_list, index=cols_list.index(guess(["Recruiter","Recuiter","Recruiter Name"])) if cols_list else 0)

        masterlist = apply_masterlist_mapping(
            auto_ml,
            {"Worker No": sel_worker_no, "Worker Name": sel_worker_nm,
             "JOIN_DATE": sel_join_date, "Recruiter": sel_recruiter}
        )

        # Build presence & match
        pres = build_presence(att_df, day_cols, present_codes=present_codes, mark_8h_codes=mark_8h_codes) if day_cols else pd.DataFrame()
        claim_all, unmatched = enrich_and_filter(
            pres, masterlist, day_cols,
            keep_only_master=(not bypass_master),
            fuzzy_threshold=fuzzy_thr
        )

        # Cap days by explicit window start/end, then add Eligible End to preview
        claim_all = cap_days_by_window(claim_all, day_cols, window_start=window_start, window_end=window_end)
        claim_all = add_eligible_end_date(claim_all)

        # Diagnostics
        if debug_mode:
            with st.expander("📇 Masterlist diagnostics", expanded=False):
                st.write(f"Total rows in masterlist: **{len(masterlist)}**")
                nn = masterlist[["Worker No","Worker Name","JOIN_DATE","Recruiter"]].notna().sum()
                st.write(f"Non-null → Worker No: **{nn['Worker No']}**, Worker Name: **{nn['Worker Name']}**, JOIN_DATE: **{nn['JOIN_DATE']}**, Recruiter: **{nn['Recruiter']}**")
                st.dataframe(masterlist.head(25), use_container_width=True)

            with st.expander("🔎 Attendance parsing diagnostics", expanded=False):
                st.write(f"Detected header row: **{meta['header_idx']}**  |  Day header row: **{meta['day_row_idx']}**")
                st.write(f"Day columns ({len(day_cols)}): {day_cols}")
                st.write(f"Attendance rows (raw): **{len(att_df)}**")
                preview_n = st.slider("How many attendance rows to preview?", 5, 200, 50, 5)
                st.dataframe(att_df.head(preview_n), use_container_width=True)

        # On-screen preview (combined; shows Eligible End Date)
        st.subheader(f"Claim — preview ({len(claim_all)})")
        preview_cols = ["Worker No","Worker Name","JOIN_DATE","Eligible End Date","Recruiter"] + day_cols + ["Total Working Days","Worker_Type"]
        preview_cols = [c for c in preview_cols if c in claim_all.columns]
        st.dataframe(claim_all[preview_cols].head(50), use_container_width=True)

        st.subheader("👥 Per-Recruiter Summary")
        rec_sum = recruiter_summary(claim_all.drop(columns=["Worker_Type"], errors="ignore"), day_cols, rate=rate_rm)
        st.dataframe(rec_sum, use_container_width=True)

        # ---------- Build export (optionally excluding Unassigned) ----------
        if exclude_unassigned_export:
            excl_mask = claim_all["Recruiter"].isna() | (claim_all["Recruiter"].astype(str).str.strip() == "") | (claim_all["Recruiter"] == "Unassigned")
            claim_export = claim_all[~excl_mask].copy()
        else:
            claim_export = claim_all.copy()

        rec_sum_export = recruiter_summary(claim_export.drop(columns=["Worker_Type"], errors="ignore"), day_cols, rate=rate_rm)
        if exclude_unassigned_export and not rec_sum_export.empty:
            body = rec_sum_export[(rec_sum_export["Recruiter"] != "Unassigned") & (rec_sum_export["Recruiter"] != "TOTAL")].copy()
            total_row = pd.DataFrame([{
                "Recruiter": "TOTAL",
                "Days": int(body["Days"].sum()) if len(body) else 0,
                "Rate (RM)": int(body["Rate (RM)"].iloc[0]) if len(body) else rate_rm,
                "Amount (RM)": int(body["Amount (RM)"].sum()) if len(body) else 0,
            }])
            rec_sum_export = pd.concat([body, total_row], ignore_index=True)

        # ---------- Write Excel (single combined sheet) ----------
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as wr:
            ordered_cols = ["Worker No","Worker Name","JOIN_DATE","Eligible End Date","Recruiter"] + day_cols + ["Total Working Days","Worker_Type"]
            ordered_cols = [c for c in ordered_cols if c in claim_export.columns]
            claim_export[ordered_cols].to_excel(wr, index=False, sheet_name="Claim")
            rec_sum_export.to_excel(wr, index=False, sheet_name="Recruiter_Summary")

        st.download_button(
            f"⬇️ Download Claim Report (Combined + Recruiter Summary){' — Unassigned Excluded' if exclude_unassigned_export else ''}",
            data=out.getvalue(),
            file_name="sustio_claim_report_project_one.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # Optional unmatched download
        if debug_mode and len(unmatched):
            buf = io.BytesIO()
            unmatched.to_excel(buf, index=False, sheet_name="Unmatched")
            st.download_button("⬇️ Download unmatched list", buf.getvalue(),
                file_name="unmatched_attendance.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("Upload both files to generate the claim report.")








