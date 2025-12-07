import csv
import json
import os
import re
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI


INDUSTRIES: List[str] = [
    "Rolnictwo",
    "Przetwórstwo",
    "Energetyka",
    "Budownictwo",
    "Handel",
    "Transport",
    "HoReCa",
    "IT",
    "Finanse",
]

MODEL_NAME = "gpt-5"


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Brak zmiennej środowiskowej OPENAI_API_KEY.")
    return OpenAI(api_key=api_key)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "rok": "year",
        "year": "year",
        "years": "year",
        "branza": "industry",
        "branża": "industry",
        "nazwa_pkd": "industry",
        "numer_nazwa_pkd": "industry",
        "sector": "industry",
        "pkd": "pkd_code",
        "value": "value",
        "wartosc": "value",
        "wartość": "value",
        "rentownosc": "value",
        "rentowność": "value",
        "ratio": "value",
        "wskaznik": "indicator",
    }
    renamed = {}
    for col in df.columns:
        key = col.replace("\ufeff", "").replace("\xa0", " ").strip().lower()
        renamed[col] = col_map.get(key, col)
    renamed_df = df.rename(columns=renamed)
    # po mapowaniu różne nagłówki mogą zlać się w jedną nazwę (np. branza+industry)
    return renamed_df.loc[:, ~renamed_df.columns.duplicated()]


def build_profitability_df(df: pd.DataFrame) -> pd.DataFrame:
    # Identyfikacja kolumn
    industry_col = next((c for c in ["industry", "nazwa_pkd", "numer_nazwa_pkd", "pkd", "pkd_code"] if c in df.columns), None)
    indicator_col = next((c for c in ["indicator", "wskaznik"] if c in df.columns), None)
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]

    if not (industry_col and indicator_col and year_cols):
        cols = ", ".join(df.columns)
        raise ValueError(f"Plik wymaga kolumn: industry/NAZWA_PKD, WSKAZNIK oraz kolumn lat. Znalezione nagłówki: {cols}")

    # melt: zachowujemy branżę i wskaźnik, rozciągamy lata
    long_df = df.melt(id_vars=[industry_col, indicator_col], value_vars=year_cols, var_name="year", value_name="raw_value")

    # czyszczenie liczb
    raw_series = long_df["raw_value"].astype(str).str.replace("\s", "", regex=True).str.replace("\xa0", "", regex=False).str.replace(",", ".")
    long_df["amount"] = pd.to_numeric(raw_series, errors="coerce")
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    long_df = long_df.dropna(subset=["amount", "year"])

    # wybór wierszy dla NP i GS
    mask_np = long_df[indicator_col].str.contains("np wynik finansowy netto", case=False, na=False)
    mask_gs = long_df[indicator_col].str.contains("gs przychody ogolem", case=False, na=False) | long_df[indicator_col].str.contains("gs przychody ogółem", case=False, na=False)

    np_df = long_df[mask_np]
    gs_df = long_df[mask_gs]

    if np_df.empty or gs_df.empty:
        raise ValueError("Brak wymaganych wskaźników: 'NP Wynik finansowy netto' lub 'GS Przychody ogółem'.")

    # pivot by industry/year
    np_pivot = np_df.pivot_table(index=[industry_col, "year"], values="amount", aggfunc="first")
    gs_pivot = gs_df.pivot_table(index=[industry_col, "year"], values="amount", aggfunc="first")

    joined = np_pivot.join(gs_pivot, how="inner", lsuffix="_np", rsuffix="_gs").reset_index()
    joined.rename(columns={"amount_np": "np", "amount_gs": "gs"}, inplace=True)

    # rentowność %
    joined["value"] = (joined["np"] / joined["gs"]).replace([pd.NA, pd.NaT], 0).fillna(0) * 100
    joined = joined[[industry_col, "year", "value"]]

    # ostatnie 5 lat dostępnych
    latest_year = joined["year"].max()
    joined = joined[joined["year"] >= latest_year - 4]

    # ujednolicone nazwy kolumn
    return joined.rename(columns={industry_col: "industry"}).dropna(subset=["year", "value"])


def normalize_industry_value(industry: str) -> str:
    """Map pełnych nazw sekcji PKD do krótkich etykiet używanych w INDUSTRIES."""
    if not isinstance(industry, str):
        return industry

    name = industry.strip().lower()

    mapping = {
        "rolnictwo": "Rolnictwo",
        "rolnictwo, leśnictwo, łowiectwo i rybactwo": "Rolnictwo",
        "przetwórstwo przemysłowe": "Przetwórstwo",
        "wytwarzanie i zaopatrywanie w energię elektryczną": "Energetyka",
        "dostawa energii elektrycznej": "Energetyka",
        "zaopatrywanie w parę": "Energetyka",
        "budownictwo": "Budownictwo",
        "handel hurtowy i detaliczny": "Handel",
        "handel; naprawa pojazdów samochodowych": "Handel",
        "naprawa pojazdów samochodowych": "Handel",
        "transport i gospodarka magazynowa": "Transport",
        "działalność związana z zakwaterowaniem i usługami gastronomicznymi": "HoReCa",
        "zakwaterowanie i gastronomia": "HoReCa",
        "informacja i komunikacja": "IT",
        "telekomunikacja": "IT",
        "działalność finansowa i ubezpieczeniowa": "Finanse",
        "usługi finansowe": "Finanse",
        "działalność w zakresie obsługi rynku nieruchomości": "Nieruchomości",
        "obsługa rynku nieruchomości": "Nieruchomości",
    }

    for key, target in mapping.items():
        if key in name:
            # dopasowujemy po fragmencie tekstu (contains), zwracamy nazwę z listy INDUSTRIES
            return target

    return industry


def convert_wide_years(df: pd.DataFrame) -> pd.DataFrame:
    # wykryj kolumny będące latami YYYY i przetop na long
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    has_year = any(col.lower() == "year" for col in df.columns)
    if has_year or not year_cols:
        return df

    industry_col = None
    for cand in ["industry", "pkd", "nazwa_pkd", "numer_nazwa_pkd"]:
        if cand in df.columns:
            industry_col = cand
            break

    value_name = "value"

    if industry_col is None:
        return df  # brak kolumny identyfikującej branżę

    melted = df.melt(id_vars=[industry_col], value_vars=year_cols, var_name="year", value_name=value_name)
    melted["year"] = pd.to_numeric(melted["year"], errors="coerce")
    return melted


def pick_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    year_candidates = [c for c in df.columns if c.lower() in ["year", "rok"]]
    industry_candidates = [c for c in df.columns if c.lower() in ["industry", "branza", "branża", "pkd", "sector", "nazwa_pkd", "numer_nazwa_pkd"]]
    value_candidates = [c for c in df.columns if c.lower() in ["value", "rentownosc", "rentowność", "ratio", "wartosc", "wartość", "wskaznik"]]
    if not (year_candidates and industry_candidates and value_candidates):
        cols = ", ".join(df.columns)
        raise ValueError(f"Plik musi zawierać kolumny: rok/year, branza/industry, rentowność/value. Znalezione nagłówki: {cols}")
    return year_candidates[0], industry_candidates[0], value_candidates[0]


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Brak pliku: {path}")
    sample_bytes = 4096
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(sample_bytes)

    try:
        dialect = csv.Sniffer().sniff(sample)
        sep = dialect.delimiter
    except Exception:
        sep = None  # pozwól pandasowi zgadnąć

    df = pd.read_csv(path, sep=sep, engine="python", on_bad_lines="warn")
    # usuń zduplikowane nazwy kolumn, zachowując pierwsze wystąpienie
    df = df.loc[:, ~df.columns.duplicated()]
    df = normalize_columns(df)
    return df


def compute_baseline(fin_df: pd.DataFrame, year_col: str, industry_col: str, value_col: str) -> Dict[str, float]:
    latest_year = fin_df[year_col].max()
    short_window = fin_df[fin_df[year_col] >= latest_year - 2]  # momentum: ostatnie 3 lata
    long_window = fin_df  # regresja do średniej: cały horyzont

    baseline = {}
    for industry in INDUSTRIES:
        short_subset = short_window[short_window[industry_col] == industry]
        long_subset = long_window[long_window[industry_col] == industry]

        short_mean = float(short_subset[value_col].mean()) if not short_subset.empty else 0.0
        long_mean = float(long_subset[value_col].mean()) if not long_subset.empty else 0.0

        # Metoda Hybrydowa (Safe Bet): większa waga momentum, mniejsza kotwica długoterminowa
        baseline[industry] = 0.7 * short_mean + 0.3 * long_mean

    return baseline


def ensure_state(baseline: Dict[str, float]) -> None:
    if "forecast" not in st.session_state:
        st.session_state.baseline = baseline.copy()
        st.session_state.forecast = baseline.copy()
        st.session_state.prev_forecast = baseline.copy()
    if "last_delta" not in st.session_state:
        st.session_state.last_delta = {}


def render_chart(fin_df: pd.DataFrame, industry: str, year_col: str, industry_col: str, value_col: str) -> None:
    latest_year = fin_df[year_col].max()
    history = fin_df[(fin_df[industry_col] == industry) & (fin_df[year_col] >= latest_year - 4)][[year_col, value_col]].copy()
    history = history.rename(columns={year_col: "year", value_col: "value"})
    history = history.loc[:, ~history.columns.duplicated()]
    if history.empty or set(history.columns) != {"year", "value"}:
        st.info(f"Brak danych do wykresu dla branży: {industry}")
        return

    forecast_year = 2025  # wymagane na wykresie jako kreska
    forecast_val = st.session_state.forecast.get(industry, 0.0)  # AI po newsie
    prev_val = st.session_state.prev_forecast.get(industry, forecast_val)
    baseline_val = st.session_state.baseline.get(industry, forecast_val)

    fig = px.line(history, x="year", y="value", markers=True, title=industry)

    last_known = history.sort_values("year").iloc[-1]["value"] if not history.empty else baseline_val

    # Safe Bet (hybryda) do 2025
    fig.add_scatter(
        x=[history["year"].max(), forecast_year],
        y=[last_known, baseline_val],
        mode="lines+markers",
        name="Safe Bet 2025",
        line=dict(color="orange", dash="dash"),
        marker=dict(size=9, color="orange"),
    )

    # Prognoza AI z promptu (po newsie) – osobna linia
    fig.add_scatter(
        x=[history["year"].max(), forecast_year],
        y=[last_known, forecast_val],
        mode="lines+markers",
        name="AI (news) 2025",
        line=dict(color="#7e3ff2", dash="solid"),
        marker=dict(size=10, color="#7e3ff2"),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Delta: {forecast_val - prev_val:+.2f}")


def analyze_news_impact(client: OpenAI, news_text: str) -> Dict[str, float]:
    system_prompt = ""  # zostaw pusty, jeśli nie chcesz narzucać dodatkowych instrukcji

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append(
        {
            "role": "user",
            "content": f"""Jesteś Analitykiem Ryzyka. Oceń wpływ poniższego newsa na rentowność 9 branż.

DANE WEJŚCIOWE:
Branże: [Rolnictwo, Przetwórstwo, Energetyka, Budownictwo, Handel, Transport, HoReCa, IT, Finanse]
News: "{news_text}"

ZASADY OCENY:
1. Skala: od -5.0 (krytyczne zagrożenie) do +5.0 (silny wzrost). 0.0 = brak wpływu.
2. Im mniej wiadomo jak dany news wpływa na branże, tym mniej na nią wpływa.
3. skacz z oceną co 0.01 punkt procentowy, ostrożnie zmieniaj punkty względem zera.

WYMAGANY FORMAT (Czysty JSON), branż może być wiele:
    "Nazwa_Branży": <float_zmiana>,
""",
        }
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    parsed = json.loads(content)

    # normalizuj nazwy branż zwracane przez model do etykiet INDUSTRIES
    def norm_key(name: str) -> str:
        if not isinstance(name, str):
            return ""
        lowered = name.strip().lower()
        mapping = {
            "rolnictwo": "Rolnictwo",
            "przetwórstwo": "Przetwórstwo",
            "przetworstwo": "Przetwórstwo",
            "energetyka": "Energetyka",
            "budownictwo": "Budownictwo",
            "handel": "Handel",
            "handel detaliczny": "Handel",
            "handel hurtowy": "Handel",
            "transport": "Transport",
            "horeca": "HoReCa",
            "ho reca": "HoReCa",
            "zakwaterowanie i gastronomia": "HoReCa",
            "it": "IT",
            "technologie informacyjne": "IT",
            "finanse": "Finanse",
            "finanse i ubezpieczenia": "Finanse",
        }
        return mapping.get(lowered, "")

    normalized = {}
    for k, v in parsed.items():
        nk = norm_key(k)
        if nk in INDUSTRIES:
            normalized[nk] = float(v)

    return normalized


def apply_adjustment(delta: Dict[str, float]) -> None:
    st.session_state.prev_forecast = st.session_state.forecast.copy()
    for industry, change in delta.items():
        current = st.session_state.forecast.get(industry, 0.0)
        st.session_state.forecast[industry] = current + change


def main() -> None:
    st.set_page_config(page_title="Dashboard Analityczny", layout="wide")
    st.title("Przewidywania rentowności wybranych branż na rok 2025")

    st.sidebar.header("News")
    news_text = st.sidebar.text_area("Wklej news do analizy", height=200)
    st.sidebar.subheader("Ostatni wynik AI")
    st.sidebar.json(st.session_state.get("last_delta", {}))

    try:
        raw_df = load_csv("wsk_fin.csv")
        fin_df = build_profitability_df(raw_df)
        year_col, industry_col, value_col = "year", "industry", "value"
        # mapuj długie nazwy PKD do krótkich etykiet z INDUSTRIES
        fin_df[industry_col] = fin_df[industry_col].apply(normalize_industry_value)
        fin_df = fin_df[fin_df[industry_col].isin(INDUSTRIES)]
        # agreguj duplikujące się sekcje PKD zmapowane do jednej etykiety
        fin_df = (
            fin_df.groupby([industry_col, year_col], as_index=False)[value_col]
            .mean(numeric_only=True)
        )
    except Exception as e:
        st.error(f"Nie można wczytać wsk_fin.csv: {e}")
        return

    if fin_df.empty:
        st.error("Brak danych po oczyszczeniu kolumn. Sprawdź format pliku.")
        return

    available = sorted(set(fin_df[industry_col].unique().tolist()))
    st.caption(f"Dostępne branże w danych: {', '.join(available) if available else 'brak'}")

    baseline = compute_baseline(fin_df, year_col, industry_col, value_col)
    ensure_state(baseline)

    available = set(fin_df[industry_col].unique().tolist())
    missing = [i for i in INDUSTRIES if i not in available]
    if missing:
        st.warning(f"Brak danych dla branż: {', '.join(missing)}")

    # Najpierw obsłuż kliknięcie, żeby wykresy w tym samym przebiegu miały już nowe prognozy
    if st.button("Analizuj wpływ newsa", type="primary"):
        if not news_text.strip():
            st.warning("Wklej treść newsa.")
        else:
            with st.spinner("Wywołuję OpenAI..."):
                try:
                    client_local = get_client()
                    delta = analyze_news_impact(client_local, news_text)
                    apply_adjustment(delta)
                    st.session_state.last_delta = delta
                    st.success("Prognozy zaktualizowane")
                    st.json(delta)
                except Exception as e:
                    st.error(f"Błąd podczas wywołania OpenAI: {e}")

    cols = st.columns(2)
    for idx, industry in enumerate(INDUSTRIES):
        with cols[idx % 2]:
            render_chart(fin_df, industry, year_col, industry_col, value_col)

    st.divider()


if __name__ == "__main__":
    main()