import datetime

import pandas as pd
import streamlit as st

from feature_utils import load_artifacts, predict_match

st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="centered",
)

LABEL_DISPLAY = {
    "Home_win": "Home Win",
    "Away_win": "Away Win",
    "Draw": "Draw",
}


@st.cache_resource
def get_artifacts():
    return load_artifacts()


def main():
    st.title("⚽ Football Match Predictor")
    st.caption(
        "Predicts Home Win / Draw / Away Win using a CatBoost model trained on "
        "150+ years of international results, ELO ratings, recent form, and "
        "head-to-head history."
    )

    try:
        artifacts = get_artifacts()
    except FileNotFoundError as e:
        st.error(
            "Model artifacts not found. Make sure the `artifacts/` folder "
            "(cat_model.cbm, schema.json, and the lookup JSON files) sits "
            "next to app.py.\n\n"
            f"Details: {e}"
        )
        st.stop()

    teams = artifacts["teams"]
    tournaments = artifacts["tournaments"]
    countries = artifacts["countries"]

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home team", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    with col2:
        away_options = [t for t in teams if t != home_team]
        away_team = st.selectbox("Away team", away_options, index=away_options.index("Argentina") if "Argentina" in away_options else 0)

    tournament = st.selectbox(
        "Tournament",
        tournaments,
        index=tournaments.index("FIFA World Cup") if "FIFA World Cup" in tournaments else 0,
    )
    country = st.selectbox(
        "Host country",
        countries,
        index=countries.index(home_team) if home_team in countries else 0,
        help="The country the match is played in (not necessarily either team's country).",
    )
    match_date = st.date_input("Match date", value=datetime.date.today())
    neutral = st.checkbox(
        "Neutral venue",
        value=(country != home_team),
        help="Check this if the match isn't played in the home team's own country.",
    )

    st.divider()

    if st.button("Predict result", type="primary", use_container_width=True):
        if home_team == away_team:
            st.warning("Home and away teams must be different.")
            st.stop()

        predicted_label, probs = predict_match(
            artifacts, home_team, away_team, tournament, country, neutral, match_date
        )

        st.subheader(f"Prediction: {LABEL_DISPLAY.get(predicted_label, predicted_label)}")

        display_probs = {LABEL_DISPLAY.get(k, k): v for k, v in probs.items()}
        ordered = ["Home Win", "Draw", "Away Win"]
        display_probs = {k: display_probs[k] for k in ordered if k in display_probs}

        cols = st.columns(3)
        for col, (label, prob) in zip(cols, display_probs.items()):
            col.metric(label, f"{prob * 100:.1f}%")

        chart_df = pd.DataFrame(
            {"Outcome": list(display_probs.keys()), "Probability": list(display_probs.values())}
        ).set_index("Outcome")
        st.bar_chart(chart_df)

        st.caption(
            f"{home_team} vs {away_team} · {tournament} · {country} · "
            f"{match_date.strftime('%d %b %Y')}"
            + (" · Neutral venue" if neutral else "")
        )

    st.divider()
    with st.expander("About this model"):
        st.write(
            "Trained with CatBoost on international results from 1872 onward, "
            "using ELO ratings, each team's form over its last 5 matches, and "
            "head-to-head history as features. Test-set accuracy on 2023+ "
            "matches was ~60% for this 3-way classification task (Home Win / "
            "Draw / Away Win), well above the ~33% random baseline."
        )


if __name__ == "__main__":
    main()
