import base64
import datetime
from pathlib import Path

import streamlit as st

from feature_utils import get_match_insights, load_artifacts, predict_match

LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

st.set_page_config(
    page_title="Scoreseer — Football Match Predictions",
    page_icon="⚽",
    layout="centered",
)

LABEL_DISPLAY = {
    "Home_win": "Home Win",
    "Away_win": "Away Win",
    "Draw": "Draw",
}

# Common team/country name -> flag emoji. Falls back to a plain circle with
# initials for anything not in this list (women's/olympic squads, historical
# teams like "West Germany", regional confederations, etc).
FLAG_EMOJI = {
    "Afghanistan": "🇦🇫", "Albania": "🇦🇱", "Algeria": "🇩🇿", "Angola": "🇦🇴",
    "Argentina": "🇦🇷", "Armenia": "🇦🇲", "Australia": "🇦🇺", "Austria": "🇦🇹",
    "Azerbaijan": "🇦🇿", "Belgium": "🇧🇪", "Bolivia": "🇧🇴", "Bosnia and Herzegovina": "🇧🇦",
    "Brazil": "🇧🇷", "Bulgaria": "🇧🇬", "Cameroon": "🇨🇲", "Canada": "🇨🇦",
    "Chile": "🇨🇱", "China PR": "🇨🇳", "China": "🇨🇳", "Colombia": "🇨🇴",
    "Costa Rica": "🇨🇷", "Croatia": "🇭🇷", "Cuba": "🇨🇺", "Czech Republic": "🇨🇿",
    "Denmark": "🇩🇰", "Ecuador": "🇪🇨", "Egypt": "🇪🇬", "England": "🏴",
    "Finland": "🇫🇮", "France": "🇫🇷", "Georgia": "🇬🇪", "Germany": "🇩🇪",
    "Ghana": "🇬🇭", "Greece": "🇬🇷", "Honduras": "🇭🇳", "Hungary": "🇭🇺",
    "Iceland": "🇮🇸", "India": "🇮🇳", "Indonesia": "🇮🇩", "Iran": "🇮🇷",
    "Iraq": "🇮🇶", "Ireland": "🇮🇪", "Israel": "🇮🇱", "Italy": "🇮🇹",
    "Ivory Coast": "🇨🇮", "Jamaica": "🇯🇲", "Japan": "🇯🇵", "Jordan": "🇯🇴",
    "Kenya": "🇰🇪", "South Korea": "🇰🇷", "Korea Republic": "🇰🇷", "Kuwait": "🇰🇼",
    "Libya": "🇱🇾", "Mexico": "🇲🇽", "Morocco": "🇲🇦", "Netherlands": "🇳🇱",
    "New Zealand": "🇳🇿", "Nigeria": "🇳🇬", "North Macedonia": "🇲🇰", "Norway": "🇳🇴",
    "Pakistan": "🇵🇰", "Panama": "🇵🇦", "Paraguay": "🇵🇾", "Peru": "🇵🇪",
    "Philippines": "🇵🇭", "Poland": "🇵🇱", "Portugal": "🇵🇹", "Qatar": "🇶🇦",
    "Romania": "🇷🇴", "Russia": "🇷🇺", "Saudi Arabia": "🇸🇦", "Scotland": "🏴",
    "Senegal": "🇸🇳", "Serbia": "🇷🇸", "Slovakia": "🇸🇰", "Slovenia": "🇸🇮",
    "South Africa": "🇿🇦", "Spain": "🇪🇸", "Sweden": "🇸🇪", "Switzerland": "🇨🇭",
    "Thailand": "🇹🇭", "Tunisia": "🇹🇳", "Turkey": "🇹🇷", "Uganda": "🇺🇬",
    "Ukraine": "🇺🇦", "United Arab Emirates": "🇦🇪", "United States": "🇺🇸", "USA": "🇺🇸",
    "Uruguay": "🇺🇾", "Uzbekistan": "🇺🇿", "Venezuela": "🇻🇪", "Vietnam": "🇻🇳",
    "Wales": "🏴", "Zambia": "🇿🇲", "Zimbabwe": "🇿🇼",
}


def flag_for(team_name: str) -> str:
    if team_name in FLAG_EMOJI:
        return FLAG_EMOJI[team_name]
    return (team_name[:2] or "?").upper()


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

.stApp {
    background: radial-gradient(circle at 50% 0%, #121a2e 0%, #0a0e1a 55%);
}

.block-container { padding-top: 2.5rem; max-width: 760px; }

/* ---- Page header ---- */
.ss-header { text-align: center; margin-bottom: 2rem; }
.ss-header .ss-logo-img {
    max-width: min(360px, 85%);
    height: auto;
    display: block;
    margin: 0 auto;
    filter: drop-shadow(0 0 24px rgba(96,165,250,0.18));
}

/* ---- Card containers (st.container(border=True, key=...)) ---- */
.st-key-match_setup, .st-key-result_card {
    background: #0f1524;
    border: 1px solid #1e293b !important;
    border-radius: 18px;
    padding: 0.4rem 0.4rem;
}

/* ---- Small text helpers ---- */
.ss-eyebrow {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
    color: #60a5fa; text-transform: uppercase; margin-bottom: 0.3rem;
}
.ss-h1 {
    font-size: 1.4rem; font-weight: 800; letter-spacing: 0.02em;
    color: #f8fafc; margin-bottom: 0.35rem;
}
.ss-desc { font-size: 0.82rem; color: #94a3b8; line-height: 1.4; margin-bottom: 0.5rem; }
.ss-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
    color: #64748b; text-transform: uppercase; margin-bottom: 0.35rem;
}
.ss-label.center { text-align: center; }

/* ---- Badges / pills ---- */
.ss-badge {
    display: inline-block; padding: 0.3rem 0.75rem; border-radius: 999px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em; white-space: nowrap;
}
.ss-badge-blue { background: rgba(59,130,246,0.12); border: 1px solid #3b82f6; color: #93c5fd; }
.ss-badge-orange { background: rgba(245,158,11,0.12); border: 1px solid #f59e0b; color: #fcd34d; }
.ss-badge-green { background: rgba(16,185,129,0.12); border: 1px solid #10b981; color: #6ee7b7; }
.ss-badge-gray { background: rgba(100,116,139,0.12); border: 1px solid #475569; color: #94a3b8; }
.ss-badge-row { text-align: right; }

.ss-chip {
    display: inline-block; padding: 0.12rem 0.5rem; border-radius: 999px;
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.03em;
    background: #2563eb; color: #eff6ff; margin-left: 0.5rem; vertical-align: middle;
}

/* ---- VS / flag circles ---- */
.ss-flag-circle {
    width: 56px; height: 56px; border-radius: 999px; background: #161d2e;
    border: 1px solid #263044; display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; margin: 0 auto 0.5rem auto;
}
.ss-vs-circle {
    width: 34px; height: 34px; border-radius: 999px; background: #fbbf24;
    color: #1e1b0a; font-weight: 800; font-size: 0.75rem;
    display: flex; align-items: center; justify-content: center;
    margin: 2.1rem auto 0 auto;
}

/* ---- Outcome rows + progress bars ---- */
.ss-outcome-row { display: flex; justify-content: space-between; align-items: center; margin-top: 1rem; }
.ss-outcome-label { font-size: 0.85rem; font-weight: 700; color: #e2e8f0; }
.ss-outcome-pct { font-size: 0.95rem; font-weight: 800; }
.ss-pct-home { color: #60a5fa; }
.ss-pct-draw { color: #e2e8f0; }
.ss-pct-away { color: #fbbf24; }

.ss-bar-track { height: 8px; border-radius: 999px; background: #1e293b; margin-top: 0.35rem; overflow: hidden; }
.ss-bar-fill { height: 100%; border-radius: 999px; }
.ss-bar-home { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.ss-bar-draw { background: #64748b; }
.ss-bar-away { background: linear-gradient(90deg, #f59e0b, #fbbf24); }

.ss-key-factors {
    margin-top: 1.1rem; padding: 0.8rem 1rem; border-radius: 12px;
    background: #0b1120; border: 1px solid #1e293b; font-size: 0.78rem; color: #cbd5e1;
}
.ss-key-factors b { color: #e2e8f0; }

.ss-caption { font-size: 0.72rem; color: #64748b; margin-top: 0.6rem; }

/* ---- Native widgets re-themed ---- */
div[data-testid="stSelectbox"] > div, div[data-testid="stDateInput"] input {
    background-color: #0b1120 !important;
    border: 1px solid #263044 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
div[data-testid="stSelectbox"] label, div[data-testid="stDateInput"] label { color: #94a3b8 !important; }

div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #60a5fa);
    color: white; border: none; border-radius: 12px;
    font-weight: 700; letter-spacing: 0.02em; padding: 0.7rem 0;
    width: 100%;
}
div.stButton > button:hover { filter: brightness(1.08); }

div[data-testid="stToggle"] label { color: #94a3b8 !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_artifacts():
    return load_artifacts()


def confidence_badge(top_prob: float) -> str:
    if top_prob >= 0.60:
        return '<span class="ss-badge ss-badge-blue">⚡ HIGH CONFIDENCE</span>'
    if top_prob >= 0.45:
        return '<span class="ss-badge ss-badge-orange">⚡ MEDIUM CONFIDENCE</span>'
    return '<span class="ss-badge ss-badge-gray">⚡ LOW CONFIDENCE</span>'


@st.cache_data
def get_logo_base64():
    return base64.b64encode(LOGO_PATH.read_bytes()).decode()


def main():
    if LOGO_PATH.exists():
        st.markdown(
            f"""
            <div class="ss-header">
                <img class="ss-logo-img" alt="Scoreseer — Football Match Predictions"
                     src="data:image/png;base64,{get_logo_base64()}" />
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Fallback if assets/logo.png is missing, so the app still runs
        st.markdown(
            """
            <div class="ss-header">
                <div style="font-size:1.5rem; font-weight:800; letter-spacing:0.12em; color:#f8fafc;">
                    ⚽ SCORESEER
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    try:
        artifacts = get_artifacts()
    except FileNotFoundError as e:
        st.error(
            "Model artifacts not found. Make sure the `artifacts/` folder "
            "(cat_model.cbm, schema.json, and the lookup JSON files) sits "
            f"next to app.py.\n\nDetails: {e}"
        )
        st.stop()

    teams = artifacts["teams"]
    tournaments = artifacts["tournaments"]
    countries = artifacts["countries"]

    with st.container(border=True, key="match_setup"):
        top_l, top_r = st.columns([3, 1])
        with top_l:
            st.markdown('<div class="ss-eyebrow">— Match Setup</div>', unsafe_allow_html=True)
            st.markdown('<div class="ss-h1">BUILD THE FIXTURE</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="ss-desc">Set the teams and match conditions. '
                "The model will combine historical strength with current context.</div>",
                unsafe_allow_html=True,
            )
        with top_r:
            st.markdown(
                '<div class="ss-badge-row"><span class="ss-badge ss-badge-blue">● MODEL READY</span></div>',
                unsafe_allow_html=True,
            )

        col1, col_vs, col2 = st.columns([5, 1, 5])
        default_home = teams.index("Brazil") if "Brazil" in teams else 0
        with col1:
            st.markdown('<div class="ss-label center">HOME TEAM</div>', unsafe_allow_html=True)
            home_team = st.selectbox("Home team", teams, index=default_home, label_visibility="collapsed")
            st.markdown(f'<div class="ss-flag-circle">{flag_for(home_team)}</div>', unsafe_allow_html=True)
        with col_vs:
            st.markdown('<div class="ss-vs-circle">VS</div>', unsafe_allow_html=True)
        with col2:
            away_options = [t for t in teams if t != home_team]
            default_away = away_options.index("Argentina") if "Argentina" in away_options else 0
            st.markdown('<div class="ss-label center">AWAY TEAM</div>', unsafe_allow_html=True)
            away_team = st.selectbox("Away team", away_options, index=default_away, label_visibility="collapsed")
            st.markdown(f'<div class="ss-flag-circle">{flag_for(away_team)}</div>', unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="ss-label">Tournament</div>', unsafe_allow_html=True)
            default_t = tournaments.index("FIFA World Cup") if "FIFA World Cup" in tournaments else 0
            tournament = st.selectbox("Tournament", tournaments, index=default_t, label_visibility="collapsed")
        with col4:
            st.markdown('<div class="ss-label">Host country</div>', unsafe_allow_html=True)
            default_c = countries.index(home_team) if home_team in countries else 0
            country = st.selectbox("Host country", countries, index=default_c, label_visibility="collapsed")

        col5, col6 = st.columns(2)
        with col5:
            st.markdown('<div class="ss-label">Match date</div>', unsafe_allow_html=True)
            match_date = st.date_input("Match date", value=datetime.date.today(), label_visibility="collapsed")
        with col6:
            st.markdown('<div class="ss-label">Neutral venue</div>', unsafe_allow_html=True)
            neutral = st.toggle("Neutral venue", value=(country != home_team), label_visibility="collapsed")

        predict_clicked = st.button("Predict result  →", use_container_width=True)
        st.markdown(
            '<div class="ss-caption">⚡ Prediction typically resolves in under 2 seconds</div>',
            unsafe_allow_html=True,
        )

    if predict_clicked:
        if home_team == away_team:
            st.warning("Home and away teams must be different.")
            st.stop()

        with st.spinner("Calculating outcome..."):
            predicted_label, probs = predict_match(
                artifacts, home_team, away_team, tournament, country, neutral, match_date
            )
            insights = get_match_insights(artifacts, home_team, away_team)

        st.session_state["last_result"] = {
            "home_team": home_team,
            "away_team": away_team,
            "predicted_label": predicted_label,
            "probs": probs,
            "insights": insights,
        }

    result = st.session_state.get("last_result")
    if result:
        home_team = result["home_team"]
        away_team = result["away_team"]
        predicted_label = result["predicted_label"]
        probs = result["probs"]
        insights = result["insights"]

        home_pct = probs.get("Home_win", 0)
        draw_pct = probs.get("Draw", 0)
        away_pct = probs.get("Away_win", 0)
        top_prob = max(home_pct, draw_pct, away_pct)

        with st.container(border=True, key="result_card"):
            top_l, top_r = st.columns([3, 1])
            with top_l:
                st.markdown('<div class="ss-eyebrow">Model Output · Revealed</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="ss-h1">{home_team.upper()} VS {away_team.upper()}</div>',
                    unsafe_allow_html=True,
                )
            with top_r:
                st.markdown(
                    f'<div class="ss-badge-row">{confidence_badge(top_prob)}</div>',
                    unsafe_allow_html=True,
                )

            predicted_chip = '<span class="ss-chip">PREDICTED RESULT</span>'

            st.markdown(
                f"""
                <div class="ss-outcome-row">
                    <div class="ss-outcome-label">Home Win{predicted_chip if predicted_label == 'Home_win' else ''}</div>
                    <div class="ss-outcome-pct ss-pct-home">{home_pct * 100:.0f}%</div>
                </div>
                <div class="ss-bar-track"><div class="ss-bar-fill ss-bar-home" style="width:{home_pct * 100:.0f}%"></div></div>

                <div class="ss-outcome-row">
                    <div class="ss-outcome-label">Draw{predicted_chip if predicted_label == 'Draw' else ''}</div>
                    <div class="ss-outcome-pct ss-pct-draw">{draw_pct * 100:.0f}%</div>
                </div>
                <div class="ss-bar-track"><div class="ss-bar-fill ss-bar-draw" style="width:{draw_pct * 100:.0f}%"></div></div>

                <div class="ss-outcome-row">
                    <div class="ss-outcome-label">Away Win{predicted_chip if predicted_label == 'Away_win' else ''}</div>
                    <div class="ss-outcome-pct ss-pct-away">{away_pct * 100:.0f}%</div>
                </div>
                <div class="ss-bar-track"><div class="ss-bar-fill ss-bar-away" style="width:{away_pct * 100:.0f}%"></div></div>
                """,
                unsafe_allow_html=True,
            )

            elo_line = (
                f"ELO gap: +{insights['elo_gap']} {insights['elo_favored']}"
                if insights["elo_favored"]
                else "ELO gap: even"
            )
            form_line = f"Recent form: {home_team} {insights['home_form']} · {away_team} {insights['away_form']}"
            h2h_line = f"Head-to-head: {insights['h2h_summary']}"

            st.markdown(
                f"""
                <div class="ss-key-factors">
                    📊 <b>Key factors</b> — {elo_line} · {form_line} · {h2h_line}
                </div>
                """,
                unsafe_allow_html=True,
            )

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
