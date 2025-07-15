import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pathlib

# ----------  LOAD TRAINED PIPELINE ----------
model_path = pathlib.Path(__file__).parent.parent / "models" / "pcos_model.pkl"
model = joblib.load(model_path)
train_cols = list(model.predict.__self__.feature_names_in_)

# ----------  OPTIONAL CSS ----------
st.markdown(
    """
    <style>
      .main .block-container {max-width: 700px; margin: auto;}
      .stButton>button {background-color:#e91e63;color:white;border-radius:8px; font-size: 16px;}
      .top-buttons {display: flex; justify-content: flex-start; margin-bottom: 20px;}
      .bottom-buttons {display: flex; justify-content: flex-end; margin-top: 40px;}
      .center-button {display: flex; justify-content: center; margin-top: 40px;}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------  PAGE STATE ----------
if "page" not in st.session_state:
    st.session_state.page = 0  # 0 = welcome, 1 = basic info, 2 = lifestyle, 3 = results

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

# helper widgets
def yn_unsure(label: str):
    return st.selectbox(label, ["Yes", "No", "Not sure"])

def yn_to_binary(answer: str):
    if answer == "Yes": return 1
    if answer == "No": return 0
    return None

# ----------  PAGE 0 : WELCOME ----------
if st.session_state.page == 0:
    st.title("Welcome to the PCOS Risk Predictor")
    name = st.text_input("What’s your name?", value=st.session_state.get("name", ""))

    if name:
        st.session_state.name = name
        st.markdown(f"### Hi {name}! 👋")
        st.write(
            "This app helps assess your risk for PCOS based on lifestyle and health factors.\n"
            "It is not a diagnosis tool, but can help you decide if you should seek further evaluation."
        )
        st.markdown('<div class="bottom-buttons">', unsafe_allow_html=True)
        st.button("Next ➜", on_click=next_page)
        st.markdown('</div>', unsafe_allow_html=True)

# ----------  PAGE 1 : BASIC INFO ----------
elif st.session_state.page == 1:
    st.title("PCOS Risk Predictor")
    st.header("Step 1 – Basic information")

    st.markdown('<div class="top-buttons">', unsafe_allow_html=True)
    st.button("⟵ Back", on_click=prev_page)
    st.markdown('</div>', unsafe_allow_html=True)

    st.session_state.age    = st.slider("Age",   10, 50, 25)
    st.session_state.weight = st.slider("Weight (kg)",   30, 150, 60)
    st.session_state.height = st.slider("Height (cm)",   120, 200, 160)

    st.markdown('<div class="bottom-buttons">', unsafe_allow_html=True)
    st.button("Next ➜", on_click=next_page, key="next-bottom")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------  PAGE 2 : LIFESTYLE ----------
elif st.session_state.page == 2:
    st.header("Step 2 – Lifestyle & Symptoms")

    st.markdown('<div class="top-buttons">', unsafe_allow_html=True)
    st.button("⟵ Back", on_click=prev_page)
    st.markdown('</div>', unsafe_allow_html=True)

    st.session_state.period_freq  = st.select_slider("Period frequency  (1 = regular … 5 = very irregular)", options=[1, 2, 3, 4, 5])
    st.session_state.acne_level   = st.select_slider("Acne severity     (1 = none … 5 = severe)", options=[1, 2, 3, 4, 5])
    st.session_state.mood_level   = st.select_slider("Mood swings       (1 = none … 5 = severe)", options=[1, 2, 3, 4, 5])

    st.session_state.exercise     = yn_unsure("Exercise regularly?")
    st.session_state.weight_gain  = yn_unsure("Recent weight gain?")
    st.session_state.skin_dark    = yn_unsure("Skin darkening near neck, armpits or groin?")
    st.session_state.hair_loss    = yn_unsure("Hair loss / thinning?")
    st.session_state.fast_food    = yn_unsure("Eat fast food frequently?")
    st.session_state.periods_reg  = yn_unsure("Are your periods regular?")

    st.session_state.blood_group = st.selectbox("Blood group", [1, 2, 3, 4, 5, 6, 7, 8], format_func=lambda n: ["A+","A−","B+","B−","AB+","AB−","O+","O−"][n-1])

    st.markdown('<div class="bottom-buttons">', unsafe_allow_html=True)
    st.button("Predict ➜", on_click=next_page, key="predict-button")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------  PAGE 3 : RESULTS ----------
elif st.session_state.page == 3:
    st.header("Step 3 – Your PCOS Risk")

    template = pd.DataFrame(np.zeros((1, len(train_cols))), columns=train_cols)
    template["age_in_years"]        = st.session_state.age
    template["weight_in_kg"]        = st.session_state.weight
    template["height_in_cm___feet"] = st.session_state.height
    template["after_how_many_months_do_you_get_your_periodsselect_1_if_every_month_regular"] = st.session_state.period_freq
    template["do_you_have_pimples_acne_on_your_face_jawline_"] = 1 if st.session_state.acne_level >= 3 else 0
    template["do_you_experience_mood_swings_"] = 1 if st.session_state.mood_level >= 3 else 0

    for col_name, answer in {
        "do_you_exercise_on_a_regular_basis_": st.session_state.exercise,
        "have_you_gained_weight_recently": st.session_state.weight_gain,
        "are_you_noticing_skin_darkening_recently": st.session_state.skin_dark,
        "do_have_hair_loss_hair_thinning_baldness_": st.session_state.hair_loss,
        "do_you_eat_fast_food_regularly_": st.session_state.fast_food,
        "are_your_periods_regular_": st.session_state.periods_reg,
    }.items():
        val = yn_to_binary(answer)
        if val is not None:
            template[col_name] = val

    template["can_you_tell_us_your_blood_group_"] = st.session_state.blood_group

    pred = model.predict(template)[0]
    reasons = []

    if st.session_state.acne_level >= 3:
        reasons.append("moderate to severe acne")
    if st.session_state.mood_level >= 3:
        reasons.append("frequent mood swings")
    if yn_to_binary(st.session_state.weight_gain) == 1:
        reasons.append("recent weight gain")
    if yn_to_binary(st.session_state.periods_reg) == 0:
        reasons.append("irregular periods")

    if pred == 1:
        st.error("⚠️ High risk of PCOS")
        if reasons:
            st.write("This assessment was influenced by:")
            for reason in reasons:
                st.markdown(f"- {reason.capitalize()}")
        st.write("Please consult a healthcare professional.")
    else:
        st.success("✅ Low risk of PCOS")
        if reasons:
            st.write("Some mild symptoms were noted:")
            for reason in reasons:
                st.markdown(f"- {reason.capitalize()}")

    st.markdown('<div class="top-buttons">', unsafe_allow_html=True)
    st.button("⟵ Back", on_click=prev_page)
    st.markdown('</div>', unsafe_allow_html=True)
