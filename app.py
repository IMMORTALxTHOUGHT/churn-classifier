import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Bank Churn Predictor", layout="centered")
st.title("🏦 Bank Customer Churn Predictor")

@st.cache_resource
def load_models():
    lr = joblib.load('outputs/lr_model.pkl')
    prep = joblib.load('outputs/preprocessor.pkl')
    rf = joblib.load('outputs/rf_model.pkl')
    return lr, prep, rf

lr_pipeline, preprocessor, rf_model = load_models()
RF_THRESHOLD = 0.5543889249926127

@st.cache_data
def compute_metrics():
    df = pd.read_csv('data/Churn_Modelling.csv')
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    _, X_temp, _, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    lr_preds = lr_pipeline.predict(X_test)
    X_test_t = preprocessor.transform(X_test)
    rf_probs = rf_model.predict_proba(X_test_t)[:, 1]
    rf_preds = (rf_probs >= RF_THRESHOLD).astype(int)

    metrics = {}
    for name, preds in [("LR", lr_preds), ("RF", rf_preds)]:
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        metrics[name] = {
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1 Score": f1_score(y_test, preds),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        }

    results = pd.DataFrame(X_test).copy()
    results['Actual'] = y_test.values
    results['LR_Prob'] = lr_pipeline.predict_proba(X_test)[:, 1]
    results['RF_Prob'] = rf_probs
    results['LR_Pred'] = lr_preds
    results['RF_Pred'] = rf_preds

    cases = {}
    fp = results[(results['Actual'] == 0) & (results['RF_Pred'] == 1)].sort_values('RF_Prob', ascending=False)
    cases['fp'] = fp.head(1).iloc[0] if len(fp) > 0 else None
    fn = results[(results['Actual'] == 1) & (results['RF_Pred'] == 0)].sort_values('RF_Prob')
    cases['fn'] = fn.head(1).iloc[0] if len(fn) > 0 else None
    tp = results[(results['Actual'] == 1) & (results['RF_Pred'] == 1)].sort_values('RF_Prob', ascending=False)
    cases['tp'] = tp.head(1).iloc[0] if len(tp) > 0 else None

    return metrics, cases

metrics, error_cases = compute_metrics()

st.write(
    "Compare **Logistic Regression** (baseline) with the final "
    "**Random Forest** classifier using its validation-tuned threshold."
)

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 92, 35)
    tenure = st.slider("Tenure (years)", 0, 10, 3)
    credit_score = st.number_input("Credit Score", 350, 850, 600)

with col2:
    balance = st.number_input("Balance (€)", 0.0, 250000.0, 50000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary (€)", 0.0, 200000.0, 50000.0)

input_data = pd.DataFrame([{
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': 1 if has_cr_card == 'Yes' else 0,
    'IsActiveMember': 1 if is_active == 'Yes' else 0,
    'EstimatedSalary': estimated_salary
}])

if st.button("Predict Churn Risk", type="primary"):
    lr_prob = lr_pipeline.predict_proba(input_data)[0][1]
    input_transformed = preprocessor.transform(input_data)
    rf_prob = rf_model.predict_proba(input_transformed)[0][1]

    st.markdown("---")
    st.subheader("📊 Model Comparison")

    col_lr, col_rf = st.columns(2)
    with col_lr:
        st.metric("Logistic Regression", f"{lr_prob:.1%}")
    with col_rf:
        st.metric("Tuned Random Forest", f"{rf_prob:.1%}")

    st.markdown("---")
    st.subheader("💡 What This Tells Us")

    diff = rf_prob - lr_prob
    if abs(diff) < 0.05:
        st.write("Both models assign similar churn probabilities to this profile.")
    elif rf_prob > lr_prob:
        st.write(
            f"**RF is +{diff:.1%} higher** than LR. The models interpret this "
            "profile differently; this comparison alone does not identify which "
            "features caused the difference."
        )
    else:
        st.write(
            f"**LR is +{abs(diff):.1%} higher** than RF. The models interpret this "
            "profile differently; this comparison alone does not identify which "
            "features caused the difference."
        )

    st.markdown("---")
    if rf_prob >= RF_THRESHOLD:
        st.error("🚨 High Risk — Recommend proactive retention offers")
    else:
        st.success("✅ Low Risk — Standard monitoring is sufficient")
    st.caption(f"Final decision: Random Forest threshold = {RF_THRESHOLD:.3f}")

with st.expander("📋 Honest Evaluation Metrics — see precision, recall, and real error cases"):
    st.subheader("Test Set Performance (2,000 customers)")

    col_l, col_r = st.columns(2)
    for i, (key, name) in enumerate([
        ("LR", "Logistic Regression"),
        ("RF", f"Random Forest (threshold={RF_THRESHOLD:.3f})"),
    ]):
        m = metrics[key]
        col = col_l if i == 0 else col_r
        with col:
            tn, fp, fn, tp = m["TN"], m["FP"], m["FN"], m["TP"]
            st.markdown(f"**{name}**")
            st.markdown(f"Accuracy  **{m['Accuracy']:.1%}**  |  Precision  **{m['Precision']:.1%}**")
            st.markdown(f"Recall    **{m['Recall']:.1%}**  |  F1 Score   **{m['F1 Score']:.1%}**")
            st.markdown(
                f"<span style='color:green'>TN {tn}</span>  "
                f"<span style='color:red'>FP {fp}</span>  "
                f"<span style='color:red'>FN {fn}</span>  "
                f"<span style='color:green'>TP {tp}</span>",
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.subheader("🔍 Where Did the Model Go Wrong? (3 Real Test Cases)")

    for case_type, label, explain in [
        ("fp", "❌ False Positive (RF said churn, actually stayed)",
         "Germany, inactivity, age, and a high balance are associated with churn in this "
         "dataset and may have raised the score. This error shows those signals do not "
         "determine an individual customer's outcome."),
        ("fn", "⚠️ False Negative (RF said stay, actually churned)",
         "Active membership, France, zero balance, and two products may have lowered the "
         "score despite the customer's age. This is a costly miss because the model did "
         "not flag a customer who actually churned."),
        ("tp", "✅ True Positive (RF correctly predicted churn)",
         "This profile contains several attributes associated with churn in the dataset, "
         "including Germany, inactivity, age, and multiple products. These associations "
         "help interpret the prediction but do not prove causation."),
    ]:
        case = error_cases[case_type]
        if case is not None:
            st.markdown(f"**{label}**")
            st.markdown(
                f"CS {int(case['CreditScore'])} | {case['Geography']} | {case['Gender']} | "
                f"Age {int(case['Age'])} | Tenure {int(case['Tenure'])} | "
                f"Balance €{case['Balance']:,.0f} | Products {int(case['NumOfProducts'])} | "
                f"{'Active' if case['IsActiveMember'] else 'Inactive'}"
            )
            st.markdown(
                f"LR predicted {case['LR_Prob']:.1%}  |  "
                f"RF predicted {case['RF_Prob']:.1%}  |  "
                f"Actual {'churned' if case['Actual'] else 'stayed'}"
            )
            st.markdown(f"*{explain}*")
            st.markdown("")

st.markdown("---")
st.caption("Dataset: Bank Customer Churn (Kaggle). LR: Logistic Regression, RF: Random Forest with class_weight='balanced'.")
