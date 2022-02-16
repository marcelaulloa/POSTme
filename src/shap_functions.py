from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd


def def_feature_cols(pipeline_rf, categorical_features):
    num_features2 = list(pipeline_rf["pre_process"].transformers_[0][2])

    cat_features2 = list(
        pipeline_rf["pre_process"]
            .transformers_[1][1]["onehot"]
            .get_feature_names(categorical_features)
    )
    feature_cols = num_features2 + cat_features2
    return feature_cols


def plot_shap(example, pipeline_rf, explainer):
    categorical_features = example.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = example.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer,
         numeric_features),
        ("cat", categorical_transformer,
         categorical_features)
    ])

    y_pred_rf = pipeline_rf.predict(example)
    y_pred_rf_proba = pipeline_rf.predict_proba(example)

    example = pipeline_rf["pre_process"].transform(example)

    shap_values = explainer.shap_values(example)

    example = pd.DataFrame(example, columns=def_feature_cols(pipeline_rf, categorical_features))

    return explainer.expected_value, shap_values, example
