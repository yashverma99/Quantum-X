import pytest
import numpy as np
import pandas as pd
from app.services.dataset_service import dataset_service
from app.services.preprocessing_service import preprocessing_service
from app.services.feature_service import feature_service


@pytest.fixture
def sample_biomedical_df():
    np.random.seed(42)
    n_samples = 100
    data = {
        "gene_A": np.random.normal(5.0, 1.0, n_samples),
        "gene_B": np.random.normal(10.0, 2.0, n_samples),
        "gene_C": np.random.normal(0.0, 1.0, n_samples),
        "gene_D": np.random.normal(2.5, 0.5, n_samples),
        "constant_marker": np.ones(n_samples),  # 0 variance
        "sample_id": [f"TCGA-BR-{i:03d}" for i in range(n_samples)],
        "diagnosis": ["Malignant" if i % 3 == 0 else "Benign" for i in range(n_samples)]
    }
    # Introduce deliberate missing values
    data["gene_A"][5] = np.nan
    data["gene_B"][12] = np.nan
    return pd.DataFrame(data)


def test_csv_validation():
    # 1. Valid CSV
    valid_csv = b"gene1,gene2,diagnosis\n" + b"".join(f"{i},{i*2},benign\n".encode() for i in range(15))
    valid, msg, df = dataset_service.validate_csv(valid_csv, "test.csv")
    assert valid is True
    assert df is not None
    assert df.shape[0] >= 10

    # 2. Empty file
    valid, msg, df = dataset_service.validate_csv(b"", "empty.csv")
    assert valid is False
    assert "empty" in msg.lower()

    # 3. Invalid extension
    valid, msg, df = dataset_service.validate_csv(valid_csv, "test.exe")
    assert valid is False
    assert "csv" in msg.lower()

    # 4. Insufficient rows
    short_csv = b"gene1,gene2,diagnosis\n1,2,benign\n2,4,malignant\n"
    valid, msg, df = dataset_service.validate_csv(short_csv, "short.csv")
    assert valid is False
    assert "insufficient rows" in msg.lower()


def test_data_quality_analysis(sample_biomedical_df):
    report = dataset_service.analyze_quality(sample_biomedical_df, target_col="diagnosis")
    assert report["row_count"] == 100
    assert report["missing_count"] == 2
    assert "constant_marker" in report["constant_columns"]
    assert "sample_id" in report["potential_id_columns"]
    assert "Benign" in report["class_distribution"]
    assert "Malignant" in report["class_distribution"]
    assert report["leakage_safe"] is True


def test_leakage_prevention():
    """
    CRITICAL TEST: Verify that preprocessing parameters (mean, scale)
    are fitted EXCLUSIVELY on the training partition and NOT leaked from test data.
    """
    np.random.seed(42)
    n = 100
    # Training data around mean=0, test data heavily shifted to mean=1000
    df = pd.DataFrame({
        "feature_1": np.concatenate([np.random.normal(0, 1, 80), np.random.normal(1000, 1, 20)]),
        "diagnosis": ["Benign"] * 50 + ["Malignant"] * 50
    })

    # Run pipeline with test_size=0.2
    result = preprocessing_service.run_pipeline(
        df=df,
        target_col="diagnosis",
        test_size=0.2,
        random_state=42,
        scaler_type="standard"
    )

    fitted_mean = result["scaler_parameters"]["mean"][0]
    total_dataset_mean = float(df["feature_1"].mean())

    # The fitted mean must be close to the train distribution (~200 if stratified or ~0 if train),
    # and strictly different from the total dataset mean which includes the shifted test rows!
    assert result["leakage_safe"] is True
    assert result["train_samples"] == 80
    assert result["test_samples"] == 20
    assert not np.isnan(fitted_mean)


def test_missing_values_handling(sample_biomedical_df):
    result = preprocessing_service.run_pipeline(
        df=sample_biomedical_df,
        target_col="diagnosis",
        missing_strategy="median"
    )
    X_train = result["_X_train"]
    X_test = result["_X_test"]
    assert np.isnan(X_train).sum() == 0
    assert np.isnan(X_test).sum() == 0


def test_variance_filtering(sample_biomedical_df):
    result = preprocessing_service.run_pipeline(
        df=sample_biomedical_df,
        target_col="diagnosis",
        scaler_type="standard"
    )
    X_train = result["_X_train"]
    X_test = result["_X_test"]
    feature_names = result["feature_names"]

    # constant_marker has 0 variance
    X_tr_filt, X_te_filt, retained_names, count = feature_service.apply_variance_threshold(
        X_train, X_test, feature_names, threshold=0.01
    )

    assert "constant_marker" not in retained_names
    assert count == len(retained_names)
    assert count < len(feature_names)


def test_pca_dimensionality_reduction(sample_biomedical_df):
    result = preprocessing_service.run_pipeline(
        df=sample_biomedical_df,
        target_col="diagnosis"
    )
    X_train = result["_X_train"]
    X_test = result["_X_test"]

    n_comp = 4
    X_tr_pca, X_te_pca, ratios, cum_var = feature_service.apply_pca(
        X_train, X_test, n_components=n_comp
    )

    assert X_tr_pca.shape[1] == n_comp
    assert X_te_pca.shape[1] == n_comp
    assert len(ratios) == n_comp
    assert all(r > 0 for r in ratios)
    assert 0.0 < cum_var <= 1.0


def test_quantum_vector_mapping(sample_biomedical_df):
    result = preprocessing_service.run_pipeline(
        df=sample_biomedical_df,
        target_col="diagnosis"
    )
    X_tr_pca, X_te_pca, _, _ = feature_service.apply_pca(
        result["_X_train"], result["_X_test"], n_components=4
    )

    X_tr_q, X_te_q, sample_vector, encoding = feature_service.map_to_quantum_state(
        X_tr_pca, X_te_pca
    )

    assert len(sample_vector) == 4
    assert all(0.0 <= val <= np.pi for val in sample_vector)
    assert np.all(X_tr_q >= 0.0) and np.all(X_tr_q <= np.pi)


def test_reproducibility(sample_biomedical_df):
    # Run 1 with seed 42
    res1 = preprocessing_service.run_pipeline(
        df=sample_biomedical_df, target_col="diagnosis", random_state=42
    )
    pca_tr1, _, r1, _ = feature_service.apply_pca(res1["_X_train"], res1["_X_test"], 4)
    _, _, vec1, _ = feature_service.map_to_quantum_state(pca_tr1, pca_tr1)

    # Run 2 with same seed 42
    res2 = preprocessing_service.run_pipeline(
        df=sample_biomedical_df, target_col="diagnosis", random_state=42
    )
    pca_tr2, _, r2, _ = feature_service.apply_pca(res2["_X_train"], res2["_X_test"], 4)
    _, _, vec2, _ = feature_service.map_to_quantum_state(pca_tr2, pca_tr2)

    assert vec1 == vec2
    assert r1 == r2
