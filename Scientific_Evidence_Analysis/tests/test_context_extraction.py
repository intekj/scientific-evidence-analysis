from modules.context_extraction import extract_context


PAPER_ONE_PASSAGE = (
    "Study areas were the Karun basin and Gorganrud basin, Iran. "
    "Flood susceptibility was assessed using the Cascade Forest Model (CFM), "
    "compared with SVM, DT, RF, DNN, LightGBM, XGBoost and CatBoost. "
    "The dataset included 21 flood-conditioning factors from multiple sources, "
    "including satellite imagery, historical reports and field data. "
    "CFM achieved the highest accuracy in both study areas, with Overall Accuracy "
    ">95%, Kappa Coefficient >0.8 and AUC >0.95."
)

PAPER_TWO_PASSAGE = (
    "The Global Flood Database was used for the study period 2000–2018. "
    "Five individual ML algorithms and an ensemble voting model were evaluated, "
    "including XGBoost, Decision Tree, Random Forest, LightGBM and GLM. "
    "XGBoost achieved AUC 0.985, RF achieved AUC 0.984 and LightGBM achieved "
    "AUC 0.982."
)


def test_extracts_locations_methods_datasets_models_and_period():
    first = extract_context(PAPER_ONE_PASSAGE)
    second = extract_context(PAPER_TWO_PASSAGE)

    assert "Karun basin" in first["Geographic location"]
    assert "Gorganrud basin" in first["Geographic location"]
    assert "Iran" in first["Geographic location"]
    assert "Cascade Forest Model" in first["Methodology"]
    assert "compared with" in first["Methodology"]
    assert "21 flood-conditioning factors" in first["Dataset"]
    assert "CFM" in first["Model or algorithm"]

    assert "Global Flood Database" in second["Dataset"]
    assert second["Geographic location"] is None
    assert "2000–2018" in second["Study period"]
    assert "ensemble voting model" in second["Methodology"]
    assert "XGBoost" in second["Model or algorithm"]


def test_metric_values_keep_reported_numbers_and_comparators():
    first = extract_context(PAPER_ONE_PASSAGE)
    second = extract_context(PAPER_TWO_PASSAGE)

    assert "Overall Accuracy >95%" in first["Metric values"]
    assert "Kappa Coefficient >0.8" in first["Metric values"]
    assert "AUC >0.95" in first["Metric values"]
    assert "AUC 0.985" in second["Metric values"]
    assert "AUC 0.984" in second["Metric values"]
    assert "AUC 0.982" in second["Metric values"]


def test_missing_context_fields_remain_unidentified():
    context = extract_context("A brief note describes the material.")
    assert all(value is None for value in context.values())


def test_generic_mention_of_region_is_not_mistaken_for_a_place_name():
    context = extract_context("Rainfall decreased across the region.")
    assert context["Geographic location"] is None


def test_dataset_name_is_not_treated_as_a_study_location():
    context = extract_context(
        "The Global Flood Database was used for the analysis."
    )
    assert context["Dataset"] is not None
    assert context["Geographic location"] is None


def test_related_work_location_is_not_assigned_as_current_study_location():
    context = extract_context(
        "Previous studies in Khuzestan Province, Iran used flood inventories. "
        "The current study area was the Karun Basin."
    )
    assert "Karun Basin" in context["Geographic location"]
    assert "Khuzestan Province" not in context["Geographic location"]


def test_dataset_collection_period_is_not_a_reported_finding():
    context = extract_context(
        "Flood data were collected from 2000 to 2018."
    )
    assert context["Study period"] is not None
    assert context["Reported findings"] is None


def test_explicit_model_performance_is_extracted_as_a_finding():
    ensemble = extract_context(
        "The voting ensemble achieved AUC = 0.994."
    )
    comparison = extract_context(
        "CFM achieved the highest accuracy across both study areas."
    )
    assert "voting ensemble achieved" in ensemble["Reported findings"]
    assert "AUC = 0.994" in ensemble["Reported findings"]
    assert "CFM achieved the highest accuracy" in comparison["Reported findings"]


def test_extracts_train_validation_input_and_ensemble_vs_individual_context():
    context = extract_context(
        "The data were split into training and validation sets using 5-fold "
        "cross-validation. Twenty-one conditioning factors were used as input "
        "variables. Five individual algorithms were compared with an ensemble "
        "voting model, which achieved AUC 0.994."
    )
    assert "training and validation" in context["Training and validation"]
    assert "conditioning factors" in context["Input variables"]
    assert "individual algorithms" in context["Individual versus ensemble performance"]
    assert "ensemble voting model" in context["Individual versus ensemble performance"]
