// Builds docs/MLOps_Assignment1_Report.docx  (run from repo root: node docs/build_report.js)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, WidthType, ShadingType, BorderStyle, LevelFormat,
  PageNumber, Footer, Header, PageBreak, ExternalHyperlink,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const f = (p) => path.join(ROOT, p);
const meta = JSON.parse(fs.readFileSync(f("models/model_metadata.json")));
const summary = JSON.parse(fs.readFileSync(f("reports/training_summary.json")));

const FONT = "Calibri";
const ACCENT = "1F4E79";
const PAGE_W = 11906, MARGIN = 1134;             // A4, 2 cm margins
const CONTENT_W = PAGE_W - 2 * MARGIN;           // 9638 DXA
const PX_PER_DXA = 96 / 1440;

// ---------- helpers ----------
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function img(file, widthPct = 0.9, caption) {
  const { w, h } = pngSize(f(file));
  const width = Math.round(CONTENT_W * PX_PER_DXA * widthPct);
  const out = [new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(f(file)),
      transformation: { width, height: Math.round(width * h / w) },
      altText: { title: caption || file, description: caption || file, name: path.basename(file) } })],
  })];
  if (caption) out.push(cap(caption));
  return out;
}
function imgPair(a, b, caption) {
  const cellW = CONTENT_W / 2;
  const mk = (file) => {
    const { w, h } = pngSize(f(file));
    const width = Math.round(cellW * PX_PER_DXA * 0.95);
    return new TableCell({
      width: { size: cellW, type: WidthType.DXA }, borders: NOBORDER,
      children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({
        type: "png", data: fs.readFileSync(f(file)),
        transformation: { width, height: Math.round(width * h / w) },
        altText: { title: file, description: file, name: path.basename(file) } })] })],
    });
  };
  return [new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [cellW, cellW],
    rows: [new TableRow({ children: [mk(a), mk(b)] })] }), cap(caption)];
}
let figNo = 0, tabNo = 0;
function cap(text) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: `Figure ${++figNo}. `, bold: true, size: 18, color: "555555" }),
               new TextRun({ text, italics: true, size: 18, color: "555555" })] });
}
function tcap(text) {
  return new Paragraph({ spacing: { before: 160, after: 80 }, keepNext: true,
    children: [new TextRun({ text: `Table ${++tabNo}. `, bold: true, size: 18, color: "555555" }),
               new TextRun({ text, italics: true, size: 18, color: "555555" })] });
}
function runs(text) {
  // **bold** and `code` inline markup
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean);
  return parts.map((p) => {
    if (p.startsWith("**")) return new TextRun({ text: p.slice(2, -2), bold: true });
    if (p.startsWith("`")) return new TextRun({ text: p.slice(1, -1), font: "Consolas", size: 19, color: "7A2E0E" });
    return new TextRun(p);
  });
}
const P = (text, opts = {}) => new Paragraph({ spacing: { after: 120 }, children: runs(text), ...opts });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: "bullets", level },
  spacing: { after: 60 }, children: runs(text) });
const N = (text) => new Paragraph({ numbering: { reference: "steps", level: 0 },
  spacing: { after: 60 }, children: runs(text) });
function code(lines) {
  return lines.map((l, i) => new Paragraph({
    shading: { type: ShadingType.CLEAR, color: "auto", fill: "F3F4F6" },
    spacing: { before: i === 0 ? 80 : 0, after: i === lines.length - 1 ? 140 : 0 },
    indent: { left: 200, right: 200 },
    children: [new TextRun({ text: l || " ", font: "Consolas", size: 17 })],
  }));
}
const NOBORDER = { top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" } };
const CELLB = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const BORDERS = { top: CELLB, bottom: CELLB, left: CELLB, right: CELLB };

function table(header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const mkCell = (t, i, isHead, highlight) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, borders: BORDERS,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    shading: isHead ? { type: ShadingType.CLEAR, color: "auto", fill: ACCENT }
      : highlight ? { type: ShadingType.CLEAR, color: "auto", fill: "E8F1FA" } : undefined,
    children: [new Paragraph({ alignment: i > 0 && opts.numeric ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: isHead ? [new TextRun({ text: t, bold: true, color: "FFFFFF", size: 18 })]
        : runs(String(t)).map((r) => { r.root?.forEach?.(() => {}); return r; }) })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => mkCell(h, i, true)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) =>
        mkCell(c, i, false, opts.highlightRow === ri)) }))],
  });
}
function placeholder(title, how) {
  const border = { style: BorderStyle.DASHED, size: 8, color: "8EA9C1" };
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: CONTENT_W, type: WidthType.DXA },
      borders: { top: border, bottom: border, left: border, right: border },
      shading: { type: ShadingType.CLEAR, color: "auto", fill: "F5F9FC" },
      margins: { top: 300, bottom: 300, left: 200, right: 200 },
      children: [
        new Paragraph({ alignment: AlignmentType.CENTER, children: [
          new TextRun({ text: `[Insert screenshot: ${title}]`, bold: true, color: "4A6A85" })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, children: [
          new TextRun({ text: how, italics: true, size: 18, color: "6B7F91" })] }),
      ] })] })],
  });
}
const sp = () => new Paragraph({ spacing: { after: 80 }, children: [] });
const r3 = (x) => x.toFixed(3);

// ---------- numbers ----------
const lr = summary.models.logistic_regression, rf = summary.models.random_forest, bl = summary.baseline;

// ---------- content ----------
const children = [];

// Title block
children.push(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1200, after: 120 }, children: [
    new TextRun({ text: "BITS Pilani · Work Integrated Learning Programmes", size: 22, color: "555555" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [
    new TextRun({ text: "Machine Learning Operations (AIMLCZG523) — Assignment 01", size: 24, color: "555555" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 200 }, children: [
    new TextRun({ text: "Heart Disease Risk Prediction:", size: 48, bold: true, color: ACCENT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, children: [
    new TextRun({ text: "an End-to-End MLOps Pipeline from Data to a Monitored Kubernetes API",
      size: 32, color: ACCENT })] }),
);
children.push(table(["Item", "Details"], [
  ["Student", "Premkumar Jaywardhan"],
  ["BITS ID", "________________"],
  ["Code repository", "https://github.com/<your-username>/heart-disease-mlops"],
  ["Deployed API", "Docker Desktop Kubernetes — http://localhost:8080 (access steps in §9)"],
  ["Demo video", "<link to recording>"],
  ["Date", "October 2026"],
], [2600, 7038]));
children.push(new Paragraph({ spacing: { before: 500, after: 120 }, children: [
  new TextRun({ text: "Contents", bold: true, size: 26, color: ACCENT })] }));
const toc = ["Overview", "Setup and Reproduction", "Data Acquisition and Exploratory Analysis",
  "Feature Engineering and Model Development", "Experiment Tracking with MLflow",
  "Model Packaging and Reproducibility", "CI/CD Pipeline and Automated Testing",
  "Model Containerisation", "Production Deployment on Kubernetes", "Monitoring and Logging",
  "Architecture", "Limitations and Next Steps"];
const half = Math.ceil(toc.length / 2);
children.push(new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [CONTENT_W / 2, CONTENT_W / 2],
  rows: Array.from({ length: half }, (_, i) => new TableRow({ children: [i, i + half].map((k) => new TableCell({
    width: { size: CONTENT_W / 2, type: WidthType.DXA }, borders: NOBORDER,
    children: [new Paragraph({ spacing: { after: 40 }, children: toc[k] ? [
      new TextRun({ text: `${k + 1}.  `, bold: true, color: ACCENT }), new TextRun(toc[k])] : [] })] })) })) }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// 1. Overview
children.push(H1("1. Overview"),
  P("The goal is to predict whether a patient has heart disease from 13 routinely collected clinical attributes, and to run that model the way a production team would: reproducible data and training, tracked experiments, automated tests and builds, a containerised API on Kubernetes, and live monitoring."),
  P(`**Outcome.** A regularised Logistic Regression was selected over a tuned Random Forest on cross-validated ROC-AUC (**${r3(lr.cv.roc_auc_mean)} ± ${r3(lr.cv.roc_auc_std)}** vs ${r3(rf.cv.roc_auc_mean)} ± ${r3(rf.cv.roc_auc_std)}). On the untouched 20 % hold-out it reaches ROC-AUC **${r3(lr.test.test_roc_auc)}**, accuracy **${r3(lr.test.test_accuracy)}** and recall **${r3(lr.test.test_recall)}**. It is served by a FastAPI service in a non-root Docker image, deployed as a 2-replica Kubernetes Deployment behind a LoadBalancer, and observed through structured JSON logs and a Prometheus + Grafana dashboard. A GitHub Actions pipeline lints, tests (34 tests, 94 % coverage), retrains with a quality gate, builds and smoke-tests the image on every push.`),
  tcap("How each assignment task maps to the deliverables"),
  table(["Task (marks)", "Implementation", "Section"], [
    ["1 Data & EDA (5)", "`download_data.py`, `heartml.data`, notebook 01, 7 figures", "§3"],
    ["2 Features & models (8)", "`ColumnTransformer`; LogReg + RF grid search, 5-fold CV", "§4"],
    ["3 Experiment tracking (5)", "MLflow: 68 runs, nested trials, plots, registry", "§5"],
    ["4 Packaging (7)", "joblib pipeline + MLflow model + metadata; pinned requirements", "§6"],
    ["5 CI/CD & tests (8)", "pytest suite; GitHub Actions lint→test→train→docker", "§7"],
    ["6 Containerisation (5)", "FastAPI `/predict`, multi-stage Dockerfile", "§8"],
    ["7 Deployment (7)", "Kustomize manifests, LoadBalancer + Ingress, probes", "§9"],
    ["8 Monitoring (3)", "JSON logs, Prometheus metrics + alerts, Grafana", "§10"],
    ["9 Documentation (2)", "This report, README, architecture diagram", "§2, §11"],
  ], [2500, 5938, 1200]),
);

// 2. Setup
children.push(H1("2. Setup and Reproduction"),
  P("Everything runs from a clean clone with one requirements file. A second, minimal file (`requirements-serve.txt`, 8 packages) is used for the container so the image carries no training or tracking libraries. All versions are pinned and were checked to resolve on Python 3.12."),
  ...code([
    "git clone https://github.com/<your-username>/heart-disease-mlops && cd heart-disease-mlops",
    "python -m venv .venv && source .venv/bin/activate",
    "pip install -r requirements.txt && pip install -e .",
    "python scripts/download_data.py      # raw + cleaned CSV",
    "python scripts/run_eda.py            # EDA figures",
    "python scripts/train.py              # tuning, MLflow tracking, models/",
    "pytest -v && flake8 src app tests scripts",
    "mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000",
  ]),
  P("The same steps are available as `make data eda train test lint` (or `make all`). The three notebooks (`01_eda`, `02_model_training`, `03_inference`) call the same package code, so notebook, script and CI results cannot drift apart. A from-scratch run in a fresh virtual environment reproduced identical metrics (fixed `random_state=42` for the split, CV folds and Random Forest)."),
);

// 3. Data & EDA
const eda = JSON.parse(fs.readFileSync(f("reports/figures/eda/eda_summary.json")));
children.push(H1("3. Data Acquisition and Exploratory Analysis"),
  H2("3.1 Acquisition"),
  P("The dataset is the processed Cleveland subset of the UCI Heart Disease collection: 303 patients, 13 attributes and a diagnosis `num` from 0 (no disease) to 4. `scripts/download_data.py` fetches the official UCI file and, if that host is unreachable, falls back to a GitHub mirror of the same 303 records, converting its text codes (e.g. `asymptomatic` → `cp=4`) to the UCI encoding. The raw table is cached in `data/raw/`; the cleaned table in `data/processed/heart_clean.csv` is committed to the repository."),
  H2("3.2 Cleaning"),
  B("Every column is coerced to numeric; UCI `?` markers and stray text become missing values."),
  B("Values outside plausible clinical ranges (e.g. cholesterol outside 100–600 mg/dl) and unknown category codes are set to missing rather than silently kept."),
  B("Duplicate and unlabelled rows are dropped; the target is binarised as `num > 0`."),
  B(`Only ${Object.values(eda.missing_raw).reduce((a, b) => a + b, 0)} cells are missing (\`ca\`: ${eda.missing_raw.ca}, \`thal\`: ${eda.missing_raw.thal}). They are **imputed inside the model pipeline** (median / most frequent), not in the CSV, so the statistics are learned on each training fold only and the API applies the identical rule to requests that omit those fields.`),
  H2("3.3 Findings"),
);
children.push(...imgPair("reports/figures/eda/class_balance.png", "reports/figures/eda/target_correlation.png",
  "Class balance (left) and correlation of each feature with the diagnosis (right)."));
children.push(
  P(`Classes are close to balanced (${eda.class_counts["No disease"]} without / ${eda.class_counts["Disease"]} with disease), so no resampling is needed and ROC-AUC is a fair selection metric. The strongest single signals are thallium defect (\`thal\`, r = ${eda.top_target_correlations.thal}), number of affected vessels (\`ca\`, ${eda.top_target_correlations.ca}), exercise angina (${eda.top_target_correlations.exang}), ST depression (${eda.top_target_correlations.oldpeak}) and, negatively, maximum heart rate.`),
);
children.push(...img("reports/figures/eda/numeric_histograms.png", 0.95,
  "Numeric features by diagnosis. Patients with disease are older, reach lower peak heart rates and show more ST depression; cholesterol and resting blood pressure separate the classes only weakly."));
children.push(...img("reports/figures/eda/categorical_disease_rate.png", 0.95,
  "Disease rate per category. Asymptomatic chest pain (cp = 4) carries a 73 % disease rate; reversible thallium defect (thal = 7) 76 %."));
children.push(...img("reports/figures/eda/correlation_heatmap.png", 0.62,
  "Pearson correlation heatmap. No feature pair exceeds |r| = 0.6, so collinearity is mild (slope vs oldpeak, 0.58, is the largest)."));
children.push(
  P("**Decisions carried into modelling.** `cp`, `thal`, `restecg` and `slope` are nominal codes, so they are one-hot encoded rather than treated as numbers. Numeric features differ in scale by two orders of magnitude, so they are standardised (needed for a regularised linear model). Weak features such as `fbs` and `chol` are kept and left to L1/L2 regularisation rather than pruned by hand on 303 rows. `restecg = 1` occurs only 4 times, so the encoder ignores unseen codes instead of failing."),
);

// 4. Modelling
children.push(H1("4. Feature Engineering and Model Development"),
  H2("4.1 Preprocessing pipeline"),
  P("One scikit-learn `ColumnTransformer` handles all feature preparation and is saved together with the classifier as a single `Pipeline`, so raw patient fields go in and probabilities come out — there is no separate preprocessing step to forget at serving time."),
  tcap("Preprocessing"),
  table(["Columns", "Steps", "Output"], [
    ["age, trestbps, chol, thalach, oldpeak, ca", "median imputation → standard scaling", "6 columns"],
    ["sex, cp, fbs, restecg, exang, slope, thal", "most-frequent imputation → one-hot (unknown → all-zero)", "19 columns"],
  ], [3600, 4438, 1600]),
  H2("4.2 Models and tuning"),
  P("Two model families with different inductive biases were compared, plus a majority-class baseline as a floor:"),
  B("**Logistic Regression** — interpretable (coefficients are log-odds), strong on small tabular data. Grid: C ∈ {0.01 … 10} (7 values) × L2/L1 penalty × class weighting = 28 configurations."),
  B("**Random Forest** — captures non-linear effects and interactions without scaling. Grid: trees ∈ {200, 500} × max depth ∈ {3, 5, none} × min leaf ∈ {1, 3, 5} × max features ∈ {√p, 0.5} = 36 configurations."),
  P("Protocol: a stratified 80/20 split (242 / 61 patients); `GridSearchCV` with stratified 5-fold CV on the 80 % scoring accuracy, precision, recall, F1 and ROC-AUC, refitting on ROC-AUC. The family with the best mean CV ROC-AUC is selected. The hold-out set is never used for any choice — only for the final report numbers."),
  tcap("Cross-validated (5-fold, mean) and hold-out results of the best configuration per family"),
  table(["Model", "CV ROC-AUC", "CV Acc.", "CV Prec.", "CV Recall", "CV F1", "Test ROC-AUC", "Test Acc."], [
    ["Logistic Regression (L2, C = 0.1)", `${r3(lr.cv.roc_auc_mean)} ± ${r3(lr.cv.roc_auc_std)}`, r3(lr.cv.accuracy_mean), r3(lr.cv.precision_mean), r3(lr.cv.recall_mean), r3(lr.cv.f1_mean), r3(lr.test.test_roc_auc), r3(lr.test.test_accuracy)],
    ["Random Forest (500 trees, leaf ≥ 3)", `${r3(rf.cv.roc_auc_mean)} ± ${r3(rf.cv.roc_auc_std)}`, r3(rf.cv.accuracy_mean), r3(rf.cv.precision_mean), r3(rf.cv.recall_mean), r3(rf.cv.f1_mean), r3(rf.test.test_roc_auc), r3(rf.test.test_accuracy)],
    ["Majority-class baseline", "0.500", r3(bl.cv_accuracy_mean), "–", "0.000", "–", "0.500", r3(bl.test_accuracy)],
  ], [2638, 1200, 800, 800, 900, 700, 1300, 1300], { numeric: true, highlightRow: 0 }),
  sp(),
);
children.push(...img("reports/figures/models/model_comparison.png", 0.72,
  "Cross-validation comparison. Logistic Regression is ahead on every metric and has a lower spread across folds."));
children.push(
  H2("4.3 Why Logistic Regression"),
  B(`It wins on the selection metric and on every other CV metric, with lower fold-to-fold variance (ROC-AUC std ${r3(lr.cv.roc_auc_std)} vs ${r3(rf.cv.roc_auc_std)}).`),
  B(`It generalises more honestly: the gap between training and CV ROC-AUC is ${r3(lr.train_minus_cv_roc_auc)} versus ${r3(rf.train_minus_cv_roc_auc)} for the forest, which memorises much of a 242-row training set.`),
  B("Its coefficients can be read by a clinician (Figure 7), the artefact is tiny (a few KB) and inference takes microseconds."),
  P("Tuning detail: the top LR configurations are tightly grouped (C = 0.1 with L2 at 0.912, with class balancing at 0.911, C = 1 with L1 at 0.910), so the choice is stable rather than a lucky grid point. Strong regularisation (C = 0.01) and weak regularisation (C = 10) both score lower, confirming a genuine optimum. All 64 configurations are stored as MLflow child runs and as CSV tables."),
);
children.push(...imgPair("reports/figures/models/logistic_regression/roc_curve.png",
  "reports/figures/models/logistic_regression/confusion_matrix.png",
  `Selected model on the hold-out set: ROC curve (AUC ${r3(lr.test.test_roc_auc)}) and confusion matrix at threshold 0.5.`));
children.push(...img("reports/figures/models/logistic_regression/feature_importance.png", 0.55,
  "Largest Logistic Regression coefficients. Affected vessels, asymptomatic chest pain and reversible thallium defect raise risk; a normal thallium scan and high peak heart rate lower it — consistent with the EDA and clinical expectation."));

// 5. Tracking
children.push(H1("5. Experiment Tracking with MLflow"),
  P("MLflow uses a SQLite backend (`mlflow.db`, which enables the model registry) with artefacts in `./mlruns`. The experiment `heart-disease-classification` is organised so that every trained configuration is inspectable and comparable:"),
  tcap("MLflow run structure (68 runs per training execution)"),
  table(["Run", "Logged parameters", "Logged metrics", "Artefacts"], [
    ["baseline_majority_class", "strategy", "CV + test accuracy, precision, recall, F1, ROC-AUC", "–"],
    ["logistic_regression / random_forest (parents)", "best hyper-parameters, grid size, folds, selection metric", "CV mean ± std of 5 metrics, 5 test metrics, train–CV gap, tuning time", "ROC, PR, confusion matrix, feature importance, full CV table, MLflow model with signature + input example"],
    ["lr_trial_00 … 27, rf_trial_00 … 35 (nested)", "that grid point's hyper-parameters", "CV mean and std of 5 metrics, train ROC-AUC, rank", "–"],
    ["model_selection", "selected model, source run id", "CV and test ROC-AUC of each family, baseline accuracy", "comparison chart, model card JSON"],
  ], [2500, 2300, 2700, 2138]),
  P("The winning run's model is registered as `heart-disease-classifier` and tagged with the alias `champion`, so a deployment can refer to \"the current champion\" instead of a hard-coded run id. In CI the whole `mlruns/` folder and `mlflow.db` are uploaded as a build artefact, giving a tracked history for every pipeline execution."),
  ...img("screenshots/01_mlflow_experiments.png", 0.86, "MLflow UI: 68 runs; the logistic_regression parent expanded to show its nested lr_trial_* runs (one per grid point)."),
  ...img("screenshots/03_mlflow_lr_run.png", 0.86, "Selected run: CV and test metrics, best hyper-parameters, child runs, tags and the linked registered model."),
  ...img("screenshots/04_mlflow_registry.png", 0.86, "Model registry: heart-disease-classifier version 1 carries the champion alias."),
);

// 6. Packaging
children.push(H1("6. Model Packaging and Reproducibility"),
  tcap("Packaged artefacts in models/"),
  table(["File", "Purpose"], [
    ["`model.joblib`", "Complete fitted pipeline (imputers, scaler, encoder, classifier). Loaded by the API; needs only scikit-learn."],
    ["`mlflow_model/`", "Same pipeline in MLflow format with input/output signature, input example, `conda.yaml`, `python_env.yaml` — servable with `mlflow models serve`."],
    ["`model_metadata.json`", `Model card: family, version (${meta.model_version}), hyper-parameters, CV and test metrics, threshold, feature order, training rows, MLflow run id, registry version, Python and scikit-learn versions.`],
  ], [2200, 7438]),
  P("Reproducibility is enforced at several levels: pinned versions in `requirements.txt`; fixed random seeds; data cleaning and feature preparation implemented once in the `heartml` package and imported by scripts, notebooks, tests and CI; the training environment recorded in the model card; and the cleaned dataset committed alongside the code. Notebook 03 shows the joblib pipeline, the MLflow pyfunc model and the REST API all return the same predictions for the same patients — including one with `ca` and `thal` missing."),
);

// 7. CI/CD
children.push(H1("7. CI/CD Pipeline and Automated Testing"),
  H2("7.1 Test suite"),
  P("34 pytest tests (94 % line coverage) run without network access on synthetic UCI-shaped data, so they can run before any training:"),
  tcap("Test coverage by area"),
  table(["File", "What is verified"], [
    ["test_data.py (8)", "UCI `?` parsing, mirror code mapping, target binarisation, column order, out-of-range → missing, duplicate/unlabelled removal, junk strings, schema errors"],
    ["test_features.py (5)", "no NaNs after transform, one-hot width, standardisation, unseen category tolerated, determinism"],
    ["test_model.py (7)", "both families fit and emit valid probabilities, beat a dummy model by > 0.1 AUC, handle missing inputs, joblib round-trip, metric helper"],
    ["test_pipeline.py (2)", "EDA writes all figures; full training run (fast grids) logs the expected 7 MLflow runs, registers the model and writes all artefacts"],
    ["test_api.py (12)", "health/ready, prediction schema and confidence maths, optional fields, 422 on invalid / unknown / missing fields, batch, Prometheus counters, model card, 503 when no model"],
  ], [2200, 7438]),
  ...img("screenshots/07_pytest.png", 0.72, "Local test run: all 34 tests pass (the same suite gates the CI pipeline)."),
  H2("7.2 GitHub Actions workflow"),
  P("`.github/workflows/ci.yml` runs on every push and pull request. Each job depends on the previous one, so any lint error, failing test, coverage drop, weak model or broken container stops the pipeline with that step's log."),
  tcap("Pipeline jobs"),
  table(["Job", "Steps", "Fails when", "Artefacts"], [
    ["lint", "flake8 on src, app, tests, scripts", "any style or error violation", "flake8 report"],
    ["test", "clean install from requirements.txt; pytest with JUnit + coverage; summary table", "any test fails or coverage < 80 %", "junit.xml, coverage.xml"],
    ["train", "download → EDA → train → quality gate → summary table", "download/training error or CV ROC-AUC < 0.85", "model, MLflow runs, figures, training log"],
    ["docker", "build with the model just trained; run; check /ready, /predict output, 422 on bad input, /metrics; push to GHCR on main", "build fails or any smoke check fails", "image tarball, container log"],
  ], [1100, 4038, 2500, 2000]),
  placeholder("GitHub Actions — successful run (lint → test → train → docker)", "Actions tab → latest run (screenshots/05_github_actions_run.png)"),
  sp(),
  placeholder("GitHub Actions — run summary with test and model tables, artefacts", "run → Summary (screenshots/06_github_actions_summary.png)"),
);

// 8. Container
children.push(H1("8. Model Containerisation"),
  P("The API (`app/main.py`) is built with FastAPI. Requests are validated by a Pydantic schema that mirrors the training ranges and rejects unknown fields, so bad input fails fast with a field-level 422 rather than producing a silent wrong prediction."),
  tcap("API endpoints"),
  table(["Endpoint", "Behaviour"], [
    ["POST /predict", "One patient → prediction (0/1), label, confidence (probability of the predicted class), P(disease), risk band, model version, request id"],
    ["POST /predict/batch", "Up to 256 patients in one call"],
    ["GET /health, /ready", "Liveness (process up) and readiness (model loaded) — used by Docker and Kubernetes probes"],
    ["GET /model", "Model card from model_metadata.json"],
    ["GET /metrics, /docs", "Prometheus exposition; Swagger UI"],
  ], [2400, 7238]),
  ...code([
    "$ curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \\",
    "       -d '{\"age\":63,\"sex\":1,\"cp\":4,\"trestbps\":145,\"chol\":233,\"fbs\":1,\"restecg\":2,",
    "            \"thalach\":150,\"exang\":0,\"oldpeak\":2.3,\"slope\":3,\"ca\":0,\"thal\":6}'",
    "{\"prediction\":1,\"label\":\"heart_disease\",\"confidence\":0.5757,\"probability_disease\":0.5757,",
    " \"risk_band\":\"moderate\",\"model_version\":\"" + meta.model_version + "\",\"request_id\":\"b8d56736f7e2\"}",
  ]),
  P("The `Dockerfile` is multi-stage: wheels are built in a builder stage and only installed into a clean `python:3.12-slim` runtime with the 8 serving packages, the `app/` code and the two model files. It runs as a fixed non-root UID (10001), defines a `HEALTHCHECK` against `/ready`, and disables Uvicorn's own access log because the application writes structured logs. The serving environment was verified in isolation — a virtual environment with only `requirements-serve.txt` and only the files the image copies — and returned the same prediction."),
  ...code([
    "docker build -t heart-disease-api:1.0.0 .",
    "docker run -d --name heart-api -p 8000:8000 heart-disease-api:1.0.0",
    "curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @scripts/sample_request.json",
  ]),
  ...img("screenshots/08a_api_local_smoke_test.png", 0.95, "Smoke test of the serving app: high- and low-risk patients, a rejected invalid request (422), Prometheus counters and the JSON log lines."),
  placeholder("docker build + docker run + /predict response", "terminal (screenshots/08_docker_build_run.png)"),
);

// 9. Deployment
children.push(H1("9. Production Deployment on Kubernetes"),
  P("The target is the Kubernetes cluster built into Docker Desktop. Because that cluster shares the local Docker image store, the locally built image is used directly (`imagePullPolicy: IfNotPresent`). The manifests are plain YAML combined with Kustomize and were validated against the Kubernetes schemas in strict mode (15/15 resources valid)."),
  tcap("Kubernetes resources (k8s/)"),
  table(["Resource", "Configuration"], [
    ["Namespace heart-api", "Isolates the application"],
    ["Deployment heart-api", "2 replicas; rolling update with maxUnavailable 0 (zero-downtime); readiness probe /ready, liveness probe /health; CPU/memory requests and limits; non-root, read-only root filesystem, no privilege escalation; Prometheus scrape annotations"],
    ["Service heart-api", "type LoadBalancer, port 8080 → container 8000 (Docker Desktop publishes it on localhost:8080; on GKE/EKS/AKS it receives a public IP)"],
    ["Ingress (optional)", "NGINX ingress, host heart-api.localhost"],
  ], [2400, 7238]),
  ...code([
    "kubectl config use-context docker-desktop",
    "docker build -t heart-disease-api:1.0.0 .",
    "kubectl apply -k k8s/",
    "kubectl -n heart-api rollout status deploy/heart-api",
    "kubectl -n heart-api get deploy,pods,svc",
    "curl -X POST http://localhost:8080/predict -H 'Content-Type: application/json' -d @scripts/sample_request.json",
  ]),
  placeholder("kubectl get deploy,pods,svc — 2/2 pods Running, LoadBalancer on localhost:8080", "screenshots/09_k8s_resources.png"),
  sp(),
  placeholder("curl /predict through the LoadBalancer + Swagger UI at localhost:8080/docs", "screenshots/10_k8s_predict.png, 11_swagger.png"),
);

// 10. Monitoring
children.push(H1("10. Monitoring and Logging"),
  P("**Logging.** A middleware writes one JSON line per request (request id, method, path, status, latency, client) and the prediction path writes one line per prediction (request id, class, P(disease), model version). The request id is returned in the `x-request-id` header so a user report can be traced to its log line. In Kubernetes the logs are read with `kubectl logs` or shipped by any log collector."),
  ...code([
    "{\"ts\": \"2026-10-19T02:52:05\", \"level\": \"INFO\", \"logger\": \"heart_api\",",
    " \"msg\": \"request\", \"request_id\": \"12823d532360\", \"method\": \"POST\",",
    " \"path\": \"/predict\", \"status\": 200, \"latency_ms\": 15.37, \"client\": \"127.0.0.1\"}",
  ]),
  P("**Metrics.** The API exports Prometheus metrics; Prometheus (deployed in a `monitoring` namespace) discovers each API pod through Kubernetes service discovery and evaluates three alert rules. Grafana is provisioned with the Prometheus datasource and a ready-made dashboard."),
  tcap("Exported metrics and how they are used"),
  table(["Metric", "Dashboard panel / alert"], [
    ["api_requests_total{endpoint, method, status}", "Requests/s, rate by endpoint and status, error rate; alert if 5xx > 5 %"],
    ["api_request_latency_seconds (histogram)", "p50 / p95 / p99 latency of /predict; alert if p95 > 500 ms"],
    ["model_predictions_total{label}", "Predictions by class — a shift in the disease share is an early model-drift signal"],
    ["model_prediction_probability (histogram)", "Distribution of P(disease) — confidence drift"],
    ["api_requests_in_progress, up", "In-flight requests, healthy targets; alert if no healthy pod"],
  ], [3800, 5838]),
  ...code([
    "kubectl apply -k monitoring/          # Prometheus :9090, Grafana :3000 (admin/admin)",
    "python scripts/load_test.py --url http://localhost:8080 --n 300   # generate traffic",
  ]),
  P("For a quick local demo without Kubernetes, `docker compose up --build` starts the same API, Prometheus and Grafana stack."),
  placeholder("Grafana dashboard after the load test", "http://localhost:3000 → Heart Disease API dashboard (screenshots/12_grafana_dashboard.png)"),
  sp(),
  placeholder("Prometheus targets (2 pods UP) and API logs", "screenshots/13_prometheus_targets.png, 14_api_logs.png"),
);

// 11. Architecture
children.push(H1("11. Architecture"),
  ...img("docs/architecture.png", 1.0, "End-to-end architecture: data and experimentation, tracking and packaging, CI/CD, Kubernetes serving, and monitoring."),
  P("Data flows left to right through the experimentation stage into MLflow and the packaged model; GitHub Actions re-runs that path on every change and produces the image; the image runs as a Kubernetes Deployment behind a LoadBalancer; Prometheus, Grafana and structured logs close the loop by showing how the live service and its predictions behave."),
);

// 12. Conclusion
children.push(H1("12. Limitations and Next Steps"),
  B("**Small dataset.** 303 patients from one hospital; hold-out estimates carry wide uncertainty (61 test patients), which is why selection relies on cross-validation. Pooling the Hungarian, Swiss and Long Beach subsets would test generalisation across sites."),
  B("**Decision threshold.** 0.5 is used for now. In screening, missing a sick patient is costlier than a false alarm; the threshold is stored in the model card so it can be tuned for higher recall without retraining or code changes."),
  B("**Drift monitoring.** Prediction mix and confidence are tracked; adding input-feature drift checks (e.g. population stability index against the training data) and delayed-label performance tracking would complete the picture."),
  B("**Delivery.** The next step is continuous deployment: have CI update the image tag in the manifests and let a GitOps controller (Argo CD) roll it out, using the registry's `champion` alias for promotion."),
  H2("Links"),
  P("Code repository: https://github.com/<your-username>/heart-disease-mlops"),
  P("Dataset: UCI Machine Learning Repository, Heart Disease (Detrano et al., 1988), https://archive.ics.uci.edu/dataset/45/heart+disease"),
);

// ---------- document ----------
const doc = new Document({
  creator: "Premkumar Jaywardhan",
  title: "MLOps Assignment 01 — Heart Disease Risk Prediction",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: ACCENT },
        paragraph: { spacing: { before: 320, after: 140 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: "2E75B6" },
        paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 1000, hanging: 270 } } } }] },
    { reference: "steps", levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 },
      margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } }, titlePage: true },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "AIMLCZG523 MLOps · Assignment 01 · Heart Disease Risk Prediction", size: 16, color: "888888" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 16, color: "888888" })] })] }) },
    children,
  }],
});

const out = f("docs/MLOps_Assignment1_Report.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out); });
