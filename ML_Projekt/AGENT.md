# AGENT.md — Arbeitskontext für weitere Modellierungs-Loops

## Ziel und aktuelle Baseline

Binäre Klassifikation für `toxic_data_01111.csv` (8.000 Zeilen, ca. 10 % Positive);
Abgabemetrik ist der F1-Score der positiven Klasse auf 2.000 ungelabelten Exam-Zeilen.

**Aktuelle CV-Baseline: F1 = 0.6938 ± 0.0257.** Das ist der gepaarte Mittelwert aus
RepeatedStratifiedKFold (5 Splits × 10 Repeats) des 6-Modell-Stacks. Der globale
In-sample-OOF-Bestwert des Stacks ist 0.6945. Beide Werte dürfen nicht verwechselt werden.

## Aktuelle Pipeline

1. Aus den 16 Rohspalten werden 13 Features erzeugt:
   `m11`, `m15`, `both_miss`, `f12`, `f12_sq`, `f08`, `f10`, `f13`, `f14`,
   `f11_val`, `f15_slog`, `f09_slog`, `f11_abs`.
2. NaN werden nach Erzeugung der Missing-Indikatoren per Trainingsmedian imputiert.
   Nur das MLP erhält zusätzlich einen `RobustScaler`.
3. Fünf klassische Basismodelle werden auf denselben 5 Stratified-Folds trainiert:
   XGBoost, LightGBM, CatBoost, MLP und Random Forest. TabPFN liefert die sechste
   OOF-Spalte.
4. Ein Logistic-Regression-Metamodell lernt aus den sechs OOF-Wahrscheinlichkeiten.
   Es gibt daher **keine fest codierten Ensemblegewichte**. Die „Gewichte“ sind gelernte
   Logistic-Koeffizienten und ändern sich je Meta-CV-Fold; für die Exam-Prognose wird
   das Metamodell auf allen OOF-Zeilen neu gefittet. Die Koeffizienten werden aktuell
   nicht als Artefakt gespeichert.
5. Der Threshold wird über `0.02 <= t < 0.95` in Schritten von `0.005` nach maximalem
   positiven F1 gewählt. Der aktuelle globale Threshold ist `0.295`.
6. Für die Abgabe werden alle sechs Basismodelle auf allen Trainingsdaten refittet,
   über das volle Metamodell kombiniert und einmalig auf das Exam-Set angewandt.
   Die committed Variante nutzt kein Pseudo-Labeling.

Artefakte:

- `artifacts/tuned_params.json`: finale Parameter von XGB/LGBM/CatBoost
- `artifacts/base_oof.npz`: Arrays `xgb`, `lgbm`, `cat`, `mlp`, `rf`, `y`, jeweils
  Shape `(8000,)`, `float64`
- `artifacts/tabpfn_oof.npz`: Arrays `tab`, `y`, jeweils Shape `(8000,)`, `float64`

## Wie F1 im Repository tatsächlich berechnet wird

`f1_score(y_true, probability >= threshold)` verwendet sklearn-Defaults und damit
die positive Klasse `1`: `F1 = 2 TP / (2 TP + FP + FN)`.

- `finalize_with_tabpfn.py` erzeugt per 5-facher `cross_val_predict` Meta-OOF-Scores
  und sucht darauf einen einzigen Best-Threshold. Das ergibt den OOF-Bestwert 0.6945.
- `eval_tabpfn_combo.py` nimmt diese bereits erzeugten Meta-OOF-Scores, erzeugt 50
  wiederholte Train/Validation-Splits, bestimmt den Threshold jeweils auf dem
  Train-Anteil der OOF-Scores und misst F1 auf dem Validation-Anteil. Der Mittelwert
  ist 0.6938, Standardabweichung 0.0257.
- `final_result.py` berechnet **nicht** den finalen 0.6938-Score neu. Es berechnet nur
  eine Default-XGB-Baseline mit 5×3 CV; dabei wird der beste Threshold direkt auf
  jedem jeweiligen Validation-Fold gewählt, was optimistisch ist. Die 0.6811 und
  übrigen Pipelinewerte werden dort lediglich hart codiert ausgegeben. Außerdem
  beschreibt die Datei noch die ältere 5-Modell-/Pseudo-Label-Pipeline.

Wichtige methodische Einschränkung: Optuna-Auswahl, OOF-Erzeugung, Meta-Lernen und
Threshold-Evaluation sind nicht vollständig in einer äußeren Nested-CV verschachtelt.
0.6938 ist die Arbeitsbaseline für Vergleiche, aber kein unverzerrter Schätzer des
Exam-Scores.

## Alle aktuell aktiven Hyperparameter

Globale und Datenparameter:

- Seed überall: `random_state = 42`, zusätzlich `np.random.seed(42)`
- Basismodell-OOF: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Meta-OOF: derselbe 5-Fold-Splitter
- Berichtete Vergleichs-CV: `RepeatedStratifiedKFold(5, 10, random_state=42)`
- Imputation: `SimpleImputer(strategy="median")`
- MLP-Skalierung: `RobustScaler()` mit sklearn-Defaults
- Threshold-Grid: Start `0.02`, Ende exklusiv `0.95`, Schritt `0.005`
- Feature-Selection-Margin: `0.0015`

Finales XGBoost:

- `n_estimators=700`
- `max_depth=5`
- `learning_rate=0.09104431333255672`
- `subsample=0.635577246610902`
- `colsample_bytree=0.7943519236694145`
- `min_child_weight=4`
- `gamma=0.2759363643718443`
- `reg_lambda=2.5414811393859753`
- `scale_pos_weight=5.9120873890781`
- `eval_metric="logloss"`, `n_jobs=-1`, `random_state=42`

Finales LightGBM:

- `n_estimators=700`
- `max_depth=5`
- `learning_rate=0.08516975393982704`
- `num_leaves=44`
- `subsample=0.9652300228889714`
- `colsample_bytree=0.8447784721716562`
- `min_child_samples=15`
- `reg_lambda=3.9440174892203115`
- `scale_pos_weight=6.653347501616505`
- `verbose=-1`, `n_jobs=-1`, `random_state=42`

Finales CatBoost:

- `iterations=350`
- `depth=7`
- `learning_rate=0.07259248719561363`
- `l2_leaf_reg=6.387926357773329`
- `scale_pos_weight=6.092130483097056`
- `verbose=0`, `thread_count=-1`, `random_state=42`

Finales MLP:

- `hidden_layer_sizes=(128, 64)`
- `alpha=0.001`
- `learning_rate_init=0.001`
- `learning_rate="adaptive"`
- `batch_size=128`
- `max_iter=1000`
- `early_stopping=True`
- `validation_fraction=0.15`
- `random_state=42`

Finaler Random Forest:

- `n_estimators=500`
- `max_depth=12`
- `min_samples_leaf=3`
- `class_weight="balanced_subsample"`
- `n_jobs=-1`, `random_state=42`

Finales TabPFN:

- Paketstand laut Dokumentation: `tabpfn==2.2.1`
- bevorzugte Konstruktion: `device="cpu"`, `ignore_pretraining_limits=True`,
  `random_state=42`
- Environment: `TABPFN_ALLOW_CPU_LARGE_DATASET=1`
- Bei API-Inkompatibilität fällt der Code stufenweise auf weniger Argumente bis zu
  `TabPFNClassifier()` zurück. Der tatsächlich verwendete Parametersatz wird nicht
  protokolliert.

Finales Metamodell:

- `LogisticRegression(C=1.0, max_iter=2000)`; alle übrigen sklearn-Defaults
- Eingabereihenfolge: XGB, LGBM, CatBoost, MLP, RF, TabPFN
- keine statischen/normalisierten Gewichte, keine persistierten Koeffizienten

## Bisher gesetzte Such- und Experimentparameter

Diese Werte sind nicht alle im finalen Modell aktiv, wurden aber im Repository versucht:

- Optuna: TPE-Sampler, Seed 42; 40 XGB-, 40 LGBM- und 25 CatBoost-Trials.
- XGB-Suchraum: Trees 200–700/50, Depth 3–7, LR 0.01–0.15 log,
  Subsample 0.6–1.0, Columns 0.5–1.0, Child weight 1–8, Gamma 0–0.4,
  L2 0.5–5.0, Positive weight 5–12.
- LGBM-Suchraum: Trees 200–700/50, Depth 3–8, LR 0.01–0.15 log,
  Leaves 15–80, Subsample 0.6–1.0, Columns 0.5–1.0,
  min child samples 5–40, L2 0–5, Positive weight 5–12.
- CatBoost-Suchraum: Iterations 200–600/50, Depth 3–7, LR 0.01–0.15 log,
  L2 leaf 1–10, Positive weight 5–12.
- Feature-Selection-XGB: 250 Trees, Depth 4, LR 0.05, positive weight 9,
  Subsample/Columns 0.8, child weight 3, Gamma 0.1; repeated 5×2 CV.
- Live-Baseline-XGB: wie oben, aber 400 Trees; repeated 5×3 CV.
- Threshold-Strategien: global; Top-K mit `K=0.07..0.155` in 0.005-Schritten;
  drei Missingness-Strata mit Fallback bei `<40` Zeilen oder `<5` Positiven.
- Pseudo-Labels: positives Exam-Label bei 3-GBM-Blend `>0.90`, negatives bei `<0.03`;
  Akzeptanz nur bei Gain `>0.001`; getestet mit repeated 5×3 CV.
- Frühere manuelle XGB-Grids: Trees 300/400/500/600, Depth 3/4/5/6,
  LR 0.02/0.03/0.05/0.1, positive weight 7/8/9/10,
  Subsample/Columns 0.7/0.8/0.85/0.9, child weight 1/2/3/5,
  Gamma 0/0.05/0.1/0.2.
- Frühere MLP-Grids: `(128,64)`, `(256,128,64)`, `(64,32,16)`,
  `(128,64,32)`, `(512,256,128)`, `(64,64)`, `(128,)`, `(256,128)`;
  Alpha 0.0001/0.0005/0.001/0.005/0.01; LR-init 0.0003/0.0005/0.001/0.002.
- Frühere LGBM-Grids: Trees 300/400/500, Depth 4/5/6, LR 0.03/0.05,
  positive weight 8/9, Leaves 31/50/63, min child samples 5/10/20,
  Subsample/Columns 0.7/0.8.
- Frühere Modelle: Logistic Regression (`balanced`, C 1, max_iter 1000),
  Gradient Boosting (200 Trees, Depth 4, LR 0.1, min leaf 10, Subsample 0.8),
  RF (300 Trees, Depth 10, min leaf 5) und MLP `(64,32)`, max_iter 500.
- Frühere gewichtete Blends: Grobraster 0–1 in 0.25-Schritten, danach ±0.15 in
  sieben Schritten. Die finalen Blendgewichte wurden nicht dokumentiert und der
  Ansatz wurde zugunsten des Stacks verworfen.

## Was noch nicht versucht wurde

`docs/4_modeling.md` enthält keinen eigentlichen Scratchpad und dokumentiert vor allem
Erfolge. Aus Code, Phase 5/6 und Referenznotizen ergeben sich diese offenen Lücken:

1. TabPFN auf rohen, NaN-nativen 16 Features oder gezielten Roh-/Engineered-Hybriden.
2. Pseudo-Labeling auf dem **6-Modell-Stack**; getestet wurde nur die alte
   5-Modell-/XGB-Variante.
3. TabPFN-spezifische Optimierung: Feature-Subset, Ensemble-Konfiguration, mehrere Seeds
   oder mehrere diversifizierte TabPFN-OOF-Spalten.
4. Native Missing-Value-Varianten der Boostingmodelle statt globaler Median-Imputation,
   insbesondere CatBoost/LightGBM mit Rohfeatures plus Indikatoren.
5. Vollständig nested Evaluation von Tuning, Base-OOF, Meta-Learner und Threshold.
6. Kalibrierte, rank-basierte oder regularisierte Stack-Varianten und systematisches
   Tuning von `C`; aktuell wurde nur Logistic Regression mit `C=1` getestet.
7. Zusätzliche diversifizierende Baumfamilien wie ExtraTrees oder HistGradientBoosting.
8. Wiederholte/seedgebagte OOF-Erzeugung für die Basismodelle; aktuell stammt jede
   OOF-Spalte aus genau einem 5-Fold-Split.

## Top-5-Hypothesen für F1 > 0.70

Priorität bedeutet erwarteter Nutzen relativ zu Kosten und Overfitting-Risiko. Jede
Hypothese muss gegen exakt dieselben gepaarten äußeren Splits getestet werden.

1. **TabPFN roh/NaN-nativ und als zusätzliche Diversitätsspalte.** TabPFN brachte bereits
   +0.0127 durch andere Fehler. Seine aktuelle median-imputierte 13-Feature-Eingabe nimmt
   ihm möglicherweise Rohverteilungen und native Missingness. Den bestehenden TabPFN
   nicht blind ersetzen: Roh- und Engineered-Variante zunächst gemeinsam stacken.
2. **Mehrere robuste TabPFN-Varianten/Seeds stacken.** Der größte bisherige Gain kam aus
   Modelldiversität, nicht aus neuen Features. Ein kleines, kontrolliertes TabPFN-Bagging
   hat deshalb die plausibelste Chance auf weitere 0.0062, sofern OOF-Korrelation und
   paired Gain den Zusatz rechtfertigen.
3. **6-Modell-Pseudo-Labeling mit Cross-Fitting und Stabilitätsfilter.** Der alte Versuch
   gewann etwa +0.004, wurde aber nie auf dem stärkeren Stack wiederholt. Nur Exam-Zeilen
   verwenden, deren Label über Seeds/Folds stabil ist; sonst verstärkt der Loop Fehler.
4. **Missingness-native Boosting-Varianten als zusätzliche Bases.** CatBoost/LGBM können
   NaN direkt modellieren; Rohfeatures plus `m11/m15` könnten Schwellen innerhalb der
   present/missing-Regime besser lernen. Als zusätzliche OOF-Spalten testen, nicht sofort
   die bewährten Modelle ersetzen.
5. **Seedgebagging plus regularisierter/rank-basierter Meta-Stack.** Einzelne 5-Fold-OOF-
   Spalten sind verrauscht. Mehrere Seeds können Varianz senken; anschließend `C` und
   Probability-vs-Rank-Eingaben strikt outer-CV-tunen. Das zielt eher auf einen echten
   Generalisationsgewinn als auf einen weiteren in-sample Threshold-Zufallstreffer.

## Bereits validierte Dead Ends

Nicht ohne neue Evidenz wiederholen: externe/enriched/Lethal-Daten, Cross-Dataset-Row-Lookup,
reine Noise-Features `f00`–`f07`, breite Transform-Bloat-Features, per-Stratum-Thresholds
und naive Equal-Weight-Blends. Top-K war ungefähr gleichauf, aber nicht besser als global.

## Delegation Rules

### An lokales Qwen (`qwen3:32b`, Ollama `http://localhost:11434`)

Delegiere klar begrenzte, überprüfbare und parallelisierbare Arbeiten:

- Repository-Suche, Parameterinventar, Log-/CSV-Zusammenfassung und Boilerplate
- Generieren isolierter Experiment-Skripte nach einer bereits festgelegten Spezifikation
- Vorschläge für kleine Feature- oder Modellvarianten
- statische Code-Reviews mit konkreter Checkliste
- lange Laufzeitexperimente, deren Command, Inputs, Seeds und Erfolgskriterium feststehen

Qwen darf nicht selbstständig `predictions.csv`, bestehende OOF-Caches oder
`tuned_params.json` überschreiben. Ergebnisse gehen in einen neuen Run-Ordner und müssen
Command, Git-Diff, Laufzeit, Seed, Fold-Scores und Artefaktpfade enthalten.

Minimaler API-Aufruf:

```bash
curl http://localhost:11434/api/chat \
  -d '{"model":"qwen3:32b","stream":false,"messages":[{"role":"user","content":"<task>"}]}'
```

### Claude/Codex macht selbst

- Hypothesenpriorisierung und Entscheidung, welches Experiment zulässig ist
- Prüfung auf Leakage, Nested-CV-Fehler, OOF-Ausrichtung und faire Baselines
- Änderungen an der kanonischen Pipeline sowie an finalen Artefakten
- Interpretation knapper F1-Deltas, Paired-Tests und Stop/Keep-Entscheidungen
- finale Code-Reviews, Dokumentation und Abgabeentscheidung
- alle Aktionen mit destruktiver Wirkung oder externer Kommunikation

Vor Delegation immer Inputs, erlaubte Dateien, erwartetes Outputformat, Seed, Zeitbudget
und Akzeptanzkriterium nennen. Lokale Modellantworten gelten als Vorschlag, nie als Beleg.

## Scratchpad-Template pro Loop

```markdown
## Run YYYY-MM-DD_HHMM — <kurzer Name>

### Hypothese
Wenn <eine Änderung>, dann verbessert sich outer-CV F1, weil <Mechanismus>.

### Vorabentscheidung
- Baseline/Commit:
- Primäre Metrik: paired outer-CV F1 vs. 0.6938
- Akzeptanz: Delta > 2×SE und kein Fold-/Seed-Kollaps
- Sekundär: Precision, Recall, Positivrate, Laufzeit, OOF-Korrelation
- Max. Budget:

### Änderung
- Dateien/Diff:
- Features:
- Modell/Hyperparameter:
- Seeds und Splits:
- Delegiert an: Claude/Codex | qwen3:32b

### Leakage-Check
- Train/Validation/Exam-Trennung:
- Preprocessing nur auf Fold-Train gefittet:
- OOF-Reihenfolge und Labels geprüft:
- Threshold nur auf Fold-Train gewählt:

### Ergebnis
- Baseline F1 (mean ± std):
- Kandidat F1 (mean ± std):
- Paired Delta (mean, SE, n):
- Precision / Recall / Positivrate:
- Laufzeit:
- Artefakte/Logs:

### Interpretation
- Vermuteter Mechanismus bestätigt?
- Fehlerdiversität/OOF-Korrelation:
- Risiken oder Messbias:

### Entscheidung
KEEP | REJECT | FOLLOW-UP

### Nächster kleinster Test
<genau ein Test>
```
