Toxic Machine Learning Project Handbook
WWI2023F
Prof. Dr. Alexander Brandt
alexander.brandt@dhbw-stuttgart.de
Version vom 22. Mai 2026
www.dhbw-stuttgart.de
Prüfungsleistungen
Sie erhalten einen individualisierten Datensatz inklusive Label
(toxic_data_bbbbb.csv), auf dem Sie ein ML-Modell trainieren sowie einen
individualisierten Testdatensatz (toxic_exam_bbbbb.csv), für welchen Sie
Vorhersagen treffen.
Insgesamt werden 60 Punkte vergeben:
1. Korrekte Vorhersage der Label zu toxic_exam_bbbbb.csv (Abgabe bis 9.7.2026,
23:59), gemessen als f1-Score der positiven Klasse (max. 36 Punkte)
2. Kurztest zur Funktionsweise Ihres Codes am (10.7.2026) (max. 24 Punkte)
Prof. Dr. Alexander Brandt
Toxic Machine Learning Project Handbook –
2 / 5
Bewertung
Bewertungsmaßstab „F1-Score der Positiv-Klasse“
0
Prof’s RF
Prof’s MLP
0 Punkte
18 Punkte
36 Punkte
Zwischen den Markierungen wird linear interpoliert.
Bewertungsmaßstab wird ggf. den Kursleistungen angepasst.
Prof. Dr. Alexander Brandt
Toxic Machine Learning Project Handbook –
3 / 5
Termine
• Auftakt am 22.5.2026: Besprechung der Aufgabenstellung, Zuordnung der
individualisierten Datensätze, erste Datenexploration
• Vier Termine vorwiegend online: Gemeinsamer Auftakt, Breakouts für
Gruppendiskussionen mit Präsentation der Findings, Breakouts für individuelle
Fragen (“Speed-Coaching”), gemeinsamer Abschluss
• Kurztest am 10.7.2026, danach gemeinsame Reflektion der Beobachtungen und
Auflösung der Datengenerierung
Prof. Dr. Alexander Brandt
Toxic Machine Learning Project Handbook –
4 / 5
Worauf es hier besonders ankommt
• Ziele festlegen: Von Beginn an die richtige Metrik im Blick
• Daten verstehen: Datenqualität, Verteilungen, Korrelationen, Imbalance
• Daten präparieren: Datensplit, Fehlende Werte, Ausreißer, neue Features
• Daten transformieren: Skalierung, Datenaugmentation
• Modellieren: Geeignetes Modell auswählen, Hyperparameter festlegen
• Training: Hyperparameter-Optimierung, Overfitting vermeiden
• Evaluation: Performance auf (eigenem) Test-Set
• Interpretieren: Feature Importance, Erklärungsmethoden
• Dokumentation: Recap der wichtigsten Schritte (vorbereitung für den Kurztest)
In dem Kurztest geht es genau um diese Punkte. Sie dürfen ihre Dokumentation
ausgedruckt mitbringen und verwenden!
Prof. Dr. Alexander Brandt
Toxic Machine Learning Project Handbook –
5 / 5




---- Additional 

Kurztest vorwiegend multiple choice
Fragen hauptsächlich zu Daten & Evaluation: e.g., welche Variable zeigte bei Ihnen eine auffällige Rechtsschiefe?
wir dürfen & sollen alle Informationen mitbringen, die in einen Report reingehören (auf Papier), so viel wie wir wollen (keine Einschränkungen)
wir müssen nur die predictions abgeben, nicht den ganzen code
MLP ist nicht wirklich stark; wir kommen locker dadrüber
Achtung: nicht Codierung (00 11) durcheinanderbringen
 