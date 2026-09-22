import numpy as np
import pandas as pd

from kneemri.reports import calibrate_states, label_report, label_reports
from kneemri.schema import ID_COL, TARGETS


def test_multilingual_rules():
    es = ("Técnica: RMN de la rodilla. Resultados: Rotura de menisco interno. Artrosis femorotibial medial. "
          "Derrame. Ligamento cruzado anterior íntegro. Sin quiste de Baker.")
    r = label_report(es)
    assert r["Medial Meniscus"] == 1 and r["Medial OA"] == 1 and r["Effusion"] == 1
    assert r["ACL"] == -1 and r["Baker's"] == -1 and r["Fracture"] == 0
    nl = "Voorste kruisband gaaf. Scheur van de mediale meniscus achterhoorn. Geen gewrichtsvocht. Beenmergoedeem tibia."
    r = label_report(nl)
    assert r["ACL"] == -1 and r["Medial Meniscus"] == 1 and r["Effusion"] == -1 and r["Contusion"] == 1
    en = "Complete tear of the ACL. No medial or lateral meniscal tear. Moderate joint effusion with synovitis. No fracture."
    r = label_report(en)
    assert r["ACL"] == 1 and r["Effusion"] == 1 and r["Synovitis"] == 1 and r["Fracture"] == -1
    de = "Vorderes Kreuzband intakt. Innenmeniskus Riss im Hinterhorn. Kein Erguss. Fraktur des Tibiaplateaus."
    r = label_report(de)
    assert r["ACL"] == -1 and r["Medial Meniscus"] == 1 and r["Effusion"] == -1 and r["Fracture"] == 1


def test_calibration_orders_states():
    rng = np.random.default_rng(0)
    n = 300
    ids = [f"s{i}" for i in range(n)]
    states = pd.DataFrame({ID_COL: ids, **{t: rng.choice([1, 0, -1], n, p=[0.3, 0.5, 0.2]) for t in TARGETS}})
    gold = pd.DataFrame({ID_COL: ids})
    for t in TARGETS:
        p = np.where(states[t] == 1, 0.85, np.where(states[t] == 0, 0.3, 0.1))
        gold[t] = (rng.random(n) < p).astype(float)
    gold.loc[gold.index[:200], TARGETS] = np.nan  # only 100 labelled rows, like the real gold subset
    cal = calibrate_states(states, gold, set(ids[200:]))
    for t in TARGETS:
        tab = cal.attrs["calibration"][t]
        assert tab[1] > tab[0] > tab[-1]
        assert set(cal[f"{t}__w"].unique()) <= {0.25, 1.0}
    lab = label_reports(pd.DataFrame({ID_COL: ["a"], "Report": ["ACL tear"]}))
    assert list(lab.columns) == [ID_COL] + TARGETS and lab.loc[0, "ACL"] == 1
