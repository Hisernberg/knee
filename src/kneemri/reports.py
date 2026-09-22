"""Radiology report -> weak labels (train only; reports are NOT available at test time).

Only 58 of 4,407 training studies carry official labels; every study has a free-text report in one of
several languages (Spanish, Dutch, German, French, Portuguese, Italian, English, ...). This module gives a
transparent multilingual rule labeler producing a tri-state per finding:

    +1 explicit positive   (pathology term near the anatomy term, not negated)
    -1 explicit negative   (anatomy term with normal/intact wording or a negated pathology term)
     0 unmentioned

and a fold-safe calibration of those states into soft targets + confidence weights using the gold rows of
the training fold only (hierarchical shrinkage: per-(target,state) rates shrunk toward per-state rates
pooled across targets, with the ordering positive > unmentioned > negative enforced).

For higher-quality teachers use scripts/label_reports_llm.py (LLM labeling); both produce the same CSV
schema (StudyInstanceUID, <target>, <target>__w) consumed by TrainConfig.aux_labels_csv.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import ID_COL, TARGETS

# ----------------------------------------------------------------------------------------------------
# Multilingual vocabularies (EN, ES, NL, DE, FR, PT, IT). Accents are stripped before matching.
# ----------------------------------------------------------------------------------------------------
NEGATION = [r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bsin\b", r"\bgeen\b", r"\bniet\b", r"\bkein\w*", r"\bnicht\b",
            r"\bpas\b", r"\bsans\b", r"\bsem\b", r"\bnao\b", r"\bsenza\b", r"\bnon\b", r"\bausencia\b", r"\bnegativ\w*",
            r"\bafwezig\w*", r"\bausente\b", r"\bexclu\w*", r"\bdescart\w*", r"\bfree of\b", r"\bfrei\b",
            r"\bno evidence\b", r"\bno signs?\b", r"\bunremarkable\b", r"\bnormal\w*", r"\bintact\w*", r"\bintegr\w*",
            r"\bconserv\w*", r"\bpreserv\w*", r"\bregelrecht\w*", r"\bunauffallig\w*", r"\bgaaf\b", r"\bgave\b",
            r"\bnormaal\b", r"\bnormale\b", r"\bindemne\w*", r"\bintegro\b", r"\bintegra\b", r"\bpreservad\w*",
            r"\bnormaux\b", r"\bnenhum\w*", r"\bninguna?\b", r"\bnessun\w*", r"\bintakt\w*", r"\bunversehrt\w*",
            r"\bohne\b", r"\bzonder\b", r"\babsence\b", r"\bassenza\b", r"\bregolar\w*", r"\bunauffallig\w*",
            r"\bnormalt?\b", r"\bopgeheven\b", r"\bniet aangetoond\b", r"\bgeen aanwijzing\w*",
            # hr/sr/sl
            r"\bbez\b", r"\bnema\b", r"\bne\b", r"\bnije\b", r"\buredn\w*", r"\bocuvan\w*", r"\bnormaln\w*", r"\bni\b",
            # tr
            r"\byok\w*", r"\bizlenmedi\b", r"\bsaptanmadi\b", r"\bnormal\w*", r"\bdogal\w*", r"\bintakt\w*",
            # bg (cyrillic)
            r"\bняма\b", r"\bлипсва\w*", r"\bне се\b", r"\bбез\b", r"\bнормал\w*", r"\bзапазен\w*", r"\bинтакт\w*",
            # el (greek)
            r"\bδεν\b", r"\bχωρις\b", r"\bφυσιολογικ\w*", r"\bακεραι\w*", r"\bαπουσια\b"]
NEG_RX = re.compile("|".join(NEGATION))

PATHOLOGY_GENERIC = [r"\btear\w*", r"\btorn\b", r"\bruptur\w*", r"\brotur\w*", r"\brupt\w*", r"\bscheur\w*",
                     r"\briss\b", r"\bgerissen\b", r"\bdechir\w*", r"\bdesgarr\w*", r"\blesion\w*", r"\blaesie\w*",
                     r"\blesao\w*", r"\bletsel\w*", r"\bsprain\w*", r"\besguince\w*", r"\bdistors\w*",
                     r"\bthicken\w*", r"\bengros\w*", r"\bespes\w*", r"\bverdik\w*", r"\bverdick\w*", r"\bedema\w*",
                     r"\boedeem\w*", r"\bodem\w*", r"\babnormal\w*", r"\banormal\w*", r"\bafwijk\w*", r"\bpartial\w*",
                     r"\bparcial\w*", r"\bpartiel\w*", r"\bcomplete\b", r"\bcomplet[ae]\b", r"\bgrad[eo]\s*(3|iii|2|ii)\b",
                     r"\bdegenerat\w*", r"\bdegeneraci\w*", r"\bsignal\w*", r"\bsenal\b", r"\bsignaal\w*",
                     r"\bhyperintens\w*", r"\bhiperintens\w*", r"\bdiscontinu\w*", r"\bfibril\w*", r"\bmucoid\w*",
                     r"\bmyxoid\w*", r"\bbucket\w*", r"\basa de cubo\b", r"\bhengsel\w*", r"\bflap\b", r"\bradial\w*",
                     r"\bhorizontal\w*", r"\bverticale?\b", r"\bcomplex\w*", r"\bcomplej\w*", r"\bdisplac\w*",
                     r"\bdesplaz\w*", r"\bextru\w*", r"\bmaceraci\w*", r"\bmacerat\w*",
                     # hr/sr: ruptura, lezija, ostecenje, prekid, edem; tr: yirtik, rüptür, hasar, lezyon, ödem
                     r"\bruptur\w*", r"\blezij\w*", r"\bostecen\w*", r"\bprekid\w*", r"\bedem\w*", r"\bpromjen\w*",
                     r"\byirtik\w*", r"\bhasar\w*", r"\blezyon\w*", r"\bodem\w*", r"\bkopma\w*",
                     # bg: руптура, лезия, увреда, оток, разкъсване; el: ρηξη, βλαβη, οιδημα
                     r"\bруптур\w*", r"\bлези\w*", r"\bувред\w*", r"\bоток\w*", r"\bразкъс\w*", r"\bедем\w*",
                     r"\bρηξ\w*", r"\bβλαβ\w*", r"\bοιδημ\w*", r"\bκακωσ\w*"]
PATH_RX = re.compile("|".join(PATHOLOGY_GENERIC))

OA_TERMS = [r"\bosteoarthr\w*", r"\barthros\w*", r"\bartros\w*", r"\bgonarthro\w*", r"\bgonartro\w*", r"\bchondr\w*",
            r"\bcondr\w*", r"\bkraakbeen\w*", r"\bknorpel\w*", r"\bcartilag\w*", r"\bcartilage\b", r"\bosteophyt\w*",
            r"\bosteofit\w*", r"\bosteofyt\w*", r"\bdegenerat\w*", r"\bdegeneraci\w*", r"\bslijtage\w*",
            r"\bjoint space narrow\w*", r"\bpinzamiento\b", r"\bsubchondral\w*", r"\bsubcondral\w*",
            r"\bsubchondra\w*", r"\bfull.thickness\b", r"\bespesor completo\b", r"\bgrad[eo]\s*(3|4|iii|iv)\b",
            r"\bthinn\w*", r"\badelgaz\w*", r"\bfissur\w*", r"\bfisur\w*", r"\bulcer\w*", r"\berosi\w*",
            r"\bhondromalac\w*", r"\bhrskavic\w*", r"\bartroz\w*", r"\bosteofit\w*", r"\bkondromalaz\w*", r"\bkikirdak\w*",
            r"\bхрущял\w*", r"\bхондромалац\w*", r"\bартроз\w*", r"\bостеофит\w*", r"\bχονδρ\w*", r"\bοστεοφυτ\w*",
            r"\bοστεοαρθρ\w*", r"\bαρθρωσ\w*"]
OA_RX = re.compile("|".join(OA_TERMS))


@dataclass(frozen=True)
class FindingSpec:
    name: str
    anatomy: list[str]  # anatomy/finding terms (regex)
    pathology: list[str] | None = None  # None -> PATHOLOGY_GENERIC; for OA -> OA_TERMS; "self" -> presence is positive
    self_positive: bool = False  # the finding term itself denotes pathology (effusion, synovitis, cyst, fracture...)


SPECS: list[FindingSpec] = [
    FindingSpec("ACL", [r"\bacl\b", r"\banterior cruciate\b", r"\bcruzado anterior\b", r"\blca\b", r"\bvoorste kruisband\w*",
                        r"\bvkb\b", r"\bvordere\w* kreuzband\w*", r"\bcroise anterieur\b", r"\bcrociato anteriore\b",
                        r"\bcruzado anterior\b", r"\bligamento cruzado anterior\b", r"\bprednj\w* krizn\w*", r"\bpkl\b",
                        r"\bon capraz\b", r"\bocb\b", r"\bпредн\w* кръстн\w*", r"\bпкв\b", r"\bπροσθι\w* χιαστ\w*", r"\bπχσ\b"]),
    FindingSpec("MCL", [r"\bmcl\b", r"\bmedial collateral\b", r"\bcolateral (medial|interno|tibial)\b", r"\blcm\b",
                        r"\blli\b", r"\bmediale collaterale band\w*", r"\bmcb\b", r"\binnenband\w*", r"\binnenband",
                        r"\bmediale\w* seitenband\w*", r"\binnenseitenband\w*", r"\bcollateral interne\b", r"\blcm\b",
                        r"\bcollaterale mediale\b", r"\btibial collateral\b", r"\bmedijaln\w* kolateraln\w*", r"\bmkl\b",
                        r"\bic yan bag\w*", r"\bmedial kollateral\w*", r"\bвътрешн\w* колатерал\w*", r"\bмедиалн\w* колатерал\w*",
                        r"\bεσω πλαγι\w*", r"\bεπλσ\b"]),
    FindingSpec("Medial Meniscus", [r"\bmedial meniscus\b", r"\bmenisco (interno|medial)\b", r"\bmediale meniscus\b",
                                    r"\bmediale\w* meniskus\w*", r"\binnenmeniskus\w*", r"\bmenisque interne\b",
                                    r"\bmenisco mediale\b", r"\bmeniscus mediaal\w*", r"\bmediaal\b", r"\bmedial meniscal\b",
                                    r"\bmeniscusscheur.{0,12}mediaal\b", r"\bcuerno posterior del menisco interno\b",
                                    r"\bmedijaln\w* menisk\w*", r"\bic menisk\w*", r"\bmedial menisk\w*", r"\bмедиалн\w* мениск\w*",
                                    r"\bвътрешн\w* мениск\w*", r"\bεσω μηνισκ\w*"]),
    FindingSpec("Lateral Meniscus", [r"\blateral meniscus\b", r"\bmenisco (externo|lateral)\b", r"\blaterale meniscus\b",
                                     r"\blaterale\w* meniskus\w*", r"\baussenmeniskus\w*", r"\bmenisque externe\b",
                                     r"\bmenisco laterale\b", r"\bmeniscus lateraal\w*", r"\blateraal\b", r"\blateral meniscal\b",
                                     r"\bdiscoid\w*", r"\bdiscoide\w*", r"\blateraln\w* menisk\w*", r"\bdis menisk\w*",
                                     r"\blateral menisk\w*", r"\bлатерал\w* мениск\w*", r"\bвъншн\w* мениск\w*", r"\bεξω μηνισκ\w*"]),
    FindingSpec("Medial OA", [r"\bmedial (compartment|tibiofemoral|femorotibial)\b", r"\bfemorotibial (medial|interno)\b",
                              r"\bcompartimento (medial|interno)\b", r"\bmediale\w* compartiment\w*", r"\bmediale\w* kompartiment\w*",
                              r"\bcompartiment (medial|interne)\b", r"\bcompartimento mediale\b", r"\bmedial femoral condyle\b",
                              r"\bcondilo femoral (medial|interno)\b", r"\bmediale femurcondyl\w*", r"\bmedial tibial plateau\b",
                              r"\bplatillo tibial (medial|interno)\b", r"\bmedijaln\w* (kondil|kompartm|odjeljk|tibijal)\w*",
                              r"\bmedial (kondil|kompartm|tibial plato)\w*", r"\bмедиалн\w* (кондил|компартм|тибиал)\w*",
                              r"\bεσω (μηριαι|κνημιαι|διαμερισμ)\w*"], OA_TERMS),
    FindingSpec("Lateral OA", [r"\blateral (compartment|tibiofemoral|femorotibial)\b", r"\bfemorotibial (lateral|externo)\b",
                               r"\bcompartimento (lateral|externo)\b", r"\blaterale\w* compartiment\w*", r"\blaterale\w* kompartiment\w*",
                               r"\bcompartiment (lateral|externe)\b", r"\bcompartimento laterale\b", r"\blateral femoral condyle\b",
                               r"\bcondilo femoral (lateral|externo)\b", r"\blaterale femurcondyl\w*", r"\blateral tibial plateau\b",
                               r"\bplatillo tibial (lateral|externo)\b", r"\blateraln\w* (kondil|kompartm|odjeljk|tibijal)\w*",
                               r"\blateral (kondil|kompartm|tibial plato)\w*", r"\bлатерал\w* (кондил|компартм|тибиал)\w*",
                               r"\bεξω (μηριαι|κνημιαι|διαμερισμ)\w*"], OA_TERMS),
    FindingSpec("PF OA", [r"\bpatellofemoral\w*", r"\bfemoropatel\w*", r"\bpatelofemoral\w*", r"\bpatello.?femor\w*",
                          r"\bretropatel\w*", r"\bpatel\w*", r"\brotul\w*", r"\bkniescheibe\w*", r"\btrochle\w*",
                          r"\bпател\w*", r"\bепикондил\w*", r"\bεπιγονατιδ\w*", r"\bπατελ\w*"], OA_TERMS),
    FindingSpec("Effusion", [r"\beffusion\w*", r"\bderrame\w*", r"\bgewrichtsvocht\w*", r"\bhydrops\b", r"\bhydrarthros\w*",
                             r"\berguss\w*", r"\bepanchement\w*", r"\bversamento\w*", r"\bliquido articular\b",
                             r"\bvocht\w*", r"\bfluid\b", r"\bliquid\w*", r"\bjoint fluid\b", r"\bizljev\w*", r"\bizliv\w*",
                             r"\befuzij\w*", r"\befuzyon\w*", r"\bmayi\b", r"\bставен излив\w*", r"\bизлив\w*", r"\bυγρ\w*",
                             r"\bυδραρθρ\w*", r"\bσυλλογ\w*"], self_positive=True),
    FindingSpec("Synovitis", [r"\bsynovit\w*", r"\bsinovit\w*", r"\bsynovial (thicken|prolifer|hypertroph)\w*",
                              r"\bengrosamiento sinovial\b", r"\bsynoviale verdik\w*", r"\bsynovialis\w*", r"\bsinovij\w*",
                              r"\bсиновит\w*", r"\bυμενιτ\w*"], self_positive=True),
    FindingSpec("Baker's", [r"\bbaker\w*", r"\bpopliteal cyst\w*", r"\bquiste popl\w*", r"\bquiste de baker\b",
                            r"\bpopliteale cyste\w*", r"\bkyste popl\w*", r"\bcisti (di )?baker\b", r"\bcisto de baker\b",
                            r"\bbakerse cyste\w*", r"\bbaker.?zyste\w*", r"\bcyste van baker\b", r"\bbakerov\w*", r"\bбейкър\w*",
                            r"\bбекер\w*", r"\bκυστ\w* baker\w*", r"\bιγνυακ\w* κυστ\w*"], self_positive=True),
    FindingSpec("Contusion", [r"\bcontusi\w*", r"\bbone bruis\w*", r"\bbone marrow (edema|oedema)\w*", r"\bedema oseo\b",
                              r"\bedema de la medula\w*", r"\bedema medular\b", r"\bbeenmergoedeem\w*", r"\bbotcontusie\w*",
                              r"\bknochenmarkodem\w*", r"\bknochenkontusion\w*", r"\bbone marrow edema\b",
                              r"\boedeme osseux\b", r"\bcontusione ossea\b", r"\bedema osseo\b", r"\bmarrow (edema|oedema)\b",
                              r"\bbotoedeem\w*", r"\bbot.?oedeem\w*", r"\btrabecul\w* (edema|oedema|fract\w*)\b",
                              r"\bmicrofract\w*", r"\bkostan\w* edem\w*", r"\bkontuzij\w*", r"\bedem kostne\w*", r"\bkemik iligi odem\w*",
                              r"\bkontuzyon\w*", r"\bкостномозъчен едем\w*", r"\bконтузи\w*", r"\bοστικ\w* οιδημ\w*",
                              r"\bμυελικ\w* οιδημ\w*", r"\bθλασ\w*"], self_positive=True),
    FindingSpec("Fracture", [r"\bfractur\w*", r"\bfraktur\w*", r"\bfratt?ur\w*", r"\bfractuur\w*", r"\bfracturas?\b",
                             r"\bbroken\b", r"\bavulsi\w*", r"\bsegond\b", r"\bimpaction\w*", r"\bimpactaci\w*",
                             r"\bdepression fracture\b", r"\bosteochondral fract\w*", r"\bimpressionsfraktur\w*",
                             r"\bprijelom\w*", r"\bfraktur\w*", r"\bkirik\w*", r"\bфрактур\w*", r"\bсчупван\w*", r"\bκαταγμ\w*"],
                self_positive=True),
]
SPEC_BY_NAME = {s.name: s for s in SPECS}
assert [s.name for s in SPECS] == TARGETS

_CLAUSE_SPLIT = re.compile(r"[.;:\n\r]+|\s-\s|\*|\|")
_NEG_WINDOW = 80  # characters before/after a finding term where a negation cue flips polarity


def normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text.lower())


def _polarity(clause: str, span: tuple[int, int]) -> int:
    """+1 if the mention is asserted, -1 if negated/normal in its neighbourhood."""
    lo, hi = max(0, span[0] - _NEG_WINDOW), min(len(clause), span[1] + _NEG_WINDOW)
    before, after = clause[lo : span[0]], clause[span[1] : hi]
    # a negation cue before the term (e.g. 'no ACL tear', 'sin derrame', 'geen scheur') or right after
    # ('ACL intact', 'menisco integro', 'kruisband gaaf')
    if NEG_RX.search(before) or NEG_RX.search(after[:45]):
        return -1
    return 1


def label_report(text: str) -> dict[str, int]:
    """Tri-state (+1/-1/0) per finding for one report."""
    t = normalise_text(text)
    clauses = [c.strip() for c in _CLAUSE_SPLIT.split(t) if c.strip()]
    out = {s.name: 0 for s in SPECS}
    for spec in SPECS:
        anat_rx = re.compile("|".join(spec.anatomy))
        path_rx = OA_RX if spec.pathology is OA_TERMS else PATH_RX
        pos, neg = 0, 0
        for c in clauses:
            for m in anat_rx.finditer(c):
                pol = _polarity(c, m.span())
                if spec.self_positive:
                    # finding term itself = pathology; "small effusion" positive, "no effusion" negative
                    (pos if pol > 0 else neg).__class__  # noqa: B018 (keeps intent explicit)
                    if pol > 0:
                        pos += 1
                    else:
                        neg += 1
                    continue
                # anatomy term: need a pathology term nearby (same clause) to be positive
                lo, hi = max(0, m.start() - _NEG_WINDOW), min(len(c), m.end() + _NEG_WINDOW)
                window = c[lo:hi]
                pm = path_rx.search(window)
                if pm is not None and pol > 0:
                    # pathology found, check it is not itself negated ('no tear of the ACL' handled by pol; 'ACL: no tear')
                    p_abs = (lo + pm.start(), lo + pm.end())
                    if _polarity(c, p_abs) > 0:
                        pos += 1
                    else:
                        neg += 1
                elif pm is not None and pol < 0:
                    neg += 1
                elif pol < 0 or NEG_RX.search(window):
                    neg += 1
                else:
                    # anatomy mentioned without pathology or normality wording -> treat as unmentioned (weak)
                    pass
        out[spec.name] = 1 if pos > 0 else (-1 if neg > 0 else 0)
    return out


def label_reports(df: pd.DataFrame, report_col: str = "Report") -> pd.DataFrame:
    rows = []
    for sid, txt in zip(df[ID_COL].astype(str), df[report_col].fillna("")):
        r = label_report(txt)
        r[ID_COL] = sid
        rows.append(r)
    return pd.DataFrame(rows)[[ID_COL] + TARGETS]


# ----------------------------------------------------------------------------------------------------
# Fold-safe calibration of states -> soft targets and weights
# ----------------------------------------------------------------------------------------------------
STATE_ORDER = {1: "pos", 0: "unk", -1: "neg"}


def calibrate_states(states: pd.DataFrame, gold: pd.DataFrame, fit_ids: set[str], k: float = 4.0,
                     unk_weight: float = 0.25, min_margin: float = 0.05) -> pd.DataFrame:
    """Map tri-states to P(y=1 | state, target) using gold rows in `fit_ids` only (training fold).

    Hierarchical shrinkage: per-(target,state) rate <- pooled per-state rate across targets (k pseudo-counts),
    then enforce pos >= unk + margin >= neg + 2*margin. Returns a DataFrame with <target> (soft label) and
    <target>__w (confidence weight: 1.0 for explicit states, `unk_weight` for unmentioned).
    """
    g = gold.copy()
    g[ID_COL] = g[ID_COL].astype(str)
    g = g[g[ID_COL].isin(fit_ids)].set_index(ID_COL)
    s = states.set_index(ID_COL)
    joined = s.join(g[TARGETS], rsuffix="__y", how="inner")
    pooled = {}
    for st in (1, 0, -1):
        num = den = 0.0
        for t in TARGETS:
            m = (joined[t] == st) & joined[f"{t}__y"].notna()
            num += float((joined.loc[m, f"{t}__y"] > 0.5).sum())
            den += float(m.sum())
        pooled[st] = (num + 1.0) / (den + 2.0)
    out = states[[ID_COL]].copy()
    table = {}
    for t in TARGETS:
        rates = {}
        for st in (1, 0, -1):
            m = (joined[t] == st) & joined[f"{t}__y"].notna()
            n_pos = float((joined.loc[m, f"{t}__y"] > 0.5).sum())
            n = float(m.sum())
            rates[st] = (n_pos + k * pooled[st]) / (n + k)
        # ordering constraint pos > unk > neg
        rates[0] = min(rates[0], rates[1] - min_margin)
        rates[-1] = min(rates[-1], rates[0] - min_margin)
        rates = {st: float(np.clip(v, 0.01, 0.99)) for st, v in rates.items()}
        table[t] = rates
        out[t] = states[t].map(rates).astype(float)
        out[f"{t}__w"] = np.where(states[t] == 0, unk_weight, 1.0)
    out.attrs["calibration"] = table
    return out
