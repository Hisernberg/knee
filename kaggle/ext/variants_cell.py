# ======================= [extension] blend variants from saved member outputs =======================
# Runs AFTER the unmodified Speedy Raptors graph has published /kaggle/working/submission.csv (the
# exact public 0.943 anchor). Everything here is CPU-only arithmetic on files the graph already wrote:
#   raptor_raw.npz                        raw probabilities of every Raptor view (anchor views + extra views)
#   _raptor_public.csv                    anchor public-Raptor ranks (4 original views, 0.60/0.10/0.10/0.20)
#   _coat_family_rank.csv                 anchor CoAt family rank (resgated / Global96 / D4, equal prob mean)
#   coat_*_predictions.npz                per-member CoAt probabilities (for the quality-weighted family)
#   btk_v32_rest_input.npz                the transformer/Rad stack (DINO + A5 + RadImageNet, calibrated)
# The anchor is copied to submission_anchor.csv first and is never modified. Every variant is validated
# against the competition contract before it is written. submission.csv is replaced by the MAIN variant
# only if MAIN was computed and validated; otherwise the anchor stays as submission.csv.
import json as _xj, os as _xo, shutil as _xsh, traceback as _xtb, time as _xtime
import numpy as _xnp, pandas as _xpd
from pathlib import Path as _XP

_X_WORK = _XP(_xo.environ.get('RSNA_EXT_WORK', '/kaggle/working'))
_X_LABELS = list(_KE_LAB)
_X_IDS = [str(u) for u in _RSNA_TEST_IDS]
_X_ARMS = [dict(a) for a in _KE_NS['ARMS']]
_X_FAILED = set(globals().get('_EXT_FAILED_ARMS', []))
_X_RECEIPT = {'schema': 'rsna_knee_ext_variants_v1', 'studies': len(_X_IDS), 'arms': [a['name'] for a in _X_ARMS],
              'failed_extra_arms': sorted(_X_FAILED), 'variants': {}, 'parity': {}, 'started': _xtime.time()}

_X_ANCHOR_W = {'maxspan-v5': 0.60, 'native384dense-v10': 0.10, 'maxspan-v5-reverse': 0.10, 'native384-v8': 0.20}
_X_MAIN_W = {'maxspan-v5': 0.40, 'native384dense-v10': 0.08, 'maxspan-v5-reverse': 0.08, 'native384-v8': 0.14,
             'finespacing-v9': 0.20, 'widedense-v4': 0.10}
_X_BAL_W = {'maxspan-v5': 0.22, 'native384dense-v10': 0.12, 'maxspan-v5-reverse': 0.08, 'native384-v8': 0.18,
            'finespacing-v9': 0.22, 'widedense-v4': 0.18}
_X_FAMILY_Q = {'global96_top3': 0.45, 'd4_swa3': 0.30, 'resgated_top3': 0.25}   # gold-58: 0.930 / n.a. / 0.909
_X_OUTER_ANCHOR = {label: float(_coatnet_weight[label]) for label in _X_LABELS}
_X_OUTER_FLAT = {label: 0.65 for label in _X_LABELS}


def _x_rankpct(x):
    """Per-column percentile rank in [0,1]; identical to the Raptor worker's rankpct."""
    order = x.argsort(0).argsort(0).astype(_xnp.float64)
    return order / max(1, (x.shape[0] - 1))


def _x_pdrank(x):
    """pandas average percentile rank, as used by every blend stage of the notebook."""
    return _xpd.DataFrame(_xnp.asarray(x, dtype=_xnp.float64)).rank(method='average', pct=True).to_numpy(_xnp.float64)


def _x_align(uids, arr):
    pos = {str(u): i for i, u in enumerate(uids)}
    return arr[[pos[u] for u in _X_IDS]]


def _x_load_npz(path, keys):
    with _xnp.load(path, allow_pickle=False) as z:
        uids = [str(u) for u in z['study_uids'].astype(str)]
        for key in keys:
            if key in z.files:
                arr = _xnp.asarray(z[key], dtype=_xnp.float64)
                break
        else:
            raise KeyError(f'{path}: none of {keys}')
    return uids, arr


def _x_csv_values(path):
    frame = _xpd.read_csv(path, dtype={'StudyInstanceUID': str})
    if frame.columns.tolist() != ['StudyInstanceUID', *_X_LABELS]:
        raise RuntimeError(f'{path}: schema drift')
    return frame.set_index('StudyInstanceUID').loc[_X_IDS, _X_LABELS].to_numpy(_xnp.float64)


def _x_public_rank_from_raw(raw, weights):
    """Weighted probability mean over views -> rankpct (the worker's formula) -> pandas pct rank (the blend's)."""
    names = [a['name'] for a in _X_ARMS]
    use = {n: w for n, w in weights.items() if n in names and n not in _X_FAILED and w > 0}
    if not use:
        raise RuntimeError('no usable Raptor view for this weight set')
    total = float(sum(use.values()))
    blend = _xnp.zeros((len(_X_IDS), len(_X_LABELS)), _xnp.float64)
    for name, w in use.items():
        blend += (w / total) * _xnp.clip(raw[names.index(name)], 0, 1)
    return _x_pdrank(_x_rankpct(blend).astype(_xnp.float32).astype(_xnp.float64)), {n: w / total for n, w in use.items()}


def _x_final(transformer, public_rank, private_rank, alpha, outer):
    hybrid = (1.0 - alpha) * public_rank + alpha * private_rank
    tr = _x_pdrank(transformer)
    out = _xnp.empty_like(tr)
    for j, label in enumerate(_X_LABELS):
        w = float(outer[label])
        out[:, j] = (1.0 - w) * tr[:, j] + w * hybrid[:, j]
    return _x_pdrank(out)


def _x_write(name, values, meta):
    values = _xnp.asarray(values, dtype=_xnp.float64)
    if values.shape != (len(_X_IDS), len(_X_LABELS)) or not _xnp.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise RuntimeError(f'{name}: invalid prediction matrix')
    frame = _xpd.DataFrame(values, columns=_X_LABELS)
    frame.insert(0, 'StudyInstanceUID', _X_IDS)
    path = _X_WORK / f'submission_{name}.csv'
    tmp = path.with_suffix('.csv.tmp')
    frame.to_csv(tmp, index=False)
    check = _xpd.read_csv(tmp, dtype={'StudyInstanceUID': str})
    if check.columns.tolist() != ['StudyInstanceUID', *_X_LABELS] or check['StudyInstanceUID'].tolist() != _X_IDS \
            or not _xnp.isfinite(check[_X_LABELS].to_numpy(_xnp.float64)).all():
        raise RuntimeError(f'{name}: CSV round trip failed')
    _xo.replace(tmp, path)
    meta = dict(meta); meta['path'] = str(path)
    _X_RECEIPT['variants'][name] = meta
    print(f'[ext] wrote {path}  {meta}', flush=True)
    return path


_x_main_path = None
try:
    _x_anchor_path = _X_WORK / 'submission_anchor.csv'
    if not _x_anchor_path.is_file():   # idempotent: the anchor copy is taken exactly once, right after the graph publishes it
        _xsh.copyfile(_X_WORK / 'submission.csv', _x_anchor_path)
    _X_RECEIPT['variants']['anchor'] = {'path': str(_x_anchor_path), 'note': 'exact Speedy Raptors output (copied, untouched)'}

    _x_tr_uids, _x_tr = _x_load_npz(_X_WORK / 'btk_v32_rest_input.npz', ['values'])
    _x_transformer = _x_align(_x_tr_uids, _x_tr)
    _x_raw_uids, _x_raw = _x_load_npz(_X_WORK / 'raptor_raw.npz', ['raw_probabilities'])
    if _x_raw.shape[0] != len(_X_ARMS):
        raise RuntimeError(f'raptor_raw has {_x_raw.shape[0]} views, ARMS has {len(_X_ARMS)}')
    _x_raw = _xnp.stack([_x_align(_x_raw_uids, _x_raw[i]) for i in range(_x_raw.shape[0])])
    _x_family_anchor = _x_pdrank(_x_csv_values(_X_WORK / '_coat_family_rank.csv'))
    _x_public_anchor_csv = _x_pdrank(_x_csv_values(_X_WORK / '_raptor_public.csv'))
    _x_anchor_written = _x_csv_values(_x_anchor_path)

    # ---- parity: the anchor must be reproducible from its parts before any variant is trusted ----
    _x_public_anchor_raw, _ = _x_public_rank_from_raw(_x_raw, _X_ANCHOR_W)
    _x_par_public = float(_xnp.abs(_x_public_anchor_raw - _x_public_anchor_csv).max())
    _x_anchor_recomputed = _x_final(_x_transformer, _x_public_anchor_csv, _x_family_anchor, 0.40, _X_OUTER_ANCHOR)
    _x_par_final = float(_xnp.abs(_x_anchor_recomputed - _x_anchor_written).max())
    _X_RECEIPT['parity'] = {'public_raptor_max_abs_diff': _x_par_public, 'final_max_abs_diff': _x_par_final}
    print(f'[ext] anchor parity: public raptor {_x_par_public:.2e}, final {_x_par_final:.2e}', flush=True)
    if _x_par_final > 1e-6:
        raise RuntimeError('anchor could not be reproduced from its parts; variants are not trustworthy')

    # ---- quality-weighted CoAt family (falls back to the anchor family if a member file is missing) ----
    _x_members = {
        'resgated_top3': (_X_WORK / 'coat_resgated_ep10_top3_predictions.npz', ['probability_mean', 'raw_probabilities']),
        'global96_top3': (_X_WORK / 'coatnet_global96_top3_predictions.npz', ['probability_mean', 'raw_probabilities']),
        'd4_swa3': (_X_WORK / 'd4_input' / 'coatnet_d4_depthzone_swa3_predictions.npz', ['probability_mean', 'raw_probabilities']),
    }
    _x_member_prob = {}
    for _x_name, (_x_path, _x_keys) in _x_members.items():
        try:
            _x_u, _x_a = _x_load_npz(_x_path, _x_keys)
            if _x_a.ndim == 3:
                _x_a = _x_a.mean(axis=0)
            _x_member_prob[_x_name] = _xnp.clip(_x_align(_x_u, _x_a), 0, 1)
        except Exception as _x_exc:
            print(f'[ext] CoAt member {_x_name} unavailable for re-weighting: {type(_x_exc).__name__}: {_x_exc}', flush=True)
    if len(_x_member_prob) >= 2:
        _x_qw = {n: _X_FAMILY_Q[n] for n in _x_member_prob}
        _x_qs = float(sum(_x_qw.values()))
        _x_family_q = _x_pdrank(sum((_x_qw[n] / _x_qs) * _x_member_prob[n] for n in _x_member_prob))
        _x_family_q_note = {n: _x_qw[n] / _x_qs for n in _x_qw}
    else:
        _x_family_q, _x_family_q_note = _x_family_anchor, 'anchor family (members unavailable)'

    # ---- variants ----
    _x_public_main, _x_main_w = _x_public_rank_from_raw(_x_raw, _X_MAIN_W)
    _x_public_bal, _x_bal_w = _x_public_rank_from_raw(_x_raw, _X_BAL_W)
    _x_main_path = _x_write('main', _x_final(_x_transformer, _x_public_main, _x_family_anchor, 0.40, _X_OUTER_ANCHOR),
                            {'raptor_weights': _x_main_w, 'family': 'anchor equal prob-mean', 'alpha': 0.40, 'outer': 'anchor per-target'})
    _x_write('c3', _x_final(_x_transformer, _x_public_bal, _x_family_q, 0.45, _X_OUTER_ANCHOR),
             {'raptor_weights': _x_bal_w, 'family': _x_family_q_note, 'alpha': 0.45, 'outer': 'anchor per-target'})
    _x_write('c4', _x_final(_x_transformer, _x_public_main, _x_family_anchor, 0.40, _X_OUTER_FLAT),
             {'raptor_weights': _x_main_w, 'family': 'anchor equal prob-mean', 'alpha': 0.40, 'outer': 'flat 0.65'})
    _x_write('c5', _x_final(_x_transformer, _x_public_main, _x_family_anchor, 0.50, _X_OUTER_ANCHOR),
             {'raptor_weights': _x_main_w, 'family': 'anchor equal prob-mean', 'alpha': 0.50, 'outer': 'anchor per-target'})
    _X_RECEIPT['status'] = 'COMPLETE'
except Exception as _x_error:
    _X_RECEIPT['status'] = 'FAILED'
    _X_RECEIPT['error'] = f'{type(_x_error).__name__}: {_x_error}'
    _X_RECEIPT['traceback'] = _xtb.format_exc()[-3000:]
    print('[ext] variants FAILED; the anchor stays as submission.csv:', _X_RECEIPT['error'], flush=True)
    _x_main_path = None

if _x_main_path is not None and (_X_WORK / 'submission_main.csv').is_file():
    _xsh.copyfile(_X_WORK / 'submission_main.csv', _X_WORK / 'submission.csv')
    _X_RECEIPT['submission_csv'] = 'main'
else:
    _X_RECEIPT['submission_csv'] = 'anchor'
_X_RECEIPT['elapsed_seconds'] = _xtime.time() - _X_RECEIPT.pop('started')
(_X_WORK / 'ext_variants_receipt.json').write_text(_xj.dumps(_X_RECEIPT, indent=2, sort_keys=True, default=str))
print('[ext] submission.csv =', _X_RECEIPT['submission_csv'], '| variants:', sorted(_X_RECEIPT['variants']), flush=True)
