"""
Upwork Job Prediction API
Endpoints:
  GET  /health           — model status + loaded features
  POST /predict          — early model (client history + keywords + exp_level + activity)
  POST /predict/live     — full model  (adds proposals + last_viewed)
"""

import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Load pipelines ────────────────────────────────────────────────────────────
BASE = Path(__file__).parent / "Dataset"

EARLY_PL = joblib.load(BASE / "early_pipeline.pkl")
LIVE_PL  = joblib.load(BASE / "pipeline.pkl")

# ── Shared lookup tables ──────────────────────────────────────────────────────
_LOC_ABBREV = {
    'usa': 'united states', 'u.s.': 'united states',
    'gbr': 'united kingdom', 'u.k.': 'united kingdom',
    'aus': 'australia', 'nld': 'netherlands',
    'can': 'canada', 'deu': 'germany', 'ind': 'india',
}
_EXP_MAP = {'entry level': 1, 'entry': 1, 'intermediate': 2, 'expert': 3}


def _g(job, key):
    """Safe numeric getter — returns 0.0 for missing/None."""
    return float(job.get(key) or 0)


def _location_dummies(job, pl):
    loc_raw  = str(job.get('client_location') or 'unknown').lower()
    loc_norm = _LOC_ABBREV.get(loc_raw, loc_raw)
    loc_grp  = loc_norm if loc_norm in pl['top_locs'] else 'other'
    loc_cols = [f for f in pl['all_features'] if f.startswith('loc_')]
    return {col: int(col == f'loc_{loc_grp}') for col in loc_cols}


def _kw_tfidf(job, pl):
    kw_str = str(job.get('keywords') or '')
    mat    = pl['vec_kw'].transform([kw_str])
    return {f'kw_{c}': v
            for c, v in zip(pl['vec_kw'].get_feature_names_out(), mat.toarray()[0])}


def _exp_features(job, pl):
    kw_str        = str(job.get('keywords') or '')
    kw_level_map  = pl.get('kw_level_map', {})
    exp_level_ord = _EXP_MAP.get(
        str(job.get('experience_level') or 'intermediate').lower().strip(), 2)
    kws    = [k.strip().lower() for k in kw_str.split(',') if k.strip()]
    levels = [kw_level_map[k] for k in kws if k in kw_level_map]
    keyword_avg_level = float(np.mean(levels)) if levels else 2.0
    level_mismatch    = abs(exp_level_ord - keyword_avg_level)
    return exp_level_ord, keyword_avg_level, level_mismatch


def _client_features(job):
    client_hire_rate   = _g(job, 'client_hire_rate')
    client_hires       = _g(job, 'client_hires')
    client_total_spent = _g(job, 'client_total_spent')
    client_jobs_posted = _g(job, 'client_jobs_posted')
    active             = _g(job, 'active')
    open_jobs          = _g(job, 'open_jobs')
    return dict(
        client_hire_rate   = client_hire_rate,
        client_is_reliable = int(client_hire_rate >= 80),
        client_hires       = client_hires,
        log_client_hires   = np.log1p(client_hires),
        client_total_spent = client_total_spent,
        log_client_spent   = np.log1p(client_total_spent),
        client_is_new      = int(client_hires == 0 and client_total_spent == 0),
        client_jobs_posted = client_jobs_posted,
        active             = active,
        hiring_capacity    = 1 / (active + 1),
        single_open_job    = int(open_jobs == 1),
        many_open_jobs     = int(open_jobs >= 5),
        open_jobs_capped   = min(open_jobs, 5),
    )


def _score(prob, threshold):
    signal   = ('Likely Hired'  if prob >= 0.70 else
                'Possible Hire' if prob >= 0.55 else
                'Borderline'    if prob >= threshold else
                'Unlikely Hired')
    decision = ('Apply' if prob >= 0.55 else
                'Watch' if prob >= threshold else
                'Skip')
    return signal, decision


# ── Early model predict ───────────────────────────────────────────────────────
def _predict_early(job: dict) -> dict:
    pl = EARLY_PL

    # Activity
    interviewing       = _g(job, 'interviewing')
    invites_sent       = _g(job, 'invites_sent')
    unanswered_invites = _g(job, 'unanswered_invites')
    answered_inv       = max(invites_sent - unanswered_invites, 0)

    exp_level_ord, keyword_avg_level, level_mismatch = _exp_features(job, pl)

    feat = {
        'interviewing'      : interviewing,
        'invites_sent'      : invites_sent,
        'unanswered_invites': unanswered_invites,
        'has_interviews'    : int(interviewing > 0),
        'has_invites'       : int(invites_sent > 0),
        'invite_reply_rate' : answered_inv / (invites_sent + 1),
        **_client_features(job),
        'exp_level_ord'     : exp_level_ord,
        'keyword_avg_level' : keyword_avg_level,
        'level_mismatch'    : level_mismatch,
        **_location_dummies(job, pl),
        **_kw_tfidf(job, pl),
    }

    X_row = pd.DataFrame([feat])[pl['all_features']].fillna(0)
    prob  = float(pl['model'].predict_proba(X_row)[0][1])
    threshold = pl.get('threshold', 0.5)
    signal, decision = _score(prob, threshold)

    return {
        'decision'   : decision,
        'signal'     : signal,
        'probability': round(prob, 4),
        'prediction' : int(prob >= threshold),
        'model'      : pl['model_name'],
        'threshold'  : round(threshold, 4),
    }


# ── Live model predict ────────────────────────────────────────────────────────
def _parse_last_viewed(val):
    if not val or str(val).strip() == '':
        return None
    val = str(val).lower().strip()
    if 'yesterday' in val: return 1.0
    if 'last week' in val: return 7.0
    if 'last month' in val: return 30.0
    m = re.search(r'(\d+)\s*(second|minute|hour|day|week|month)', val)
    if not m: return None
    n, unit = int(m.group(1)), m.group(2)
    return {'second': 1/86400, 'minute': 1/1440, 'hour': 1/24,
            'day': 1, 'week': 7, 'month': 30}[unit] * n


def _predict_live(job: dict) -> dict:
    pl = LIVE_PL

    # Proposals
    proposal_min = _g(job, 'proposal_min')
    proposal_max = _g(job, 'proposal_max')
    proposal_mid = (proposal_min + proposal_max) / 2

    # Last viewed
    last_viewed  = _parse_last_viewed(job.get('last_viewed_client', ''))
    days_viewed  = last_viewed if last_viewed is not None else 999
    recently_viewed = int(days_viewed <= 7)

    # Activity
    invites_sent       = _g(job, 'invites_sent')
    unanswered_invites = _g(job, 'unanswered_invites')
    interviewing       = _g(job, 'interviewing')
    answered_inv       = max(invites_sent - unanswered_invites, 0)

    # Budget
    hourly_rate = _g(job, 'hourly_rate')
    has_budget  = int(_g(job, 'estimated_budget') > 0 or hourly_rate > 0)

    exp_level_ord, keyword_avg_level, level_mismatch = _exp_features(job, pl)

    feat = {
        'has_budget'         : has_budget,
        'proposal_min'       : proposal_min,
        'proposal_max'       : proposal_max,
        'proposal_mid'       : proposal_mid,
        'recently_viewed'    : recently_viewed,
        'client_active_on_job': int(days_viewed <= 7 or invites_sent > 0 or interviewing > 0),
        'has_invites'        : int(invites_sent > 0),
        'invite_reply_rate'  : answered_inv / (invites_sent + 1),
        'has_interviews'     : int(interviewing > 0),
        **_client_features(job),
        'exp_level_ord'      : exp_level_ord,
        'keyword_avg_level'  : keyword_avg_level,
        'level_mismatch'     : level_mismatch,
        **_location_dummies(job, pl),
        **_kw_tfidf(job, pl),
    }

    X_row = pd.DataFrame([feat])[pl['all_features']].fillna(0)
    prob  = float(pl['model'].predict_proba(X_row)[0][1])
    threshold = pl.get('threshold', 0.5)

    is_new = days_viewed >= 999 and invites_sent == 0 and interviewing == 0
    if is_new:
        ch = _client_features(job)
        good_client = ch['client_hire_rate'] >= 65 and ch['client_hires'] >= 5
        rich_client = ch['client_total_spent'] >= 1000
        signal   = ('New Job — Good Client'  if (good_client and rich_client) else
                    'New Job — Decent Client' if good_client else
                    'New Job — Weak Client'   if ch['client_hires'] >= 1 else
                    'New Job — No History')
        decision = ('Apply' if (good_client and rich_client) else
                    'Watch' if good_client else 'Skip')
    else:
        signal, decision = _score(prob, threshold)

    return {
        'decision'      : decision,
        'signal'        : signal,
        'probability'   : round(prob, 4),
        'prediction'    : int(prob >= threshold),
        'is_new_posting': is_new,
        'model'         : pl['model_name'],
        'threshold'     : round(threshold, 4),
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get('/health')
def health():
    return jsonify({
        'status' : 'ok',
        'models' : {
            'early': {
                'name'      : EARLY_PL['model_name'],
                'features'  : len(EARLY_PL['all_features']),
                'threshold' : round(EARLY_PL.get('threshold', 0.5), 4),
            },
            'live': {
                'name'      : LIVE_PL['model_name'],
                'features'  : len(LIVE_PL['all_features']),
                'threshold' : round(LIVE_PL.get('threshold', 0.5), 4),
            },
        }
    })


@app.post('/predict')
def predict_early():
    """
    Early model — client history + keywords + exp_level + activity (interviewing/invites).

    Required JSON fields:
      keywords, experience_level,
      interviewing, invites_sent, unanswered_invites,
      client_location, client_total_spent, client_hire_rate,
      client_hires, active, client_jobs_posted, open_jobs
    """
    job = request.get_json(silent=True)
    if not job:
        return jsonify({'error': 'JSON body required'}), 400
    try:
        result = _predict_early(job)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.post('/predict/live')
def predict_live():
    """
    Full live model — adds proposals + last_viewed to the early feature set.

    Additional fields over /predict:
      proposal_min, proposal_max, last_viewed_client,
      estimated_budget, hourly_rate
    """
    job = request.get_json(silent=True)
    if not job:
        return jsonify({'error': 'JSON body required'}), 400
    try:
        result = _predict_live(job)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
