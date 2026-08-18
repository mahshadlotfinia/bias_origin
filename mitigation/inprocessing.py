"""
mitigation/inprocessing.py
Created on June 29, 2026

@author: Mahshad Lotfinia
https://github.com/mahshadlotfinia
"""

import numpy as np

from analysis.heads import fit_standardizer, scaled_block, to_float64


def _restore_numpy_inf_aliases():
    for _name, _val in (("PINF", np.inf), ("NINF", -np.inf)):
        if not hasattr(np, _name):
            setattr(np, _name, _val)


_restore_numpy_inf_aliases()


def _group_ids(g):
    g = np.asarray(g, dtype=object).astype(str)
    levels = np.unique(g)
    return np.searchsorted(levels, g).astype(np.int64), len(levels)


def group_dro(X_tr, y_tr, g_tr, X_va, y_va, g_va, X_te, g_te, seed=0,
              epochs=80, lr=1e-2, eta=0.01):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    sc, Xd = fit_standardizer(to_float64(X_tr))
    Xtr = torch.tensor(Xd, dtype=torch.float32)
    del Xd
    ytr = torch.tensor(np.asarray(y_tr, np.float32))
    gid, n_groups = _group_ids(g_tr)
    gid = torch.tensor(gid)
    model = nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    q = torch.ones(n_groups) / n_groups
    model.train()
    for _ in range(epochs):
        logits = model(Xtr).squeeze(-1)
        li = bce(logits, ytr)
        lg = torch.stack([li[gid == k].mean() if (gid == k).any() else torch.tensor(0.0)
                          for k in range(n_groups)])
        with torch.no_grad():
            q = q * torch.exp(eta * lg.detach())
            q = q / q.sum()
        loss = (q * lg).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        va = torch.sigmoid(model(torch.tensor(scaled_block(sc, X_va),
                                              dtype=torch.float32)).squeeze(-1)).numpy()
        te = torch.sigmoid(model(torch.tensor(scaled_block(sc, X_te),
                                              dtype=torch.float32)).squeeze(-1)).numpy()
    return va, te


def adversarial(X_tr, y_tr, g_tr, X_va, y_va, g_va, X_te, g_te, seed=0,
                epochs=80, lr=1e-3, adv_lambda=1.0, hidden=64):
    import torch
    import torch.nn as nn
    from torch.autograd import Function
    torch.manual_seed(seed)

    class _GRL(Function):
        @staticmethod
        def forward(ctx, x, lamb):
            ctx.lamb = lamb
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad):
            return -ctx.lamb * grad, None

    sc, Xd = fit_standardizer(to_float64(X_tr))
    Xtr = torch.tensor(Xd, dtype=torch.float32)
    del Xd
    ytr = torch.tensor(np.asarray(y_tr, np.float32))
    gid, n_groups = _group_ids(g_tr)
    gid = torch.tensor(gid)

    predictor = nn.Linear(Xtr.shape[1], 1)
    adversary = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, n_groups))
    opt = torch.optim.Adam(list(predictor.parameters()) + list(adversary.parameters()),
                           lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    predictor.train(); adversary.train()
    for _ in range(epochs):
        logit = predictor(Xtr).squeeze(-1)
        pred_loss = bce(logit, ytr)
        adv_logits = adversary(_GRL.apply(logit.unsqueeze(-1), adv_lambda))
        adv_loss = ce(adv_logits, gid)
        loss = pred_loss + adv_loss
        opt.zero_grad(); loss.backward(); opt.step()
    predictor.eval()
    with torch.no_grad():
        va = torch.sigmoid(predictor(torch.tensor(scaled_block(sc, X_va),
                                                  dtype=torch.float32)).squeeze(-1)).numpy()
        te = torch.sigmoid(predictor(torch.tensor(scaled_block(sc, X_te),
                                                  dtype=torch.float32)).squeeze(-1)).numpy()
    return va, te


def fairlearn_reduction(X_tr, y_tr, g_tr, X_va, y_va, g_va, X_te, g_te, seed=0):
    from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
    from sklearn.linear_model import LogisticRegression
    sc, Xtr = fit_standardizer(to_float64(X_tr))
    base = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    eg = ExponentiatedGradient(base, EqualizedOdds())
    eg.fit(Xtr, np.asarray(y_tr), sensitive_features=np.asarray([str(v) for v in g_tr]))

    predictors = getattr(eg, "predictors_", getattr(eg, "_predictors", None))
    weights = getattr(eg, "weights_", getattr(eg, "_weights", None))
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)

    def _ensemble(X):
        Xs = scaled_block(sc, X)
        acc = np.zeros(Xs.shape[0])
        for w, p in zip(weights, predictors):
            if w <= 0:
                continue
            acc += w * p.predict_proba(Xs)[:, 1]
        return acc
    return _ensemble(X_va), _ensemble(X_te)
