"""Measured-data summaries and figures; no generated experimental scores."""
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from .core import CONDITIONS,degrade,stable_seed,make_prior,footprint
from .backend import localize

GROUP = ["scene_id","condition","sigma_xy_m","profile","retrieval_mode","reference_mode"]


def summarize_batch(frame, output, threshold_m=10.):
    """Separate pose accuracy from acceptance; include technical errors in counts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    groups = ['reference_mode','matching_method','sigma_xy_m','profile','retrieval_mode','condition']
    for keys, group in frame.groupby(groups, dropna=False, sort=False):
        error = pd.to_numeric(group.error_xy_m, errors='coerce')
        technical = group.status.eq('technical_error')
        accepted = group.accepted.eq(True) & ~technical
        pnp = group.get('valid_pnp', pd.Series(False, index=group.index)).eq(True) & ~technical
        accurate = pnp & error.le(threshold_m)
        row = dict(zip(groups, keys), n_attempted=len(group),
                   n_frames=len(group[['scene_id','sample_id']].drop_duplicates()),
                   n_technical=int(technical.sum()), n_pnp=int(pnp.sum()),
                   n_within_threshold=int(accurate.sum()),
                   n_correct=int((accepted & accurate).sum()),
                   n_false_accept=int(group.status.eq('false_accept').sum()),
                   n_rejected=int(group.status.eq('rejected').sum()),
                   n_rejected_within_threshold=int((group.status.eq('rejected') & accurate).sum()),
                   median_accepted_error_m=error[accepted].median(),
                   p95_accepted_error_m=error[accepted].quantile(.95))
        # All attempted rows form the denominator; counts above expose failures.
        row['correct_fraction_all'] = row['n_correct']/row['n_attempted']
        row['within_threshold_fraction_all'] = row['n_within_threshold']/row['n_attempted']
        rows.append(row)
    overview = pd.DataFrame(rows)
    overview.to_csv(output/'overview_by_condition.csv', index=False)
    reasons = (frame.loc[frame.status.eq('rejected')]
               .groupby(['scene_id','condition','rejection_reason'], dropna=False)
               .size().rename('attempts').reset_index())
    reasons.to_csv(output/'rejection_reasons.csv', index=False)
    return overview, reasons


def summarize(frame,output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    rows = []
    for keys,g in frame.groupby(GROUP,dropna=False):
        technical = g.status.eq("technical_error")
        evaluable = g.loc[~technical]
        n = len(evaluable)
        accepted = evaluable.loc[evaluable.accepted.eq(True)]
        errors = pd.to_numeric(accepted.error_xy_m,errors="coerce").dropna()
        row = dict(zip(GROUP,keys),n_attempted=len(g),n_algorithm_trials=n,
                   n_technical=int(technical.sum()),n_accepted=len(accepted),n_errors=len(errors))
        row["technical_fraction"] = float(technical.mean())
        for name,status in [("correct","correct"),("rejected","rejected"),("false_accept","false_accept")]:
            row["n_"+name] = int(evaluable.status.eq(status).sum())
            row[name+"_fraction"] = row["n_"+name]/n if n else np.nan
        row.update(error_mean_m=errors.mean(),error_median_m=errors.median(),
                   error_p95_m=errors.quantile(.95) if len(errors) else np.nan)
        for threshold in [3,5,10,20]:
            count = int((evaluable.accepted.eq(True)&pd.to_numeric(evaluable.error_xy_m,errors="coerce").le(threshold)).sum())
            row[f"acc_{threshold}m"] = count/n if n else np.nan
        for level in [25,50,75]:
            values = evaluable.get(f"recall_overlap_{level}",pd.Series(dtype=float)).dropna()
            row[f"n_recall_{level}"] = len(values)
            row[f"recall_overlap_{level}"] = values.astype(float).mean() if len(values) else np.nan
        for field in ["inliers","total_s","retrieval_s","feature_extraction_s","retrieval_online_s","gallery_encoding_s","matching_s","pnp_s"]:
            values = pd.to_numeric(evaluable.get(field,pd.Series(dtype=float)),errors="coerce").dropna()
            row[f"n_{field}"] = len(values)
            row[f"mean_{field}"] = values.mean()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output/"summary_by_condition.csv",index=False)
    # Explicit pairing: technical failures never masquerade as algorithm degradation.
    pair_keys = ["experiment_id","pair_id","retrieval_mode","reference_mode"]
    base = frame.loc[frame.condition.eq("clear"),pair_keys+["status","accepted","error_xy_m","inliers"]].copy()
    base = base.rename(columns={k:"clear_"+k for k in ["status","accepted","error_xy_m","inliers"]})
    paired = frame.merge(base,on=pair_keys,how="left",validate="many_to_one")
    paired["pair_valid"] = ~paired.status.eq("technical_error") & paired.clear_status.notna() & ~paired.clear_status.eq("technical_error")
    paired["delta_correct"] = np.where(paired.pair_valid,paired.status.eq("correct").astype(int)-paired.clear_status.eq("correct").astype(int),np.nan)
    paired["delta_error_xy_m"] = np.where(paired.pair_valid & paired.accepted.eq(True) & paired.clear_accepted.eq(True),
        pd.to_numeric(paired.error_xy_m,errors="coerce")-pd.to_numeric(paired.clear_error_xy_m,errors="coerce"),np.nan)
    paired["delta_inliers"] = np.where(paired.pair_valid,
        pd.to_numeric(paired.inliers,errors="coerce")-pd.to_numeric(paired.clear_inliers,errors="coerce"),np.nan)
    paired.to_csv(output/"paired_differences.csv",index=False)
    return summary,paired


def plot_results(frame,output):
    output = Path(output)/"figures"
    output.mkdir(parents=True,exist_ok=True)
    measurements = frame.loc[~frame.status.eq("technical_error")].copy()
    if measurements.empty: return []
    paths = []
    # Separate scenes, uncertainty, profiles and retrieval modes rather than hide pooling.
    for keys,g in measurements.groupby(["scene_id","sigma_xy_m","profile","retrieval_mode"]):
        scene,sigma,profile,mode = keys
        stem = f"{scene}-{sigma:g}-{profile}-{mode}"
        fig,axes = plt.subplots(2,2,figsize=(15,10))
        order = list(CONDITIONS)
        for level in [25,50,75]:
            col = f"recall_overlap_{level}"
            if col in g:
                axes[0,0].plot(order,g.groupby("condition")[col].mean().reindex(order),marker="o",label=f"{level}% overlap")
        axes[0,0].set_title("Recall@K: географічне перекриття")
        axes[0,0].legend()
        axes[0,0].set_ylim(0,1.05)
        if "inliers" in g:
            axes[0,1].plot(order,g.groupby("condition").inliers.mean().reindex(order),marker="o")
        axes[0,1].set_title("Середня кількість PnP inliers")
        fractions = pd.crosstab(g.condition,g.status,normalize="index").reindex(order).fillna(0)
        fractions.reindex(columns=["correct","rejected","false_accept"],fill_value=0).plot.bar(stacked=True,ax=axes[1,0])
        axes[1,0].set_title("Статуси / усі алгоритмічні спроби")
        for condition,h in g.loc[g.accepted.eq(True)].groupby("condition"):
            values = np.sort(pd.to_numeric(h.error_xy_m,errors="coerce").dropna())
            if len(values): axes[1,1].step(values,np.arange(1,len(values)+1)/len(values),label=condition,where="post")
        axes[1,1].set_title("Накопичена частка похибок прийнятих рішень")
        axes[1,1].set_xlabel("Горизонтальна похибка, м")
        if axes[1,1].lines: axes[1,1].legend()
        for ax in axes.flat:
            if ax is not axes[1,1]: ax.tick_params(axis="x",labelrotation=45)
            ax.grid(alpha=.2)
        fig.suptitle(f"{scene}; σXY={sigma:g} м; {profile}; {mode}; N={len(g)}")
        fig.tight_layout()
        path = output/(stem+".png")
        fig.savefig(path,dpi=150)
        paths.append(path)
        plt.close(fig)
    for keys,g in measurements.groupby(["scene_id","profile","retrieval_mode"]):
        fig,ax = plt.subplots(figsize=(8,5))
        for condition,h in g.groupby("condition"):
            values = h.assign(correct=h.status.eq("correct")).groupby("sigma_xy_m").correct.mean()
            ax.plot(values.index,values.values,marker="o",label=condition)
        ax.set(xlabel="σ горизонтальної координати, м",ylabel="Частка коректних локалізацій",ylim=(0,1.05),title="; ".join(map(str,keys)))
        ax.legend()
        fig.tight_layout()
        path = output/("uncertainty-"+"-".join(map(str,keys))+".png")
        fig.savefig(path,dpi=150)
        paths.append(path)
        plt.close(fig)
    return paths


def show_visibility(sample,config):
    seed = stable_seed(config["seed"],sample["scene_id"],sample["sample_id"],"smoke")
    fig,axes = plt.subplots(2,3,figsize=(15,9))
    smoke = []
    for ax,condition in zip(axes.flat,CONDITIONS):
        image,t,meta = degrade(sample["rgb"],condition,config,seed)
        ax.imshow(image)
        ax.set_title(f"{condition}; t={meta['transmission_mean']:.2f}")
        ax.axis("off")
        if condition.startswith("smoke"): smoke.append((condition,t))
    fig.tight_layout()
    fig2,axes2 = plt.subplots(1,2,figsize=(10,4))
    for ax,(name,t) in zip(axes2,smoke):
        im = ax.imshow(t,vmin=0,vmax=1,cmap="gray")
        ax.set_title(name+": пропускання")
        fig2.colorbar(im,ax=ax)
    return fig,fig2


def diagnostic_figures(backend,sample,reference,config,output):
    """Optional repeated diagnostic calls; kept outside scientific result CSV."""
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    prior,_ = make_prior(sample["xyz_enu"],sample["rotation"],config["sigma_xy_m"][0],
                        config["profiles"][config["active_profiles"][0]],stable_seed(config["seed"],"visual_diagnostic"))
    try:
        visible,center = footprint(sample["xyz_enu"],sample["rotation"],sample["K"],sample["dist"],
                                   sample["rgb"].shape,reference.plane_at(sample['xyz_enu'][:2]),center_crop=True)
    except ValueError as exc:
        if str(exc) != 'unsupported_prior_geometry':
            raise
        visible = None  # Matching diagnostics remain useful without a planar GT footprint.
    for condition in ["clear","smoke_strong"]:
        image,_,_ = degrade(sample["rgb"],condition,config,
            stable_seed(config["seed"],sample["scene_id"],sample["sample_id"],"smoke"))
        prediction = localize(backend,image,sample["K"],sample["dist"],reference,prior,
                              config["retrieval_modes"][0],123,capture=True)
        candidates = prediction["candidates"]
        fig,axes = plt.subplots(1,len(candidates)+1,figsize=(4*(len(candidates)+1),4),squeeze=False)
        axes[0,0].imshow(image)
        axes[0,0].set_title(condition)
        for ax,candidate in zip(axes[0,1:],candidates):
            r,co,h,w = [candidate[k] for k in ["row","col","height","width"]]
            ax.imshow(cv2.cvtColor(reference.bgr[r:r+h,co:co+w],cv2.COLOR_BGR2RGB))
            overlap = (visible.intersection(Polygon(candidate["polygon_enu"])).area/visible.area
                       if visible is not None else float('nan'))
            ax.set_title(f"rank {candidate['rank']}; score {candidate['score']:.3f}\noverlap {overlap:.2f} (оцінювання)")
        for ax in axes.flat: ax.axis("off")
        fig.tight_layout()
        fig.savefig(output/f"retrieval-{condition}.png",dpi=140)
        plt.close(fig)
        for candidate,result in zip(candidates,prediction["candidate_results"]):
            vis = result.get("visualization")
            if vis is None: continue
            r,co,h,w = [candidate[k] for k in ["row","col","height","width"]]
            patch = cv2.cvtColor(reference.bgr[r:r+h,co:co+w],cv2.COLOR_BGR2RGB)
            canvas = np.zeros((max(image.shape[0],h),image.shape[1]+w,3),np.uint8)
            canvas[:image.shape[0],:image.shape[1]] = image
            canvas[:h,image.shape[1]:] = patch
            fig,ax = plt.subplots(figsize=(14,6))
            ax.imshow(canvas)
            # Deterministic subset for legibility; no cherry-picking by GT.
            for a,b in list(zip(vis["query_pts"],vis["map_pts"]))[:80]:
                ax.plot([a[0],b[0]+image.shape[1]],[a[1],b[1]],color="lime",linewidth=.4)
            ax.set_title(f"{condition}; rank {candidate['rank']}; matches={result['raw_matches']}; inliers={result['inliers']}")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(output/f"matching-{condition}-rank{candidate['rank']}.png",dpi=140)
            plt.close(fig)
