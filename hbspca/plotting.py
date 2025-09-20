from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_cumulative_variance(var_df: pd.DataFrame, title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(var_df)+1)),
                             y=var_df["Cumulative EVR"],
                             mode="lines+markers",
                             name="Cumulative variance"))
    fig.update_layout(title=title, xaxis_title="Number of PCs",
                      yaxis_title="Cumulative explained variance ratio",
                      template="plotly_white", height=420)
    return fig

def make_scores_scatter(scores_df: pd.DataFrame, pcs: Tuple[int, int], color_by: Optional[str] = None,
                        title: str = "", opacity: float = 0.3, randomize_trace_order: bool = False,
                        use_gl: bool = True) -> go.Figure:
    p1, p2 = pcs
    xlab, ylab = f"PC{p1}", f"PC{p2}"
    Trace = go.Scattergl if use_gl else go.Scatter
    fig = go.Figure()

    if scores_df.empty or xlab not in scores_df.columns or ylab not in scores_df.columns:
        fig.update_layout(title=title, xaxis_title=xlab, yaxis_title=ylab, template="plotly_white", height=600)
        return fig

    df = scores_df.copy()
    if "point_name" not in df.columns:
        if "ID" in df.columns:
            df["point_name"] = df["ID"].astype(str)
        elif "Province" in df.columns:
            df["point_name"] = df["Province"].astype(str)
        elif "County" in df.columns:
            df["point_name"] = df["County"].astype(str)
        else:
            df["point_name"] = df.index.astype(str)

    if color_by and color_by in df.columns:
        col = color_by
        df[col] = df[col].astype(object).where(df[col].notna(), "(missing)")
        groups = list(df.groupby(col, dropna=False))
        if randomize_trace_order and len(groups) > 1:
            groups = sorted(groups, key=lambda kv: hash(str(kv[0])) % 997)
        for key, grp in groups:
            fig.add_trace(Trace(x=grp[xlab], y=grp[ylab], mode="markers", name=str(key),
                                marker=dict(size=5, opacity=opacity),
                                text=grp["point_name"],
                                hovertemplate="<b>%{text}</b><br>"
                                              f"{xlab}=%{{x:.3f}}<br>{ylab}=%{{y:.3f}}"
                                              "<extra>%{fullData.name}</extra>"))
    else:
        fig.add_trace(Trace(x=df[xlab], y=df[ylab], mode="markers", name="points",
                            marker=dict(size=5, opacity=opacity),
                            text=df["point_name"],
                            hovertemplate="<b>%{text}</b><br>"
                                          f"{xlab}=%{{x:.3f}}<br>{ylab}=%{{y:.3f}}"
                                          "<extra></extra>"))
    fig.update_layout(title=title, xaxis_title=xlab, yaxis_title=ylab, template="plotly_white",
                      height=600, legend=dict(itemsizing="trace", traceorder="normal"))
    return fig

def make_biplot(scores_df: pd.DataFrame, load_df: pd.DataFrame, pcs: Tuple[int, int], title: str = "",
                color_by: Optional[str] = None, opacity: float = 0.3, randomize_trace_order: bool = False) -> go.Figure:
    p1, p2 = pcs
    xlab, ylab = f"PC{p1}", f"PC{p2}"
    fig = make_scores_scatter(scores_df, pcs, color_by=color_by, title=title,
                              opacity=opacity, randomize_trace_order=randomize_trace_order)
    if not load_df.empty:
        for _, r in load_df.iterrows():
            fig.add_trace(go.Scatter(x=[0, r.get(xlab, 0.0)], y=[0, r.get(ylab, 0.0)],
                                     mode="lines+markers+text",
                                     text=[None, r.get("feature", "")],
                                     textposition="top center",
                                     marker=dict(size=[0, 4]), line=dict(width=1),
                                     showlegend=False, hoverinfo="text"))
    fig.update_layout(xaxis_title=xlab, yaxis_title=ylab)
    return fig



from typing import Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def make_scores_scatter3d(
    scores_df: pd.DataFrame,
    pcs: Tuple[int, int, int],
    color_by: Optional[str] = None,
    title: str = "",
    opacity: float = 0.3,
    randomize_trace_order: bool = False,
) -> go.Figure:
    p1, p2, p3 = pcs
    xlab, ylab, zlab = f"PC{p1}", f"PC{p2}", f"PC{p3}"
    fig = go.Figure()

    if scores_df.empty or any(c not in scores_df.columns for c in (xlab, ylab, zlab)):
        fig.update_layout(title=title, template="plotly_white", height=700,
                          scene=dict(xaxis_title=xlab, yaxis_title=ylab, zaxis_title=zlab))
        return fig

    df = scores_df.copy()
    if "point_name" not in df.columns:
        if "ID" in df.columns:
            df["point_name"] = df["ID"].astype(str)
        elif "Province" in df.columns:
            df["point_name"] = df["Province"].astype(str)
        elif "County" in df.columns:
            df["point_name"] = df["County"].astype(str)
        else:
            df["point_name"] = df.index.astype(str)

    if color_by and color_by in df.columns:
        col = color_by
        df[col] = df[col].astype(object).where(df[col].notna(), "(missing)")
        groups = list(df.groupby(col, dropna=False))
        if randomize_trace_order and len(groups) > 1:
            idx = np.random.permutation(len(groups))
            groups = [groups[i] for i in idx]
        for key, grp in groups:
            fig.add_trace(go.Scatter3d(
                x=grp[xlab], y=grp[ylab], z=grp[zlab],
                mode="markers", name=str(key),
                marker=dict(size=3, opacity=opacity),
                text=grp["point_name"],
                hovertemplate="<b>%{text}</b><br>"
                              f"{xlab}=%{{x:.3f}}<br>"
                              f"{ylab}=%{{y:.3f}}<br>"
                              f"{zlab}=%{{z:.3f}}"
                              "<extra>%{fullData.name}</extra>",
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=df[xlab], y=df[ylab], z=df[zlab],
            mode="markers", name="points",
            marker=dict(size=3, opacity=opacity),
            text=df["point_name"],
            hovertemplate="<b>%{text}</b><br>"
                          f"{xlab}=%{{x:.3f}}<br>"
                          f"{ylab}=%{{y:.3f}}<br>"
                          f"{zlab}=%{{z:.3f}}"
                          "<extra></extra>",
        ))

    fig.update_layout(
        title=title, template="plotly_white", height=700,
        scene=dict(xaxis_title=xlab, yaxis_title=ylab, zaxis_title=zlab),
        legend=dict(itemsizing="trace", traceorder="normal"),
    )
    return fig


def make_biplot3d(
    scores_df: pd.DataFrame,
    load_df: pd.DataFrame,
    pcs: Tuple[int, int, int],
    title: str = "",
    color_by: Optional[str] = None,
    opacity: float = 0.3,
    randomize_trace_order: bool = False,
) -> go.Figure:
    # Start with colored 3D scores
    fig = make_scores_scatter3d(
        scores_df, pcs, color_by=color_by, title=title,
        opacity=opacity, randomize_trace_order=randomize_trace_order,
    )

    p1, p2, p3 = pcs
    xlab, ylab, zlab = f"PC{p1}", f"PC{p2}", f"PC{p3}"
    if not load_df.empty and all(c in load_df.columns for c in (xlab, ylab, zlab)):
        for _, r in load_df.iterrows():
            fig.add_trace(go.Scatter3d(
                x=[0, r.get(xlab, 0.0)],
                y=[0, r.get(ylab, 0.0)],
                z=[0, r.get(zlab, 0.0)],
                mode="lines+markers+text",
                text=[None, r.get("feature", "")],
                textposition="top center",
                marker=dict(size=[0, 2]),
                line=dict(width=1),
                showlegend=False,
                hovertext=r.get("feature", ""),
            ))
    return fig
