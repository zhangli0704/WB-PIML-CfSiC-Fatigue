def _metric_lookup(metrics: pd.DataFrame, model: str, key: str) -> float:
    return float(metrics.loc[metrics["model"].eq(model), key].iloc[0])


@dataclass(frozen=True)
class FigureSlideImage:


    title: str
    png_bytes: bytes
    image_kind: str


class FigureDeckCollector:


    def __init__(self, data_name: str) -> None:
        self.data_name = str(data_name)
        self.images: list[FigureSlideImage] = []

    @staticmethod
    def _panel_png(fig, ax, dpi: int = 240) -> bytes:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tight = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        width_in, height_in = fig.get_size_inches()
        pad_x = max(0.08, tight.width * 0.04)
        pad_y = max(0.08, tight.height * 0.06)
        crop = Bbox.from_extents(
            max(0.0, tight.x0 - pad_x),
            max(0.0, tight.y0 - pad_y),
            min(float(width_in), tight.x1 + pad_x),
            min(float(height_in), tight.y1 + pad_y),
        )
        buffer = BytesIO()
        fig.savefig(
            buffer, format="png", dpi=dpi, bbox_inches=crop,
            facecolor="white", edgecolor="none",
        )
        return buffer.getvalue()

    def add_figure(self, fig, output_path: Path, full_title: str,
                   panel_axes, panel_titles) -> None:

        panel_axes = list(panel_axes)
        panel_titles = list(panel_titles)
        if len(panel_axes) != len(panel_titles):
            raise ValueError("Each PowerPoint subplot must have exactly one title")
        fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
        self.images.append(FigureSlideImage(full_title, output_path.read_bytes(), "full figure"))
        for ax, title in zip(panel_axes, panel_titles):
            if not ax.get_visible() or not ax.axison:
                continue
            self.images.append(FigureSlideImage(
                str(title), self._panel_png(fig, ax), "individual subplot"
            ))

    @staticmethod
    def _add_title(slide, title: str, top: float = 0.25, size: int = 28) -> None:
        box = slide.shapes.add_textbox(Inches(0.55), Inches(top), Inches(12.25), Inches(0.62))
        frame = box.text_frame
        frame.clear()
        paragraph = frame.paragraphs[0]
        paragraph.text = title
        paragraph.font.name = "Arial"
        paragraph.font.size = Pt(size)
        paragraph.font.bold = True
        paragraph.font.color.rgb = RGBColor(22, 52, 78)
        paragraph.alignment = PP_ALIGN.LEFT

    @staticmethod
    def _add_image_contain(slide, png_bytes: bytes, left: float, top: float,
                           width: float, height: float) -> None:
        with Image.open(BytesIO(png_bytes)) as picture:
            pixel_width, pixel_height = picture.size
        scale = min(width / pixel_width, height / pixel_height)
        draw_width = pixel_width * scale
        draw_height = pixel_height * scale
        draw_left = left + (width - draw_width) / 2
        draw_top = top + (height - draw_height) / 2
        slide.shapes.add_picture(
            BytesIO(png_bytes), Inches(draw_left), Inches(draw_top),
            width=Inches(draw_width), height=Inches(draw_height),
        )

    def write(self, output_path: Path) -> None:

        if any(value is None for value in (Image, Presentation, RGBColor, PP_ALIGN, Inches, Pt)):
            raise ImportError("PowerPoint export requires Pillow and python-pptx")
        presentation = Presentation()
        presentation.slide_width = Inches(13.333)
        presentation.slide_height = Inches(7.5)
        blank_layout = presentation.slide_layouts[6]

        cover = presentation.slides.add_slide(blank_layout)
        cover.background.fill.solid()
        cover.background.fill.fore_color.rgb = RGBColor(244, 248, 251)
        accent = cover.shapes.add_shape(1, Inches(0), Inches(0), Inches(0.25), Inches(7.5))
        accent.fill.solid(); accent.fill.fore_color.rgb = RGBColor(21, 101, 128)
        accent.line.fill.background()
        self._add_title(cover, "WB-PIML — Figures", top=1.75, size=34)
        subtitle = cover.shapes.add_textbox(Inches(0.62), Inches(2.65), Inches(11.8), Inches(1.6))
        frame = subtitle.text_frame
        frame.clear()
        paragraph = frame.paragraphs[0]
        paragraph.text = (
            f"8 composite figures and every non-empty subplot\n"
            f"Generated from {self.data_name}"
        )
        paragraph.font.name = "Arial"; paragraph.font.size = Pt(21)
        paragraph.font.color.rgb = RGBColor(58, 77, 91)

        total = len(self.images) + 1
        for slide_number, item in enumerate(self.images, 2):
            slide = presentation.slides.add_slide(blank_layout)
            slide.background.fill.solid()
            slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
            self._add_title(slide, item.title)
            self._add_image_contain(slide, item.png_bytes, 0.45, 1.0, 12.43, 5.95)
            footer = slide.shapes.add_textbox(Inches(0.55), Inches(7.02), Inches(12.2), Inches(0.25))
            footer_frame = footer.text_frame
            footer_frame.clear()
            footer_p = footer_frame.paragraphs[0]
            footer_p.text = f"{item.image_kind}  |  {self.data_name}  |  {slide_number}/{total}"
            footer_p.font.name = "Arial"; footer_p.font.size = Pt(9)
            footer_p.font.color.rgb = RGBColor(102, 117, 127)
            footer_p.alignment = PP_ALIGN.RIGHT

        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}_", suffix=".pptx", dir=output_path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            presentation.save(temporary_path)
            with zipfile.ZipFile(temporary_path, "r") as archive:
                damaged_member = archive.testzip()
                if damaged_member is not None:
                    raise IOError(f"PowerPoint ZIP validation failed at {damaged_member}")
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def make_figures(out: Path, predictions: pd.DataFrame, metrics: pd.DataFrame,
                 effects: pd.DataFrame, uq: pd.DataFrame, runout: pd.DataFrame,
                 source_gains: pd.DataFrame, deck: FigureDeckCollector) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    exact = predictions.loc[predictions["is_exact"]]
    lim = [float(exact["logN"].min() - 0.3), float(exact["logN"].max() + 0.3)]

    chosen = ["WB", "ML-Ens", "ExtraTrees", "ML-Strong", "WB-PIML"]
    fig = plt.figure(figsize=(15, 9.5))
    grid = fig.add_gridspec(2, 6)
    positions = [
        (0, slice(0, 2)), (0, slice(2, 4)), (0, slice(4, 6)),
        (1, slice(1, 3)), (1, slice(3, 5)),
    ]
    axes = []
    for row, columns in positions:
        if axes:
            axes.append(fig.add_subplot(
                grid[row, columns], sharex=axes[0], sharey=axes[0]
            ))
        else:
            axes.append(fig.add_subplot(grid[row, columns]))
    for ax, model in zip(axes, chosen):
        part = exact.loc[exact["model"].eq(model)]
        ax.fill_between(lim, np.array(lim) - LOG5, np.array(lim) + LOG5, color="#f6c979", alpha=0.18)
        ax.plot(lim, lim, "k-", lw=1.4)
        ax.scatter(part["logN"], part["pred_logN"], s=28, alpha=0.72, edgecolor="white", linewidth=0.3)
        ax.set(xlim=lim, ylim=lim, title=f"{model}\nSB-RMSE={_metric_lookup(metrics, model, 'SB_RMSE'):.3f}")
    fig.supxlabel(r"Observed exact-fracture $\log_{10}(N_f)$")
    fig.supylabel(r"Campaign-disjoint prediction $\log_{10}(N_f)$")
    fig.suptitle("Campaign-disjoint outer-test predictions for exact-fracture records", y=0.99)
    fig.tight_layout()
    deck.add_figure(
        fig, out / "01_exact_failure_predictions.png",
        "Figure 1 — Exact-fracture outer-test predictions",
        axes,
        [f"Figure 1{chr(65 + index)} — {model} exact-fracture predictions"
         for index, model in enumerate(chosen)],
    )
    plt.close(fig)

    keys = [("SB_RMSE", "RMSE ↓"), ("SB_R2", r"$R^2$ ↑"), ("SB_F5", r"$F_5$ ↑"), ("SB_C_star", r"$C^*$ ↓")]
    chosen = ["WB", "ML-Ens", "ExtraTrees", "ET-Walker", "ML-Strong", "WB-PIML"]
    selected = metrics.loc[metrics["model"].isin(chosen)].set_index("model").loc[chosen].reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    colors = ["#777777", "#4c78a8", "#72b7b2", "#54a24b", "#b279a2", "#f58518"]
    for ax, (key, title) in zip(axes, keys):
        y = selected[key].to_numpy(float)
        lo, hi = selected[f"{key}_lo"].to_numpy(float), selected[f"{key}_hi"].to_numpy(float)
        ax.bar(selected["model"], y, color=colors, edgecolor="white")
        ax.errorbar(np.arange(len(y)), y, yerr=np.vstack([y - lo, hi - y]), fmt="none", ecolor="black", capsize=3)
        ax.set_title(title); ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Campaign-disjoint point-prediction metrics for exact fractures (95% campaign-cluster intervals)")
    fig.tight_layout()
    deck.add_figure(
        fig, out / "02_campaign_balanced_metrics.png",
        "Figure 2 — Campaign-balanced primary metrics",
        axes,
        [f"Figure 2{chr(65 + index)} — {title.replace(' $', ' ')}"
         for index, (_, title) in enumerate(keys)],
    )
    plt.close(fig)

    ablation_models = ["WB", "WB-CD", "WB-M", "WB-Ox", "ExtraTrees", "ET-Walker",
                       "WB-PIML", "WB-PIML-M", "WB-PIML-Ox"]
    physical = metrics.loc[metrics["model"].isin(ablation_models)].set_index("model").loc[ablation_models].reset_index()
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    y = physical["SB_RMSE"].to_numpy(float)
    ax.bar(physical["model"], y, color="#72b7b2", edgecolor="black", alpha=0.85)
    ax.errorbar(np.arange(len(y)), y,
                yerr=np.vstack([y - physical["SB_RMSE_lo"], physical["SB_RMSE_hi"] - y]),
                fmt="none", ecolor="black", capsize=4)
    ax.set_ylabel("Campaign-balanced RMSE ↓")
    ax.set_title("Retrained ablation of the physical trunk and residual correction")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    deck.add_figure(
        fig, out / "03_retrained_ablation.png",
        "Figure 3 — Retrained model ablation",
        [ax], ["Figure 3A — Competing-damage and residual ablation"],
    )
    plt.close(fig)

    paired_comparators = ["ExtraTrees", "ML-Strong", "ML-Ens"]
    paired_colors = {"ExtraTrees": "#72b7b2", "ET-Walker": "#54a24b", "ML-Strong": "#b279a2", "ML-Ens": "#4c78a8"}
    effect_labels = {
        "RMSE_gain": "RMSE gain",
        "MAE_gain": "MAE gain",
        "R2_gain": r"$R^2$ gain",
        "F5_gain": r"$F_5$ gain",
        "C_star_gain": r"$C^*$ gain",
    }
    paired_rows = effects.loc[
        effects["comparator"].isin(paired_comparators)
        & effects["effect"].isin(effect_labels)
    ]
    paired_low = float(min(0.0, paired_rows["CI95_lo"].min()))
    paired_high = float(max(0.0, paired_rows["CI95_hi"].max()))
    paired_pad = 0.05 * max(paired_high - paired_low, 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), sharey=True)
    for ax, comparator in zip(axes, paired_comparators):
        effect_plot = effects.loc[effects["comparator"].eq(comparator)].iloc[::-1].reset_index(drop=True)
        x = effect_plot["estimate"].to_numpy(float)
        ax.errorbar(
            x, np.arange(len(effect_plot)),
            xerr=np.vstack([x - effect_plot["CI95_lo"], effect_plot["CI95_hi"] - x]),
            fmt="o", color=paired_colors[comparator], ecolor=paired_colors[comparator], capsize=4,
        )
        ax.axvline(0, color="black", lw=1.2)
        ax.set_yticks(
            np.arange(len(effect_plot)),
            [effect_labels.get(value, value) for value in effect_plot["effect"]],
        )
        ax.set_xlim(paired_low - paired_pad, paired_high + paired_pad)
        ax.set_xlabel("Paired gain (positive favors WB-PIML)")
        ax.set_title(f"WB-PIML vs {comparator}")
    fig.suptitle("Paired comparison with predictive controls (95% campaign-cluster bootstrap intervals)", y=1.01)
    fig.tight_layout()
    deck.add_figure(
        fig, out / "04_paired_effect_intervals.png",
        "Figure 4 — Paired performance-effect intervals",
        axes,
        [f"Figure 4{chr(65 + index)} — WB-PIML versus {model}"
         for index, model in enumerate(paired_comparators)],
    )
    plt.close(fig)

    proposed = exact.loc[exact["model"].eq("WB-PIML")].sort_values("logN")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    axes[0].errorbar(proposed["logN"], proposed["pred_logN"],
                     yerr=np.vstack([proposed["pred_logN"] - proposed["lower90"], proposed["upper90"] - proposed["pred_logN"]]),
                     fmt="o", ms=3.5, alpha=0.45, ecolor="#90caf9", color="#4c78a8")
    interval_ylim = [
        float(min(lim[0], proposed["lower90"].min()) - 0.2),
        float(max(lim[1], proposed["upper90"].max()) + 0.2),
    ]
    axes[0].plot(lim, lim, "k-", lw=1.2); axes[0].set(xlim=lim, ylim=interval_ylim,
        xlabel=r"Observed exact-fracture $\log_{10}(N_f)$",
        ylabel=r"WB-PIML prediction $\log_{10}(N_f)$",
        title="Outer-test predictions with empirical 90% intervals")
    axes[1].errorbar(uq["nominal"], uq["coverage"],
                     yerr=np.vstack([uq["coverage"] - uq["coverage_lo"], uq["coverage_hi"] - uq["coverage"]]),
                     fmt="o-", capsize=4, color="#f58518", label="WB-PIML")
    axes[1].plot([0.75, 0.95], [0.75, 0.95], "k--", label="ideal")
    axes[1].set(xlim=(0.76, 0.94), ylim=(0.60, 1.01), xlabel="Nominal coverage", ylabel="Campaign-balanced empirical coverage",
                title="Empirical coverage (95% campaign-cluster confidence intervals)"); axes[1].legend()
    fig.tight_layout()
    deck.add_figure(
        fig, out / "05_empirical_prediction_intervals.png",
        "Figure 5 — Empirical prediction intervals",
        axes,
        ["Figure 5A — Outer-test empirical 90% intervals",
         "Figure 5B — Empirical interval coverage"],
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    nll_plot = runout.dropna(subset=["SB_runout_NLL"])
    axes[0].bar(nll_plot["model"], nll_plot["SB_runout_NLL"], color="#72b7b2")
    axes[0].set(title="Runout survival negative log-likelihood", ylabel="Campaign-balanced survival NLL ↓")
    axes[1].bar(runout["model"], runout["SB_point_below_runout_limit_rate"], color="#e45756")
    axes[1].set(title="Point predictions relative to runout limits", ylabel="Fraction below stopping cycles ↓")
    for ax in axes: ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    deck.add_figure(
        fig, out / "06_runout_evaluation.png",
        "Figure 6 — Runout evaluation under right censoring",
        axes,
        ["Figure 6A — Runout survival negative log-likelihood",
         "Figure 6B — Point prediction versus runout limit"],
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    gains = source_gains.sort_values("RMSE_gain").reset_index(drop=True)
    source_labels = [
        f"{row['plot_label']} (n={int(row['n_exact'])})"
        for _, row in gains.iterrows()
    ]
    ax.barh(source_labels, gains["RMSE_gain"],
            color=np.where(gains["RMSE_gain"] >= 0, "#54a24b", "#e45756"))
    ax.axvline(0, color="black", lw=1.2); ax.set_xlabel("Source RMSE gain: ML-Ens − WB-PIML")
    ax.set_title("Variation in RMSE gain across data sources")
    fig.tight_layout()
    deck.add_figure(
        fig, out / "07_source_level_gains.png",
        "Figure 7 — Source-specific RMSE gain",
        [ax], ["Figure 7A — Source-level RMSE gain"],
    )
    plt.close(fig)


def make_validation_figure(out: Path, validation_summary: pd.DataFrame,
                           external_summary: pd.DataFrame,
                           deck: FigureDeckCollector) -> None:

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    tier_order = ["record_interpolation", "series_disjoint", "source_disjoint", "campaign_disjoint"]
    tier_labels = ["Record\ninterpolation", "Series\ndisjoint", "Source\ndisjoint", "Campaign\ndisjoint"]
    fixed_models = ["WB", "ExtraTrees", "ET-Walker", "ML-Ens", "WB-PIML"]
    external_models = ["WB", "ExtraTrees", "ET-Walker", "ML-Ens", "ML-Strong", "WB-PIML"]
    colors = {"WB": "#777777", "WB-PIML": "#f58518", "ExtraTrees": "#72b7b2",
              "ET-Walker": "#54a24b", "ML-Ens": "#4c78a8", "ML-Strong": "#b279a2"}
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6), sharey=True)
    x = np.arange(len(tier_order))
    fixed_width = 0.15
    fixed_offsets = (np.arange(len(fixed_models)) - (len(fixed_models) - 1) / 2) * fixed_width
    for offset, model in zip(fixed_offsets, fixed_models):
        values, errors = [], []
        for tier in tier_order:
            selected = validation_summary.loc[
                validation_summary["strategy"].eq(tier) & validation_summary["model"].eq(model)
            ]
            if len(selected) != 1:
                raise AssertionError(f"Expected one fixed validation summary for {tier}/{model}")
            row = selected.iloc[0]
            values.append(float(row["SB_RMSE_mean"]))
            errors.append(float(row["SB_RMSE_sd"]) if np.isfinite(row["SB_RMSE_sd"]) else 0.0)
        axes[0].bar(x + offset, values, fixed_width, yerr=errors, capsize=3,
                    label=model, color=colors[model], edgecolor="white")
    axes[0].set_xticks(x, tier_labels)
    axes[0].set_ylabel(r"Campaign-balanced RMSE in $\log_{10}(N_f)$ ↓")
    axes[0].set_title("Fixed-configuration repeated validation\n(error bars: across-repeat SD)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper center",
                   bbox_to_anchor=(0.5, -0.20), ncol=3)

    scenario_order = ["LOSO", "LOAO", "LOTO", "LOEO"]
    x = np.arange(len(scenario_order))
    external_width = 0.13
    external_offsets = (np.arange(len(external_models)) - (len(external_models) - 1) / 2) * external_width
    for offset, model in zip(external_offsets, external_models):
        values = []
        for scenario in scenario_order:
            selected = external_summary.loc[
                external_summary["scenario"].eq(scenario) & external_summary["model"].eq(model),
                "SB_RMSE",
            ]
            if len(selected) != 1:
                raise AssertionError(f"Expected one external summary for {scenario}/{model}")
            values.append(float(selected.iloc[0]))
        axes[1].bar(x + offset, values, external_width, label=model,
                    color=colors[model], edgecolor="white")
    axes[1].set_xticks(x, scenario_order)
    axes[1].set_title("Factor-held-out stress tests\n(model selection within each training partition)")
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper center",
                   bbox_to_anchor=(0.5, -0.20), ncol=3)
    axes[1].text(
        0.01, -0.43,
        "LOSO: source; LOAO: coarse architecture; LOTO: temperature regime; LOEO: environment\n"
        "Campaign overlap is reported separately for each holdout",
        transform=axes[1].transAxes, fontsize=8.5, va="top",
    )
    fig.suptitle("Prediction errors under different validation schemes", y=1.02)
    fig.tight_layout()
    deck.add_figure(
        fig, out / "08_validation_scope_comparison.png",
        "Figure 8 — Validation-scope comparison", axes,
        ["Figure 8A — Fixed-configuration repeated validation",
         "Figure 8B — Factor-held-out tests including ML-Strong"],
    )
    plt.close(fig)


def write_workbook(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    if Font is None:
        warnings.warn("openpyxl unavailable; workbook was not written")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_", suffix=".xlsx", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with pd.ExcelWriter(temporary_path, engine="openpyxl") as writer:
            for sheet, table in tables.items():
                table.to_excel(writer, sheet_name=sheet[:31], index=False)
            workbook = writer.book
            for sheet in workbook.worksheets:
                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions
                sheet.sheet_view.showGridLines = False
                for cell in sheet[1]:
                    cell.font = Font(color="FFFFFF", bold=True)
                    cell.fill = PatternFill("solid", fgColor="1F4E78")
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                for column in range(1, sheet.max_column + 1):
                    values = [str(sheet.cell(row, column).value or "") for row in range(1, min(sheet.max_row, 250) + 1)]
                    sheet.column_dimensions[get_column_letter(column)].width = min(max(11, max(map(len, values)) + 2), 48)
                for row in sheet.iter_rows(min_row=2):
                    for cell in row:
                        cell.alignment = Alignment(vertical="top", wrap_text=False)
            text_layouts = {
                "README": {"A": 28, "B": 100},
                "Protocol": {"A": 38, "B": 110},
                "Data_Audit": {"A": 48, "B": 80},
                "Metric_Definitions": {"A": 22, "B": 34, "C": 100},
                "Model_Specs": {"A": 24, "B": 44, "D": 110},
                "Validation_Definitions": {"A": 24, "B": 100, "C": 100},
                "Claim_Gate": {"A": 46, "B": 12, "C": 110, "D": 20},
                "References": {"A": 44, "B": 70, "C": 100},
            }
            for sheet_name, widths in text_layouts.items():
                if sheet_name not in workbook.sheetnames:
                    continue
                sheet = workbook[sheet_name]
                for column, width in widths.items():
                    sheet.column_dimensions[column].width = width
                for row_index in range(2, sheet.max_row + 1):
                    sheet.row_dimensions[row_index].height = 32
                    for cell in sheet[row_index]:
                        cell.alignment = Alignment(vertical="top", wrap_text=True)
        with zipfile.ZipFile(temporary_path) as archive:
            damaged_member = archive.testzip()
            if damaged_member is not None:
                raise IOError(f"Workbook ZIP validation failed at {damaged_member}")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_readme(path: Path, data_path: Path, metrics: pd.DataFrame, effects: pd.DataFrame,
                 gate: pd.DataFrame, df: pd.DataFrame, parameters: pd.DataFrame,
                 n_runout: int) -> None:
    proposed = metrics.loc[metrics["model"].eq(PROPOSED_MODEL)].iloc[0]
    anchor = metrics.loc[metrics["model"].eq("WB-PIML-Anchor")].iloc[0]
    proposed_parameters = parameters.loc[parameters["model"].eq(PROPOSED_MODEL)]
    mix_text = ", ".join(f"{value:.2f}" for value in proposed_parameters["physics_mix"])
    comparison_models = ["WB", "WB-CD", "RF", "ExtraTrees", "ET-Walker", "GBDT", "SVR",
                         "ML-Ens", "ML-Strong", *CENSORED_MODELS, PROPOSED_MODEL]
    primary_best = metrics.loc[metrics["model"].isin(comparison_models)].sort_values("SB_RMSE").iloc[0]
    paired_lines = []
    for name in PRIMARY_COMPARATORS:
        selected = effects.loc[effects["comparator"].eq(name) & effects["effect"].eq("RMSE_gain")]
        if len(selected) != 1:
            raise AssertionError(f"README requires one paired RMSE result for {name}")
        result = selected.iloc[0]
        paired_lines.append(
            f"- Paired RMSE gain over {name}: **{result['estimate']:.3f}**, "
            f"95% campaign-bootstrap CI **[{result['CI95_lo']:.3f}, {result['CI95_hi']:.3f}]**."
        )
    n_exact = int(df["is_exact"].sum())
    text = f"""# WB-PIML results

## Analysis

- Data: `{data_path.name}` / `{SHEET}` only: {n_exact} exact fractures and {n_runout} right-censored runouts.
- All {len(df)} retained records have unit likelihood weight. Exact fractures contribute density terms; runouts contribute survival terms and never enter point-residual regression or point-life metrics as exact failures.
- Labels, evidence fields and campaign IDs are read from the workbook. The retained table has {df['record_equivalence_id'].nunique()} distinct equivalence IDs. ID uniqueness is a structural check and does not establish independence of every cross-publication experiment; unresolved source provenance remains qualified in the workbook.
- {df['source_id'].nunique()} sources belong to {df['campaign_id'].nunique()} reviewed campaigns. Primary validation uses five campaign-disjoint outer folds and campaign-disjoint development folds; no campaign crosses a primary train/test split.
- The six-parameter physical trunk (including the training-estimated water multiplier) combines Walker--Basquin mechanical damage and time-dependent environmental damage. It is fitted by joint censored AFT likelihood with mechanical-coefficient regularization and a nonnegative penalty on the training mean squared environmental log-life decrement. The environmental-penalty strength is selected within development folds.
- The hybrid candidates retain the physical prediction with unit weight and add an exact-fracture residual. The three residual families are ExtraTrees, ExtraTrees with training-fitted physical features, and Matérn kernel ridge. Matérn uses nu=1.5, length scale=2 and ridge alpha=1, with zero residual prior.
- All residual candidates use the original observable numeric inputs and coarse/detailed architecture, but exclude environment labels and environmental derived features. The mechanically augmented residual adds mechanical predicted life and Walker stress fitted on the corresponding development subset. Environment remains an input to the physical trunk and direct predictive controls; direct physics-feature controls retain environmental decrement. Sources, campaigns and record IDs are not predictive features.
- Every hybrid uses soft temperature support `exp[-(nearest exact-development temperature distance / 250 C)^2]`. Support depends on development covariates only. The same-temperature flag reports availability and does not imply zero residual at unseen temperatures.
- WB-PIML evaluates {len(HYBRID_CANDIDATES)} candidates: three residual families, gamma in {{0.75, 0.85, 0.95}}, environmental strength in {{0, 0.1, 1}}, and eta in {{0.75, 1}}. ML-Strong separately selects among {len(CONTROL_CANDIDATES)} candidates: 18 ExtraTrees, 18 gradient boosting, 18 Matérn, one fixed ML ensemble and nine direct ExtraTrees controls with the same training-fitted physical features. ML-Strong is a selected predictive control, and its full pool includes physics-informed candidates.
- Each pool independently retains candidates with inner campaign-balanced exact RMSE within {NEAR_TIE_RMSE:.3f} of its own minimum, then minimizes calibrated inner runout survival NLL; RMSE and candidate index resolve ties. All candidates in a pool are evaluated on the same development folds. Total search budgets are 54 and 64, not equal. ET-Walker retains its separate gamma/tree search.
- The WB-CD reference survival comparison is a diagnostic, not a selection filter. No affine output calibration is applied. LOTO is a separate factor-held-out stress test and does not choose the hybrid configuration.
- Repeated record-, series-, source- and campaign-disjoint validation uses a fixed secondary ExtraTrees residual: gamma={GAMMA}, environmental strength=0, eta=0.75, leaf=3, max_features=0.85, and soft support at 250 C. Duplicate/label and S-star sensitivity use the same fixed configuration; these results are distinct from the nested-selected primary model.
- LOSO/LOAO/LOTO/LOEO re-select WB-PIML, ML-Strong, ExtraTrees and ET-Walker using only the remaining records. Factor-held-out splits do not impose campaign separation; actual overlap is reported per holdout. LOAO retains coarse architecture groups while detailed architecture remains a predictive descriptor.
- The 80% and 90% prediction intervals use campaign-balanced empirical absolute errors from development OOF predictions. They are not formal finite-sample conformal or design-life guarantees.

## Results

- Lowest campaign-balanced RMSE among the listed physical baselines, predictive controls and proposed model: **{primary_best['model']} = {primary_best['SB_RMSE']:.3f}**.
- WB-PIML campaign-balanced RMSE: **{proposed['SB_RMSE']:.3f}**.
- Selected hybrid branch RMSE: **{anchor['SB_RMSE']:.3f}**; reported-model difference: **{anchor['SB_RMSE'] - proposed['SB_RMSE']:.3f}**.
- Physical-trunk weights by fold: **[{mix_text}]**; every value is 1.00.
{chr(10).join(paired_lines)}

`SB_*` denotes campaign-balanced metrics. `Claim_Gate` reports computed criteria, including comparisons with ML-Strong; a lower point estimate alone does not establish superiority or independent external validation.

## Files

- One results workbook: `WB_PIML_results.xlsx`, including selection traces, fold assignments, source audits and all validation summaries.
- Eight composite PNGs and `{PPTX_OUTPUT_NAME}`; the presentation includes each composite and every non-empty subplot.
- `protocol.json` and this README describe the executed method. CSV exports are optional through `--export-csv`.
"""
    path.write_text(text, encoding="utf-8")

