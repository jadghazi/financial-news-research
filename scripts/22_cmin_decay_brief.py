"""Create a two-page meeting brief and standalone scientific PNGs; no fitting."""
import json
import math
from pathlib import Path
import subprocess
from html import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.graphics.shapes import Drawing, String, Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderPDF

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/cmin_decay_48h_2026-10-14_v1'
POPPLER = Path('C:/Users/user/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe')
BLUE, ORANGE, NAVY = colors.HexColor('#176b94'), colors.HexColor('#bd601c'), colors.HexColor('#183349')


def scientific_curve(summary):
    drawing = Drawing(515, 200)
    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = 64, 36, 425, 123
    plot.data = [[(v['age_sessions_since_proxy_arrival'] + offset, v['delta_log_loss'])
                  for v in summary['age_sensitivity'] if v['arm'] == arm]
                 for arm, offset in [('pooled_uniform', -.06), ('pooled_decay', .06)]]
    extrema = [0.] + [x for v in summary['age_sensitivity'] for x in v['conditional_95_interval']]
    spread = max(max(extrema) - min(extrema), .0002)
    lo, hi = min(extrema) - .18 * spread, max(extrema) + .18 * spread
    plot.xValueAxis.valueMin, plot.xValueAxis.valueMax, plot.xValueAxis.valueStep = -.35, 4.35, 1
    plot.xValueAxis.valueSteps = [0, 1, 2, 3, 4]
    plot.yValueAxis.valueMin, plot.yValueAxis.valueMax = lo, hi
    plot.yValueAxis.valueSteps = [lo + (hi - lo) * i / 4 for i in range(5)]
    plot.yValueAxis.labelTextFormat = lambda v: f'{v:.4f}'
    plot.xValueAxis.labels.fontSize = plot.yValueAxis.labels.fontSize = 9
    plot.lines[0].strokeColor, plot.lines[1].strokeColor = BLUE, ORANGE
    plot.lines[0].strokeWidth = plot.lines[1].strokeWidth = 1.5
    drawing.add(plot)
    def xy(age, value):
        return (plot.x + (age + .35) / 4.7 * plot.width,
                plot.y + (value - lo) / (hi - lo) * plot.height)
    y0 = xy(0, 0)[1]
    drawing.add(Line(plot.x, y0, plot.x + plot.width, y0, strokeColor=colors.HexColor('#999999'), strokeDashArray=[3, 3]))
    for arm, offset, color in [('pooled_uniform', -.06, BLUE), ('pooled_decay', .06, ORANGE)]:
        for value in summary['age_sensitivity']:
            if value['arm'] != arm:
                continue
            age = value['age_sessions_since_proxy_arrival'] + offset
            x, lower = xy(age, value['conditional_95_interval'][0])
            upper = xy(age, value['conditional_95_interval'][1])[1]
            center = xy(age, value['delta_log_loss'])[1]
            drawing.add(Line(x, lower, x, upper, strokeColor=color, strokeWidth=.8))
            drawing.add(Line(x - 3, lower, x + 3, lower, strokeColor=color, strokeWidth=.8))
            drawing.add(Line(x - 3, upper, x + 3, upper, strokeColor=color, strokeWidth=.8))
            drawing.add(Line(x - 2.5, center, x + 2.5, center, strokeColor=color, strokeWidth=2.5))
    drawing.add(String(64, 180, 'Change in raw model log loss after news-bucket removal', fontName='Helvetica-Bold', fontSize=10, fillColor=NAVY))
    drawing.add(String(167, 12, 'Age since proxy arrival (trading sessions)', fontSize=9))
    drawing.add(String(310, 164, 'Equal weighting', fontSize=9, fillColor=BLUE))
    drawing.add(String(407, 164, '48-hour decay', fontSize=9, fillColor=ORANGE))
    return drawing


def prescribed_curve():
    drawing = Drawing(515, 100)
    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = 64, 28, 425, 50
    plot.data = [[(hour, 2 ** (-hour / 48)) for hour in range(0, 169, 3)]]
    plot.xValueAxis.valueMin, plot.xValueAxis.valueMax, plot.xValueAxis.valueSteps = 0, 168, [0, 48, 96, 144]
    plot.yValueAxis.valueMin, plot.yValueAxis.valueMax, plot.yValueAxis.valueSteps = 0, 1, [0, .5, 1]
    plot.yValueAxis.labelTextFormat = lambda v: f'{v:.1f}'
    plot.lines[0].strokeColor, plot.lines[0].strokeWidth = ORANGE, 1.7
    plot.xValueAxis.labels.fontSize = plot.yValueAxis.labels.fontSize = 9
    drawing.add(plot)
    drawing.add(String(64, 87, 'Imposed relative recency multiplier - not an empirical finding', fontName='Helvetica-Bold', fontSize=10, fillColor=NAVY))
    drawing.add(String(176, 5, 'Additional age (calendar hours)', fontSize=9))
    return drawing


def main():
    summary = json.loads((STUDY / 'analysis/summary.json').read_text())
    run = json.loads((STUDY / 'summary.json').read_text())
    target = ROOT / 'output/pdf/CMIN_news_decay_meeting_brief_2026-10-14.pdf'
    if target.exists():
        raise ValueError('Existing brief preserved; refusing overwrite')
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / 'tmp/pdfs/cmin_decay_2026-10-14'
    tmp.mkdir(parents=True, exist_ok=True)
    figures = ROOT / 'output/figures'
    figures.mkdir(parents=True, exist_ok=True)
    drawings = {'cmin_decay_age_sensitivity_2026-10-14': scientific_curve(summary),
                'cmin_decay_prescribed_weights_2026-10-14': prescribed_curve()}
    for name, drawing in drawings.items():
        temporary = tmp / (name + '.pdf')
        renderPDF.drawToFile(drawing, str(temporary))
        subprocess.run([str(POPPLER), '-png', '-r', '180', '-singlefile', str(temporary), str(figures / name)], check=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('BriefTitle', fontName='Helvetica-Bold', fontSize=20, leading=24, textColor=NAVY, spaceAfter=9))
    styles.add(ParagraphStyle('BriefBody', fontName='Helvetica', fontSize=9.5, leading=13, spaceAfter=8))
    styles.add(ParagraphStyle('BriefSmall', fontName='Helvetica', fontSize=8.2, leading=10.5, textColor=colors.HexColor('#475462'), spaceAfter=6))
    styles.add(ParagraphStyle('BriefHead', fontName='Helvetica-Bold', fontSize=11, leading=15, textColor=NAVY, spaceBefore=9, spaceAfter=5))
    def p(text, style='BriefBody'):
        return Paragraph(text, styles[style])
    def table(data, widths):
        result = Table(data, colWidths=widths, hAlign='LEFT')
        result.setStyle(TableStyle([('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf2f5')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'), ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, -1), NAVY), ('LEADING', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3), ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('LINEBELOW', (0, 0), (-1, 0), .5, colors.HexColor('#c0cbd3'))]))
        return result
    primary = summary['primary']
    delta, ci = primary['delta_log_loss'], primary['conditional_95_interval']
    practical = delta <= -.005
    all_negative = all(v['delta_log_loss'] < 0 for v in summary['period_seed_differences'])
    supported = practical and all_negative and ci[1] < 0
    verdict = ('Decay showed a consistent practical improvement within this exploratory study.' if supported else
               'No stable, practically meaningful forecasting improvement from 48-hour decay was established.')
    story = [p('Does explicit news decay improve forecasts?', 'BriefTitle'),
        p('FININ-inspired public CMIN study | 14 October 2026 | Exploratory temporal validation', 'BriefSmall'),
        p('<b>' + verdict + '</b>'),
        p('FININ links news interactions and market-conditioned weighting to information diffusion, but does not impose an explicit age-decay law. We tested a fixed recency prior as an independent extension, keeping the next-session prediction target.'),
        p('Matched experiment', 'BriefHead'),
        p('Same 12-company cohort, five price sessions, eligible headlines, frozen RoBERTa and sentiment, both attention stages, and 96-value final MLP input. Equal temporal weighting gives equal mass to nonempty news buckets; decay multiplies within-bucket weights by 2<super>-age/48</super> and normalizes across retained stories. Eight fits: two arms, two seeds, two chronological periods; no half-life search.'),
        p('Forecast results (lower log loss is better)', 'BriefHead')]
    data = [['Model / reference', 'Log loss', 'Accuracy']]
    for arm, name in [('pooled_uniform', 'Equal temporal weighting'), ('pooled_decay', '48-hour decay'), ('training_prior', 'Training-prevalence probabilities')]:
        value = summary['metrics'][arm]
        data.append([name, f"{value['mean_seed_log_loss']:.6f}", f"{value['mean_seed_accuracy']:.2%}"])
    story += [table(data, [285, 110, 110]), Spacer(1, 8),
        p('Decay did not beat the training-prevalence probability reference on log loss. Both decay models in Jan-Apr 2021 predicted UP on all 984 cases; their probabilities still varied. Validation selected neural forecasts in all eight fits, with no constant fallback.', 'BriefSmall'),
        p(f'<b>Decay minus equal-weighting log loss: {delta:+.6f}</b>; conditional 95% interval [{ci[0]:+.6f}, {ci[1]:+.6f}]. The predeclared practical improvement criterion was -0.005.'),
        p('Repeatability across period and seed', 'BriefHead')]
    fit_map = {(v['fold'], v['arm'], v['seed']): v for v in summary['fits']}
    data = [['Evaluation period', 'Seed', 'Decay - equal loss']]
    for value in summary['period_seed_differences']:
        period = 'May-Aug 2020' if value['fold'] == 0 else 'Jan-Apr 2021'
        data.append([period, str(value['seed']), f"{value['delta_log_loss']:+.6f}"])
    story += [table(data, [285, 110, 110]), Spacer(1, 7),
        p(f"{summary['evaluation_company_days']:,} matched company-days on {summary['evaluation_unique_dates']} dates. Losses average raw seed-model losses, not ensemble probabilities. Date-block intervals keep stocks together and condition on saved fits; they omit training uncertainty.", 'BriefSmall'),
        p('Limits on the conclusion', 'BriefHead'),
        p('Existing evaluation periods were already inspected, so these are exploratory results. Source-date timing and D+2 availability are proxies. Normalized decay changes relative allocation; it does not make all old news fade absolutely. Improvement could reflect regularization or noise filtering rather than market diffusion. The existing price-only model has different geometry and is a contextual reference.'),
        PageBreak(), p('Measured news-age sensitivity', 'BriefTitle'),
        p('For each selected raw checkpoint, remove one arrival-age bucket, retain every price feature, and recompute predictions without retraining.', 'BriefBody'),
        drawings['cmin_decay_age_sensitivity_2026-10-14'],
        p('Positive values: removing that bucket worsened forecast loss. Negative values: removal improved loss. Bars are pointwise conditional 95% date-block intervals. Removal also renormalizes other news and changes coverage; these are fixed-model sensitivities, not independent age contributions or a market diffusion curve. Effects are tiny and non-monotone; nine of ten pointwise intervals include zero. The isolated equal-weighting age-3 signal does not establish a decay law.', 'BriefSmall')]
    coverage = [v for v in summary['age_sensitivity'] if v['arm'] == 'pooled_uniform']
    data = [['Arrival age (sessions)', 'Cases with removed news', 'Headlines removed']]
    for value in coverage:
        data.append([str(value['age_sessions_since_proxy_arrival']),
                     f"{value['cases_with_removed_headlines']:,} / {summary['evaluation_company_days']:,}",
                     f"{value['headlines_removed']:,}"])
    story += [table(data, [180, 185, 140]), Spacer(1, 7),
        drawings['cmin_decay_prescribed_weights_2026-10-14'],
        p('The 48-hour curve was chosen before fitting. Calendar-hour decay and trading-session diagnostic ages use different clocks. No economic half-life was estimated, and this curve was not fitted to outcomes.', 'BriefSmall'),
        p('Connection to FININ and evidence', 'BriefHead'),
        p('<link href="https://aclanthology.org/2024.findings-emnlp.189.pdf" color="#176b94">Wang, Cohen and Ma (2024), FININ</link>: sections 4.2-4.3 (Eqs. 6-7), section 6.4 (diffusion interpretation). Our study tests an explicit relative-recency extension; it neither reproduces nor refutes FININ\'s economic claims.', 'BriefSmall'),
        p('Frozen protocol, configurations, all eight checkpoints, dated predictions, age-mask logits, coverage and analysis hashes: artifacts/cmin_decay_48h_2026-10-14_v1/. Negative outcomes are retained; no follow-up tuning was performed.', 'BriefSmall')]
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#d2dbe0'))
        canvas.line(40, 32, A4[0] - 40, 32)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#62727e'))
        canvas.drawString(40, 19, 'FININ research | Relative recency experiment | Conditional news timing')
        canvas.drawRightString(A4[0] - 40, 19, str(doc.page))
        canvas.restoreState()
    SimpleDocTemplate(str(target), pagesize=A4, rightMargin=40, leftMargin=40,
        topMargin=36, bottomMargin=43, title='CMIN news decay: exploratory FININ extension',
        author='FININ research').build(story, onFirstPage=footer, onLaterPages=footer)
    subprocess.run([str(POPPLER), '-png', '-r', '130', str(target), str(tmp / 'brief')], check=True)
    print(json.dumps({'pdf': str(target), 'preview_directory': str(tmp), 'verdict': verdict}))


if __name__ == '__main__':
    main()
