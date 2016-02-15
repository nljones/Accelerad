#include <qwt_plot.h>
#include <qwt_plot_grid.h>
#include <qwt_symbol.h>
#include <qwt_legend.h>
#include <qwt_scale_draw.h>

#include "metricsplot.h"

#ifdef ACCELERAD_RT
MetricsPlot::MetricsPlot() : QwtPlot()
{
	dataCount = 0;

	setTitle("Metrics");
	setCanvasBackground(Qt::white);

	setAxisTitle(QwtPlot::xBottom, "Frame");
	setAxisTitle(QwtPlot::yLeft, "Daylight Glare Probability");
	setAxisScale(QwtPlot::yLeft, 0.0, 1.0);
	enableAxis(QwtPlot::yRight);
	setAxisTitle(QwtPlot::yRight, "Illuminance [lux]");
	setAxisScale(QwtPlot::yRight, 0.0, 1.0);
	insertLegend(new QwtLegend());

	QwtPlotGrid *grid = new QwtPlotGrid();
	grid->attach(this);

	addCurve("Ev", Qt::green, 2, QwtPlot::yRight, Ev);
	addCurve("DGP", Qt::blue, 4, QwtPlot::yLeft, DGP);
	addCurve("RAMMG", Qt::darkYellow, 4, QwtPlot::yRight, RAMMG);

	for (int i = 0; i < HISTORY; i++)
		timeData[i] = -1 - i;

	resize(600, 400);
	show();
}

void MetricsPlot::addCurve(const QString title, const Qt::GlobalColor color, qreal width, QwtPlot::Axis axis, MetricsPlot::Metrics metric)
{
	QwtPlotCurve *curve = new QwtPlotCurve(title);
	curve->setRenderHint(QwtPlotItem::RenderAntialiased, true);
	curve->setPen(color, width);
	curve->setYAxis(axis);
	curve->attach(this);
	data[metric].curve = curve;
}

void MetricsPlot::addData(double ev, double dgp, double rammg)
{
	for (int i = dataCount; i > 0; i--)
	{
		for (int c = 0; c < MetricsCount; c++)
		{
			if (i < HISTORY)
				data[c].data[i] = data[c].data[i - 1];
		}
	}

	data[Ev].data[0] = ev;
	if (axisInterval(QwtPlot::yRight).maxValue() < ev)
		setAxisScale(QwtPlot::yRight, 0.0, pow(10.0, ceil(log10(ev))));
	data[DGP].data[0] = dgp;
	if (axisInterval(QwtPlot::yLeft).maxValue() < dgp)
		setAxisScale(QwtPlot::yLeft, 0.0, pow(10.0, ceil(log10(dgp))));
	data[RAMMG].data[0] = rammg;
	if (axisInterval(QwtPlot::yRight).maxValue() < rammg)
		setAxisScale(QwtPlot::yRight, 0.0, pow(10.0, ceil(log10(rammg))));

	if (dataCount < HISTORY)
		dataCount++;

	for (int j = 0; j < HISTORY; j++)
		timeData[j]++;

	setAxisScale(QwtPlot::xBottom, std::max(0.0, timeData[HISTORY - 1]), std::max(1.0, timeData[0]), dataCount > 15 ? 0.0 : 1.0);

	for (int c = 0; c < MetricsCount; c++)
		data[c].curve->setRawSamples(timeData, data[c].data, dataCount);

	replot();
}
#endif /* ACCELERAD_RT */
