#include <qwt_plot.h>
#include <qwt_plot_grid.h>
#include <qwt_symbol.h>
#include <qwt_legend.h>
#include <qwt_scale_draw.h>

#include "metricsplot.h"
#include "ray.h"

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
	setAxisTitle(QwtPlot::yRight, do_irrad ? "Illuminance [lux]" : "Luminance [cd/m2]");
	setAxisScale(QwtPlot::yRight, 0.0, 1.0);
	insertLegend(new QwtLegend());

	QwtPlotGrid *grid = new QwtPlotGrid();
	grid->setPen(QColor::fromRgb(192, 192, 192), 1.0);
	grid->attach(this);

	addCurve("Mean", Qt::darkYellow, 2, QwtPlot::yRight, AvLum);
	addCurve("Ev", Qt::green, 2, QwtPlot::yRight, Ev);
	addCurve("DGP", Qt::blue, 4, QwtPlot::yLeft, DGP);
	addCurve("Task", Qt::red, 2, QwtPlot::yRight, TaskLum);
	addCurve("CR", Qt::cyan, 4, QwtPlot::yLeft, CR);
	addCurve("RAMMG", Qt::magenta, 2, QwtPlot::yRight, RAMMG);

	data[Ev].curve->hide();

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
	data[metric].axis = axis;
}

/* Add a point to a curve and choose the upper bounds for its axis. */
void MetricsPlot::addPoint(MetricsPlot::Metrics metric, double value)
{
	data[metric].data[0] = value;
	if (data[metric].curve->isVisible() && axisInterval(data[metric].axis).maxValue() < value) {
		double upper = pow(10.0, ceil(log10(value)));
		if (upper / 5 > value)
			upper /= 5;
		else if (upper / 2 > value)
			upper /= 2;
		setAxisScale(data[metric].axis, 0.0, upper);
	}
}

void MetricsPlot::addData(double *values)
{
	for (int i = dataCount; i > 0; i--)
		if (i < HISTORY)
			for (int c = 0; c < MetricsCount; c++)
				data[c].data[i] = data[c].data[i - 1];

	addPoint(AvLum, values[AvLum]);
	addPoint(Ev, values[Ev]);
	addPoint(DGP, values[DGP]);
	addPoint(TaskLum, values[TaskLum] > 0.0 ? values[TaskLum] : -1.0);
	addPoint(CR, values[CR] > 0.0 ? 1.0 / values[CR] : -1.0f);
	addPoint(RAMMG, values[RAMMG]);

	if (dataCount < HISTORY)
		dataCount++;

	for (int j = 0; j < HISTORY; j++)
		timeData[j]++;

	setAxisScale(QwtPlot::xBottom, std::max(0.0, timeData[HISTORY - 1]), std::max(HISTORY * 1.0, timeData[0]));

	for (int c = 0; c < MetricsCount; c++)
		data[c].curve->setRawSamples(timeData, data[c].data, dataCount);

	replot();
}
#endif /* ACCELERAD_RT */
