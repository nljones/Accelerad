#include "accelerad.h"

#define HISTORY 50 // frames

#ifdef ACCELERAD_RT
#include <qwt_plot.h>
#include <qwt_plot_curve.h>

class MetricsPlot : public QwtPlot
{
	//Q_OBJECT
public:
	enum Metrics
	{
		AvLum,
		//LumBackg,
		Ev,
		//EvDir,
		DGP,
		//DGI,
		//UGR,
		//VCP,
		//CGI,
		//Lveil,
		TaskLum,
		CR,
		RAMMG,

		MetricsCount
	};

	MetricsPlot();

	void addData(double *values);

protected:

private:
	struct
	{
		QwtPlotCurve *curve;
		Axis axis;
		double data[HISTORY];
	} data[MetricsCount];

	void addCurve(const QString title, const Qt::GlobalColor color, qreal width, Axis axis, Metrics metric);

	void addPoint(Metrics metric, double value);

	double timeData[HISTORY];

	int dataCount;
};
#endif
