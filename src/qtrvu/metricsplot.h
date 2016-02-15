#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include "accelerad.h"

#define HISTORY 60 // seconds

#ifdef ACCELERAD_RT
class MetricsPlot : public QwtPlot
{
	//Q_OBJECT
public:
	enum Metrics
	{
		//AvLum,
		//LumBackg,
		Ev,
		//EvDir,
		DGP,
		//DGI,
		//UGR,
		//VCP,
		//CGI,
		//Lveil,
		RAMMG,

		MetricsCount
	};

	MetricsPlot();

	void addData(double ev, double dgp, double rammg);

protected:

private:
	struct
	{
		QwtPlotCurve *curve;
		double data[HISTORY];
	} data[MetricsCount];

	void addCurve(const QString title, const Qt::GlobalColor color, qreal width, Axis axis, Metrics metric);

	double timeData[HISTORY];

	int dataCount;
};
#endif
