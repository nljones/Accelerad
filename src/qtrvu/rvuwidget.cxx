#include "rvuwidget.h"

#include <QApplication>
#include <QCursor>
#include <QColor>
#include <QPainter>
#include <QPixmap>
#include <QMouseEvent>

#ifdef ACCELERAD_RT
#include "fvect.h"
#include "rpaint.h"
#endif

RvuWidget::RvuWidget(QWidget* parent) : QWidget(parent), m_x(0), m_y(0)
{
  m_image = new QPixmap(200, 200);
  m_image->fill(QColor(0, 0, 0));
  m_painter = new QPainter(m_image);
  m_do_pick = false;
}

RvuWidget::~RvuWidget()
{
  m_painter->end();
  delete m_painter;
  delete m_image;
}

void RvuWidget::resizeImage(int X, int Y)
{
  m_painter->end();
  delete m_image;
  m_image =  new QPixmap(X, Y);
  m_image->fill(QColor(0, 0, 0));
  m_painter->begin(m_image);
  m_painter->setPen(Qt::NoPen);
}

void RvuWidget::drawRect(int x, int y, int width, int height,
                         const QColor &color)
{
  m_painter->fillRect(x, y, width, height, color);
}

#ifdef ACCELERAD_RT
void RvuWidget::drawImage(int x, int y, int width, int height, const uchar *data)
{
	QTransform transf = m_painter->transform();
	transf.reset();
	transf.scale(1, -1);
	transf.translate(0, 1 - height);
	m_painter->setTransform(transf);

	QImage *image = new QImage(data, width, height, QImage::Format_ARGB32_Premultiplied);

	m_painter->drawImage(x, y, *image);
}

/* Move camera with scroll wheel */
void RvuWidget::wheelEvent(QWheelEvent *event)
{
	FVECT origin, direction;
	VIEW nv = ourview;
	const int x = event->x();
	const int y = this->height() - event->y() - 1;
	const int delta = event->delta() / 120;

	if (viewray(origin, direction, &nv, (x + .5) / hresolu, (y + .5) / vresolu) < -FTINY)
		return;

	VSUM(nv.vp, origin, direction, delta); //TODO adjust delta
	newview(&nv);
}

/* Pan, zoom, and swivel camera with right mouse button */
void RvuWidget::mouseMoveEvent(QMouseEvent *event)
{
	if (event->buttons() & Qt::RightButton)
	{
		const int x = event->x() - p_x;
		const int y = event->y() - p_y;
		p_x = event->x();
		p_y = event->y();

		if (event->modifiers() & Qt::ShiftModifier) { /* Pan */
			VIEW nv = ourview;
			FVECT u, v;
			VCROSS(u, nv.vdir, nv.vup);
			if (normalize(u) < FTINY) return;
			VCROSS(v, u, nv.vdir);
			if (normalize(v) < FTINY) return;
			VSUM(nv.vp, nv.vp, u, -10.0 * x / this->width());
			VSUM(nv.vp, nv.vp, v, 10.0 * y / this->height());
			newview(&nv);
		}
		else if (event->modifiers() & Qt::ControlModifier) { /* Zoom */
			VIEW nv = ourview;
			double mag = 1.0 - 2.0 * y / this->height();
			if (mag < -FTINY)		/* negative zoom is reduction */
				mag = -1.0 / mag;
			else if (mag <= FTINY)	/* too small */
				return;
			zoomview(&nv, mag);
			newview(&nv);
		}
		else { /* Swivel */
			FVECT vc;
			VSUM(vc, ourview.vp, ourview.vdir, 1); //TODO adjust distance
			moveview(-100.0 * x / this->width(), 100.0 * y / this->height(), 1, vc);
		}
	}
}
#endif /* ACCELERAD_RT */

void RvuWidget::getPosition(int *x, int *y)
{
  this->m_do_pick = true;
  while(this->m_do_pick)
    {
    QApplication::processEvents();
    }
  *x = m_x;
  *y = m_y;
}

void RvuWidget::paintEvent(QPaintEvent *)
{
  QPainter painter(this);
  // Draw QImage
  painter.drawPixmap(0, 0, *m_image);

  // Draw the crosshairs
  painter.setPen(QColor(0, 255, 0));
  painter.drawLine(m_x - 10, m_y, m_x + 10, m_y);
  painter.drawLine(m_x, m_y - 10, m_x, m_y + 10);
}

void RvuWidget::mousePressEvent(QMouseEvent *event)
{
  if (event->button() == Qt::LeftButton)
    {
    this->m_do_pick = false;
    // Set the cursor position
    m_x = event->x();
    m_y = event->y();
    }
#ifdef ACCELERAD_RT
  /* Pan and swivel */
  else if (event->button() == Qt::RightButton) {
	p_x = event->x();
	p_y = event->y();
  }
#endif /* ACCELERAD_RT */
  this->repaint();
}

void RvuWidget::setPosition(int x, int y)
{
  m_x = x;
  m_y = y;
  this->repaint();
}
