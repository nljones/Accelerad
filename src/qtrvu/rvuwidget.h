
#ifndef __RvuWidget_H
#define __RvuWidget_H

#include <QWidget>

#include "accelerad.h"

class QPixmap;
class QColor;
class QMouseEvent;
class QPainter;

class RvuWidget : public QWidget
{
  Q_OBJECT

public:
  RvuWidget(QWidget* parent = 0);
  ~RvuWidget();

  /** Draw a rectangle to the widget (stored in a QImage). */
  void drawRect(int x, int y, int width, int height, const QColor &color);

#ifdef ACCELERAD_RT
  /** Draw an image to the widget (stored in a QImage). */
  void drawImage(int x, int y, int width, int height, const uchar *data);

  /** Receive mouse wheel events, move camera. */
  void RvuWidget::wheelEvent(QWheelEvent *event);

  /** Receive mouse wheel events, pan, zoom, and swivel camera */
  void RvuWidget::mouseMoveEvent(QMouseEvent *event);
#endif

  /** Resize the stored QImage to the supplied width and height. */
  void resizeImage(int width, int height);

  void getPosition(int *x, int *y);

  void setPosition(int x, int y);
protected:
  /** Simple draws the QImage and the crosshairs. */
  void paintEvent(QPaintEvent *event);

  /** Receive mouse events, move the crosshairs on left clicks. */
  void mousePressEvent(QMouseEvent *event);

  /** The QImage that the ray tracing code paints to. */
  QPixmap *m_image;
  QPainter *m_painter;

  /** X and Y position of the crosshairs. */
  int m_x;
  int m_y;
  bool m_do_pick;

#ifdef ACCELERAD_RT
  /** Last X and Y for camera movement */
  int p_x;
  int p_y;
#endif
};

#endif
