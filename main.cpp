#include <QApplication>
#include <QtCore/qlibraryinfo.h>

#include "mainwindow.h"

int main(int argc, char *argv[]) {
    // qDebug() << QLibraryInfo::location(QLibraryInfo::PluginsPath);

    QApplication app(argc, argv);
    MainWindow window;
    window.setWindowTitle("ImageViewer");
    window.show();
    return app.exec();
}
