#include "mainwindow.h"

#include <QApplication>
#include <QDebug>
#include <QFileDialog>
#include <QKeyEvent>
#include <QMessageBox>
#include <QPixmap>
#include <QRegularExpression>

#include "controller/PolygonController.hpp"
#include "controller/RectController.hpp"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), fileMenu(nullptr), viewMenu(nullptr),
                                          currentImage(nullptr) {
    initUI();
}

MainWindow::~MainWindow() {}

void MainWindow::initUI() {
    this->resize(800, 600);
    // setup menubar
    fileMenu = menuBar()->addMenu("&File");
    viewMenu = menuBar()->addMenu("&View");

    // setup toolbar
    fileToolBar = addToolBar("File");
    viewToolBar = addToolBar("View");

    // main area for image display
    imageScene = new QGraphicsScene(this);
    imageView = new QGraphicsView(imageScene);
    setCentralWidget(imageView);

    // setup status bar
    mainStatusBar = statusBar();
    mainStatusLabel = new QLabel(mainStatusBar);
    mainStatusBar->addPermanentWidget(mainStatusLabel);
    mainStatusLabel->setText("Image Information will be here!");

    createActions();
}

void MainWindow::createActions() {
    // create actions, add them to menus
    openAction = new QAction("&Open", this);
    fileMenu->addAction(openAction);
    saveAsAction = new QAction("&Save as", this);
    fileMenu->addAction(saveAsAction);
    exitAction = new QAction("E&xit", this);
    fileMenu->addAction(exitAction);

    prevAction = new QAction("&Previous Image", this);
    viewMenu->addAction(prevAction);
    nextAction = new QAction("&Next Image", this);
    viewMenu->addAction(nextAction);
    zoomInAction = new QAction("Zoom in", this);
    viewMenu->addAction(zoomInAction);
    zoomOutAction = new QAction("Zoom Out", this);
    viewMenu->addAction(zoomOutAction);
    rectAction = new QAction("&Select Rectangle", this);
    viewMenu->addAction(rectAction);
    polygonAction = new QAction("&Select Polygon", this);
    viewMenu->addAction(polygonAction);

    // add actions to toolbars
    fileToolBar->addAction(openAction);
    viewToolBar->addAction(prevAction);
    viewToolBar->addAction(nextAction);
    viewToolBar->addAction(zoomInAction);
    viewToolBar->addAction(zoomOutAction);
    viewToolBar->addAction(rectAction);
    viewToolBar->addAction(polygonAction);

    // connect the signals and slots
    connect(exitAction, SIGNAL(triggered(bool)), QApplication::instance(), SLOT(quit()));
    connect(openAction, SIGNAL(triggered(bool)), this, SLOT(openImage()));
    connect(saveAsAction, SIGNAL(triggered(bool)), this, SLOT(saveAs()));
    connect(zoomInAction, SIGNAL(triggered(bool)), this, SLOT(zoomIn()));
    connect(zoomOutAction, SIGNAL(triggered(bool)), this, SLOT(zoomOut()));
    connect(prevAction, SIGNAL(triggered(bool)), this, SLOT(prevImage()));
    connect(nextAction, SIGNAL(triggered(bool)), this, SLOT(nextImage()));
    connect(rectAction, SIGNAL(triggered(bool)), this, SLOT(rectROI()));
    connect(polygonAction, SIGNAL(triggered(bool)), this, SLOT(polygonROI()));

    setupShortcuts();
}

void MainWindow::openImage() {
    QFileDialog dialog(this);
    dialog.setWindowTitle("Open Image");
    dialog.setFileMode(QFileDialog::ExistingFile);
    dialog.setNameFilter(tr("Images (*.png *.bmp *.jpg)"));
    QStringList filePaths;
    if (dialog.exec()) {
        filePaths = dialog.selectedFiles();
        showImage(filePaths.at(0));
    }
}

void MainWindow::showImage(QString path) {
    imageScene->clear();
    imageView->resetTransform();
    QPixmap image(path);
    currentImage = imageScene->addPixmap(image);
    imageScene->update();
    imageView->setSceneRect(image.rect());
    QString status = QString("%1, %2x%3, %4 Bytes").arg(path).arg(image.width()).arg(image.height()).arg(
            QFile(path).size());
    mainStatusLabel->setText(status);
    currentImagePath = path;
}

void MainWindow::zoomIn() {
    imageView->scale(1.2, 1.2);
}

void MainWindow::zoomOut() {
    imageView->scale(1 / 1.2, 1 / 1.2);
}

void MainWindow::prevImage() {
    QFileInfo current(currentImagePath);
    QDir dir = current.absoluteDir();
    QStringList nameFilters;
    nameFilters << "*.png"
                << "*.bmp"
                << "*.jpg";
    QStringList fileNames = dir.entryList(nameFilters, QDir::Files, QDir::Name);
    if (fileNames.size()) {
        int idx = fileNames.indexOf(current.fileName());
        --idx;
        if (idx < 0) {
            idx += fileNames.size();
        }
        showImage(dir.absoluteFilePath(fileNames.at(idx)));
    } else {
        QMessageBox::information(this, "Information", "No image files.");
    }
}

void MainWindow::nextImage() {
    QFileInfo current(currentImagePath);
    QDir dir = current.absoluteDir();
    QStringList nameFilters;
    nameFilters << "*.png"
                << "*.bmp"
                << "*.jpg";
    QStringList fileNames = dir.entryList(nameFilters, QDir::Files, QDir::Name);
    if (fileNames.size()) {
        int idx = fileNames.indexOf(current.fileName());
        ++idx;
        if (idx >= fileNames.size()) {
            idx -= fileNames.size();
        }
        showImage(dir.absoluteFilePath(fileNames.at(idx)));
    } else {
        QMessageBox::information(this, "Information", "No image files.");
    }
}

void MainWindow::saveAs() {
    if (currentImage == nullptr) {
        QMessageBox::information(this, "Information", "Nothing to save.");
        return;
    }
    QFileDialog dialog(this);
    dialog.setWindowTitle("Save Image As ...");
    dialog.setFileMode(QFileDialog::AnyFile);
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    dialog.setNameFilter(tr("Images (*.png *.bmp *.jpg)"));
    QStringList fileNames;
    if (dialog.exec()) {
        fileNames = dialog.selectedFiles();
        QString pattern(".+\\.(png|bmp|jpg)");
        QRegularExpression re(QRegularExpression::anchoredPattern(pattern));
        QRegularExpressionMatch match = re.match(fileNames.at(0));
        if (match.hasMatch()) {
            currentImage->pixmap().save(fileNames.at(0));
        } else {
            QMessageBox::information(this, "Information", "Save error: bad format or filename.");
        }
    }
}

void MainWindow::rectROI() {
    QFileInfo current(currentImagePath);
    QDir dir = current.absoluteDir();
    std::string filename = dir.absoluteFilePath(current.fileName()).toStdString();
    rect::Controller controller;
    std::string log_file_name = QDir::currentPath().toStdString() + "/" + "glcm-analysis.csv";
    controller.Run(filename, 1, 256, log_file_name);
}

void MainWindow::polygonROI() {
    QFileInfo current(currentImagePath);
    QDir dir = current.absoluteDir();
    std::string filename = dir.absoluteFilePath(current.fileName()).toStdString();
    polygon::Controller controller;
    std::string log_file_name = QDir::currentPath().toStdString() + "/" + "glcm-analysis.csv";
    controller.Run(filename, 1, 256, log_file_name);
}

void MainWindow::setupShortcuts() {
    QList<QKeySequence> shortcuts;
    shortcuts << Qt::Key_Plus << Qt::Key_Equal;
    zoomInAction->setShortcuts(shortcuts);

    shortcuts.clear();
    shortcuts << Qt::Key_Minus << Qt::Key_Underscore;
    zoomOutAction->setShortcuts(shortcuts);

    shortcuts.clear();
    shortcuts << Qt::Key_Up << Qt::Key_Left;
    prevAction->setShortcuts(shortcuts);

    shortcuts.clear();
    shortcuts << Qt::Key_Down << Qt::Key_Right;
    nextAction->setShortcuts(shortcuts);

    shortcuts.clear();
    shortcuts << QKeySequence(Qt::CTRL | Qt::Key_R);
    rectAction->setShortcuts(shortcuts);

    shortcuts.clear();
    shortcuts << QKeySequence(Qt::CTRL | Qt::Key_P);
    polygonAction->setShortcuts(shortcuts);
}
