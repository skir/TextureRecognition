#-------------------------------------------------
#
# Project created by QtCreator 2016-01-12T19:01:54
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = test
CONFIG   += console
CONFIG   -= app_bundle
CONFIG += c++11
PKGCONFIG += opencv

TEMPLATE = app


SOURCES += main.cpp \
    filterprocessor.cpp \
    kernels.cpp \
    imageutils.cpp \
    videoprocessor.cpp \
    constants.cpp \
    saliencyFineGrained/staticSaliencyFineGrained.cpp

LIBS += `pkg-config opencv --libs`
#LIBS += /usr/local/lib/libopencv_saliency.so.3.1.0

HEADERS += \
    filterprocessor.h \
    kernels.h \
    imageutils.h \
    videoprocessor.h \
    constants.h \
    saliencyFineGrained/staticSaliencyFineGrained.h

QMAKE_CXXFLAGS += -fopenmp

LIBS += -fopenmp

unix:!macx: LIBS += -L$$PWD/../../../../../../../../../../usr/local/lib/ -lopencv_saliency

INCLUDEPATH += $$PWD/../../../../../../../../../../usr/local/include/opencv2/saliency
DEPENDPATH += $$PWD/../../../../../../../../../../usr/local/include/opencv2/saliency
