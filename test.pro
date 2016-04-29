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
    constants.cpp

LIBS += `pkg-config opencv --libs`

HEADERS += \
    filterprocessor.h \
    kernels.h \
    imageutils.h \
    videoprocessor.h \
    constants.h

QMAKE_CXXFLAGS += -fopenmp

LIBS += -fopenmp
