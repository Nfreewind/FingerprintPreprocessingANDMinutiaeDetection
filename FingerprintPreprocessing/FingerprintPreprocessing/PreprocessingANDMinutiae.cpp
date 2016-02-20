#include "stdafx.h"
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
#include "stdio.h"
#include "opencv2\imgproc\types_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2\highgui\highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <opencv2\core\core.hpp>
#include <opencv2/opencv.hpp>
#include <stdarg.h>

using namespace cv;
using namespace std;

const int WSIZE = 12;
const int DWSIZE = 2 * WSIZE + 1;
const double PI = 3.1415926;
const int maxCycle = 10;
const int maxPoints = 500;

CvRect erasedPixels[maxCycle][maxPoints];
int nrErased[maxCycle];

CvRect thorns[maxPoints][maxCycle];
int thornLength[maxPoints] = { 0 };
int nrThorns;

double direction[64][64];
double smdir[64][64];
double energy[64][64];
double hfEnergy[64][64];
int radius[64][64];
unsigned char mask[64][64];

//Images ----> NEW
//------
IplImage *imU, *imD;
IplImage *imWindow;
IplImage *imPigWindow;
IplImage *imBigWindow;
IplImage *imFcos;
IplImage *imFsin;
IplImage *imFsqr;
IplImage *imBigEnhanced;
IplImage *imEnhanced;
IplImage *imFinal;
IplImage *imOutput;
IplImage *imOrient;
IplImage *imSOrient;
IplImage *imEnergy;
IplImage *imhfEnergy;
IplImage *imCoherence;
IplImage* imCos;
IplImage* imSin;
IplImage* invCos;
IplImage* invSin;

//Minutiae
//--------
struct Minutia
{
	int event; // 1 - elagazas, 2 - vegzodes
	double x;
	double y;
};

//Set Value
//---------
void SetValue(IplImage* im, int x, int y, bool value)
{
	im->imageData[y*im->widthStep + x] = (unsigned char)(value ? 0 : 255);
}

//Get Value
//---------
bool GetValue(IplImage* im, int x, int y)
{
	return ((unsigned char)im->imageData[y*im->widthStep + x] < 128);
}

//Set Color 1
//-----------
void setColor(IplImage* im, int x, int y, unsigned char value)
{
	im->imageData[y*im->widthStep + x] = value;
}

//Get Color 1
//-----------
unsigned char getColor(IplImage* im, int x, int y)
{
	return (unsigned char)im->imageData[y*im->widthStep + x];
}

//Set Color 2
//-----------
void setColor2(IplImage* im, int x, int y, double v)
{
	((double*)im->imageData)[y*im->widthStep / sizeof(double)+x] = v;
}

//Get Color 2
//-----------
double getColor2(IplImage* im, int x, int y)
{
	return ((double*)im->imageData)[y*im->widthStep / sizeof(double)+x];
}

//Set RGB
//-------
void setRGB(IplImage* im, int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
	im->imageData[y*im->widthStep + 3 * x + 0] = b;
	im->imageData[y*im->widthStep + 3 * x + 1] = g;
	im->imageData[y*im->widthStep + 3 * x + 2] = r;
}

//Get RGB
//-------
void getRGB(IplImage* im, int x, int y, unsigned char& r, unsigned char& g, unsigned char& b)
{
	b = im->imageData[y*im->widthStep + 3 * x + 0];
	g = im->imageData[y*im->widthStep + 3 * x + 1];
	r = im->imageData[y*im->widthStep + 3 * x + 2];
}

//Mark Blue
//---------
bool isBlue(IplImage* im, int x, int y)
{
	unsigned char r, g, b;
	getRGB(im, x, y, r, g, b);
	return (b == 255 && r == 0 && g == 0);
}

//Mark Red
//--------
bool isRed(IplImage* im, int x, int y)
{
	unsigned char r, g, b;
	getRGB(im, x, y, r, g, b);
	return (r == 255 && g == 0 && b == 0);
}

//Mark Green
//----------
bool isGreen(IplImage* im, int x, int y)
{
	unsigned char r, g, b;
	getRGB(im, x, y, r, g, b);
	return (g == 255 && r == 0 && b == 0);
}

//Mark Gray
//---------
bool isGray(IplImage* im, int x, int y)
{
	unsigned char r, g, b;
	int i, j;
	i = (x / WSIZE) * WSIZE + WSIZE / 2;
	j = (y / WSIZE) * WSIZE + WSIZE / 2;
	getRGB(im, i, j, r, g, b);
	return (g == 192 && r == 192 && b == 192);
}

//Need to thinning the fingerprint image
//--------------------------------------
int GolayL(IplImage* src, IplImage* dst)
{
	int count;
	int i = 0;
	do
	{
		count = 0;
		for (int index = 1; index <= 8; index++)
		{
			for (int x = 1; x<src->width - 1; x++)
			for (int y = 1; y<src->height - 1; y++)
			if (GetValue(src, x, y))
			{
				switch (index){
				case 1:
					if (!GetValue(src, x - 1, y - 1) && !GetValue(src, x, y - 1) && !GetValue(src, x + 1, y - 1) &&
						GetValue(src, x, y) &&
						GetValue(src, x - 1, y + 1) && GetValue(src, x, y + 1) && GetValue(src, x + 1, y + 1))
					{
						SetValue(dst, x, y, false);
						count++;
					}
					break;
				case 2:
					if (!GetValue(src, x, y - 1) && !GetValue(src, x + 1, y - 1) &&
						GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
						GetValue(src, x, y + 1))
					{
						SetValue(dst, x, y, false);
						count++;
					}
					break;
				case 3:
					if (GetValue(src, x - 1, y - 1) && !GetValue(src, x + 1, y - 1) &&
						GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
						GetValue(src, x - 1, y + 1) && !GetValue(src, x + 1, y + 1))
					{
						SetValue(dst, x, y, false);
						count++;
					}
					break;
				case 4:
					if (GetValue(src, x, y - 1) &&
						GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
						!GetValue(src, x, y + 1) && !GetValue(src, x + 1, y + 1))
					{
						SetValue(dst, x, y, false);
						count++;
					}
					break;
				case 5:
					if (GetValue(src, x - 1, y - 1) && GetValue(src, x, y - 1) && GetValue(src, x + 1, y - 1) &&
						GetValue(src, x, y) &&
						!GetValue(src, x - 1, y + 1) && !GetValue(src, x, y + 1) && !GetValue(src, x + 1, y + 1))
					{
						SetValue(dst, x, y, false);
						count++;
					}
					break;
				case 6:
					if (GetValue(src, x, y - 1) &&
						!GetValue(src, x - 1, y) && GetValue(src, x, y) && GetValue(src, x + 1, y) &&
						!GetValue(src, x - 1, y + 1) && !GetValue(src, x, y + 1))
					{
						SetValue(dst, x, y, false);
						count++;
					}
					break;
				case 7:
					if (!GetValue(src, x - 1, y - 1) && GetValue(src, x + 1, y - 1) &&
						!GetValue(src, x - 1, y) && GetValue(src, x, y) && GetValue(src, x + 1, y) &&
						!GetValue(src, x - 1, y + 1) && GetValue(src, x + 1, y + 1))
					{
						SetValue(dst, x, y, false);
						count++;
					}
					break;
				case 8:
					if (!GetValue(src, x - 1, y - 1) && !GetValue(src, x, y - 1) &&
						!GetValue(src, x - 1, y) && GetValue(src, x, y) && GetValue(src, x + 1, y) &&
						GetValue(src, x, y + 1))
					{
						SetValue(dst, x, y, false);
						count++;
					}
					break;
				}
			}
			cvResize(dst, src);
		}
		printf("i = %d, count = %d\n", i, count);
		cvShowImage("The Fingerprint", src);

		cvWaitKey(300);
		i++;
	} while (count>0);
	return i;
}

//Need to find the ridge ending in fingerprint
//--------------------------------------------
int GolayE(IplImage* im, IplImage* im2)
{
	int count;
	int i = 0;
	int cycle = 0;
	do
	{
		count = 0;
		for (int index = 1; index <= 8; index++)
		{
			for (int x = 1; x<im->width - 1; x++)
			for (int y = 1; y<im->height - 1; y++)
			if (GetValue(im, x, y))
			{
				switch (index){
				case 1:
					if (GetValue(im, x, y) && GetValue(im, x, y - 1) &&
						!GetValue(im, x - 1, y) && !GetValue(im, x + 1, y) &&
						!GetValue(im, x - 1, y + 1) && !GetValue(im, x, y + 1) && !GetValue(im, x + 1, y + 1)
						&& GetValue(im2, x, y))
					{
						SetValue(im2, x, y, false);
						erasedPixels[cycle][count++] = cvRect(x, y, 2, 1);
					}
					break;
				case 2:
					if (!GetValue(im, x - 1, y - 1) &&
						!GetValue(im, x - 1, y) && GetValue(im, x, y) && !GetValue(im, x + 1, y) &&
						!GetValue(im, x - 1, y + 1) && !GetValue(im, x, y + 1) && !GetValue(im, x + 1, y + 1)
						&& GetValue(im2, x, y))
					{
						SetValue(im2, x, y, false);
						erasedPixels[cycle][count++] = cvRect(x, y, 2, 1);
					}
					break;
				case 3:
					if (!GetValue(im, x - 1, y - 1) && !GetValue(im, x, y - 1) &&
						!GetValue(im, x - 1, y) && GetValue(im, x, y) && GetValue(im, x + 1, y) &&
						!GetValue(im, x - 1, y + 1) && !GetValue(im, x, y + 1)
						&& GetValue(im2, x, y))
					{
						SetValue(im2, x, y, false);
						erasedPixels[cycle][count++] = cvRect(x, y, 2, 1);
					}
					break;
				case 4:
					if (!GetValue(im, x - 1, y - 1) && !GetValue(im, x, y - 1) && !GetValue(im, x + 1, y - 1) &&
						!GetValue(im, x - 1, y) && GetValue(im, x, y) &&
						!GetValue(im, x - 1, y + 1) && !GetValue(im, x, y + 1)
						&& GetValue(im2, x, y))
					{
						SetValue(im2, x, y, false);
						erasedPixels[cycle][count++] = cvRect(x, y, 2, 1);
					}
					break;
				case 5:
					if (!GetValue(im, x - 1, y - 1) && !GetValue(im, x, y - 1) && !GetValue(im, x + 1, y - 1) &&
						!GetValue(im, x - 1, y) && GetValue(im, x, y) && !GetValue(im, x + 1, y) &&
						GetValue(im, x, y + 1)
						&& GetValue(im2, x, y))
					{
						SetValue(im2, x, y, false);
						erasedPixels[cycle][count++] = cvRect(x, y, 2, 1);
					}
					break;
				case 6:
					if (!GetValue(im, x - 1, y - 1) && !GetValue(im, x, y - 1) && !GetValue(im, x + 1, y - 1) &&
						!GetValue(im, x - 1, y) && GetValue(im, x, y) && !GetValue(im, x + 1, y) &&
						!GetValue(im, x + 1, y + 1)
						&& GetValue(im2, x, y))
					{
						SetValue(im2, x, y, false);
						erasedPixels[cycle][count++] = cvRect(x, y, 2, 1);
					}
					break;
				case 7:
					if (!GetValue(im, x, y - 1) && !GetValue(im, x + 1, y - 1) &&
						GetValue(im, x - 1, y) && GetValue(im, x, y) && !GetValue(im, x + 1, y) &&
						!GetValue(im, x, y + 1) && !GetValue(im, x + 1, y + 1)
						&& GetValue(im2, x, y))
					{
						SetValue(im2, x, y, false);
						erasedPixels[cycle][count++] = cvRect(x, y, 2, 1);
					}
					break;
				case 8:
					if (!GetValue(im, x, y - 1) && !GetValue(im, x + 1, y - 1) &&
						GetValue(im, x, y) && !GetValue(im, x + 1, y) &&
						!GetValue(im, x - 1, y + 1) && !GetValue(im, x, y + 1) && !GetValue(im, x + 1, y + 1)
						&& GetValue(im2, x, y))
					{
						SetValue(im2, x, y, false);
						erasedPixels[cycle][count++] = cvRect(x, y, 2, 1);
					}
					break;
				}
			}
		}
		cvCopy(im2, im);
		printf("i = %d, count = %d\n", i, count);
		cvShowImage("The Fingerprint", im);
		nrErased[cycle] = count;

		cvWaitKey(300);
		i++;
		cycle++;

	} while (count>0 && cycle<maxCycle);
	return i;
}

//Need to find the ridge bifurcation in fingerprint
//-------------------------------------------------
bool GolayQcond(IplImage* src, int x, int y, int index)
{
	bool res;
	switch (index){
	case 1:
		res = GetValue(src, x - 1, y - 1) && !GetValue(src, x, y - 1) && GetValue(src, x + 1, y - 1) &&
			GetValue(src, x, y) &&
			GetValue(src, x - 1, y + 1) && !GetValue(src, x, y + 1);
		break;
	case 2:
		res = GetValue(src, x - 1, y - 1) &&
			!GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
			GetValue(src, x - 1, y + 1) && GetValue(src, x + 1, y + 1);
		break;
	case 3:
		res = !GetValue(src, x, y - 1) && GetValue(src, x + 1, y - 1) &&
			GetValue(src, x, y) &&
			GetValue(src, x - 1, y + 1) && !GetValue(src, x, y + 1) && GetValue(src, x + 1, y + 1);
		break;
	case 4:
		res = GetValue(src, x - 1, y - 1) && GetValue(src, x + 1, y - 1) &&
			!GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
			GetValue(src, x + 1, y + 1);
		break;
	case 5:
		res = GetValue(src, x - 1, y + 1) && !GetValue(src, x, y + 1) && GetValue(src, x + 1, y + 1) &&
			GetValue(src, x, y) &&
			GetValue(src, x - 1, y - 1) && !GetValue(src, x, y - 1);
		break;
	case 6:
		res = GetValue(src, x + 1, y - 1) &&
			!GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
			GetValue(src, x - 1, y + 1) && GetValue(src, x + 1, y + 1);
		break;
	case 7:
		res = !GetValue(src, x, y + 1) && GetValue(src, x + 1, y + 1) &&
			GetValue(src, x, y) &&
			GetValue(src, x - 1, y - 1) && !GetValue(src, x, y - 1) && GetValue(src, x + 1, y - 1);
		break;
	case 8:
		res = GetValue(src, x - 1, y - 1) && GetValue(src, x + 1, y - 1) &&
			!GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
			GetValue(src, x - 1, y + 1);
		break;
	case 9:
		res = GetValue(src, x - 1, y - 1) && !GetValue(src, x, y - 1) && !GetValue(src, x + 1, y - 1) &&
			!GetValue(src, x - 1, y) && GetValue(src, x, y) && GetValue(src, x + 1, y) &&
			!GetValue(src, x - 1, y + 1) && GetValue(src, x, y + 1) && !GetValue(src, x + 1, y + 1);
		break;
	case 10:
		res = !GetValue(src, x - 1, y - 1) && GetValue(src, x, y - 1) && !GetValue(src, x + 1, y - 1) &&
			!GetValue(src, x - 1, y) && GetValue(src, x, y) && GetValue(src, x + 1, y) &&
			GetValue(src, x - 1, y + 1) && !GetValue(src, x, y + 1) && !GetValue(src, x + 1, y + 1);
		break;
	case 11:
		res = !GetValue(src, x - 1, y - 1) && GetValue(src, x, y - 1) && !GetValue(src, x + 1, y - 1) &&
			GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
			!GetValue(src, x - 1, y + 1) && !GetValue(src, x, y + 1) && GetValue(src, x + 1, y + 1);
		break;
	case 12:
		res = !GetValue(src, x - 1, y - 1) && !GetValue(src, x, y - 1) && GetValue(src, x + 1, y - 1) &&
			GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
			!GetValue(src, x - 1, y + 1) && GetValue(src, x, y + 1) && !GetValue(src, x + 1, y + 1);
		break;
	case 13:
		res = !GetValue(src, x, y - 2) &&
			GetValue(src, x - 1, y - 1) && GetValue(src, x + 1, y - 1) &&
			!GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
			!GetValue(src, x - 1, y + 1) && GetValue(src, x, y + 1) && !GetValue(src, x + 1, y + 1);
		break;
	case 14:
		res = GetValue(src, x - 1, y - 1) && !GetValue(src, x, y - 1) && !GetValue(src, x + 1, y - 1) &&
			!GetValue(src, x - 2, y) && GetValue(src, x, y) && GetValue(src, x + 1, y) &&
			GetValue(src, x - 1, y + 1) && !GetValue(src, x, y + 1) && !GetValue(src, x + 1, y + 1);
		break;
	case 15:
		res = !GetValue(src, x - 1, y - 1) && GetValue(src, x, y - 1) && !GetValue(src, x + 1, y - 1) &&
			!GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 1, y) &&
			GetValue(src, x - 1, y + 1) && GetValue(src, x + 1, y + 1) &&
			!GetValue(src, x, y + 2);

		break;
	case 16:
		res = !GetValue(src, x - 1, y - 1) && !GetValue(src, x, y - 1) && GetValue(src, x + 1, y - 1) &&
			GetValue(src, x - 1, y) && GetValue(src, x, y) && !GetValue(src, x + 2, y) &&
			!GetValue(src, x - 1, y + 1) && !GetValue(src, x, y + 1) && GetValue(src, x + 1, y + 1);
		break;
	case 17:
		res = !GetValue(src, x - 1, y - 1) && GetValue(src, x, y - 1) && !GetValue(src, x + 1, y - 1) &&
			GetValue(src, x - 1, y) && GetValue(src, x, y) && GetValue(src, x + 1, y) &&
			!GetValue(src, x - 1, y + 1) && GetValue(src, x, y + 1) && !GetValue(src, x + 1, y + 1);
		break;
	}
	return res;
}

//Need to find the ridge bifurcation in fingerprint
//-------------------------------------------------
int GolayQ(IplImage* src, IplImage* dst)
{
	int count = 0;

	for (int x = 2; x<src->width - 2; x++)
	for (int y = 2; y<src->height - 2; y++)
	if (GetValue(src, x, y))
	{
		for (int index = 1; index <= 17; index++)
		{
			if (GolayQcond(src, x, y, index))
			{
				setRGB(dst, x, y, 255, 0, 0);
				count++;
			}
		}
	}

	return count;
}

//Absolute
//--------
int abso(int a)
{
	return a<0 ? -a : a;
}

//Sinus and cosinus table
//-----------------------
void createSinCosTable()
{
	imCos = cvCreateImage(cvSize((2 * WSIZE + 1)*(2 * WSIZE + 1), (2 * WSIZE + 1)*(2 * WSIZE + 1)), IPL_DEPTH_64F, 1);
	imSin = cvCreateImage(cvSize((2 * WSIZE + 1)*(2 * WSIZE + 1), (2 * WSIZE + 1)*(2 * WSIZE + 1)), IPL_DEPTH_64F, 1);

	for (int i = -WSIZE; i <= WSIZE; i++) for (int j = -WSIZE; j <= WSIZE; j++)
	{
		double om1 = i*PI / ((double)(4 * WSIZE)), om2 = j*PI / ((double)(4 * WSIZE));
		for (int p = -WSIZE; p <= WSIZE; p++) for (int q = -WSIZE; q <= WSIZE; q++)
		{
			setColor2(imCos, (i + WSIZE)*(2 * WSIZE + 1) + (WSIZE + p), (j + WSIZE)*(2 * WSIZE + 1) + (WSIZE + q), cos(om1*p + om2*q));
			setColor2(imSin, (i + WSIZE)*(2 * WSIZE + 1) + (WSIZE + p), (j + WSIZE)*(2 * WSIZE + 1) + (WSIZE + q), sin(om1*p + om2*q));
		}
	}
	FILE* f = fopen("costable.dat", "wb+");
	fwrite(imCos->imageData, 1, (2 * WSIZE + 1)*(2 * WSIZE + 1)*(2 * WSIZE + 1)*(2 * WSIZE + 1)*sizeof(double), f);
	fclose(f);
	f = fopen("sintable.dat", "wb+");
	fwrite(imSin->imageData, 1, (2 * WSIZE + 1)*(2 * WSIZE + 1)*(2 * WSIZE + 1)*(2 * WSIZE + 1)*sizeof(double), f);
	fclose(f);
}

//Create new window image
//-----------------------
void createWindowImage()
{
	int x, y;
	imWindow = cvCreateImage(cvSize(2 * WSIZE + 1, 2 * WSIZE + 1), IPL_DEPTH_64F, 1);
	cvSet(imWindow, cvScalar(1.0, 0.0, 0.0, 0.0));

	for (x = 0; x<WSIZE / 2; x++) for (y = 0; y<WSIZE / 2; y++)
	{
		double value = 0.25*(1.0 - cos(2.0*x*PI / WSIZE))*(1.0 - cos(2.0*y*PI / WSIZE));
		((double*)imWindow->imageData)[y*imWindow->width + x] = value;
		((double*)imWindow->imageData)[y*imWindow->width + (2 * WSIZE - x)] = value;
		((double*)imWindow->imageData)[(2 * WSIZE - y)*imWindow->width + x] = value;
		((double*)imWindow->imageData)[(2 * WSIZE - y)*imWindow->width + (2 * WSIZE - x)] = value;

	}

	for (y = 0; y<WSIZE / 2; y++)
	{
		double value = 0.5*(1.0 - cos(2.0*y*PI / WSIZE));
		for (x = WSIZE / 2; x <= 3 * WSIZE / 2; x++)
		{
			((double*)imWindow->imageData)[y*imWindow->width + x] = value;
			((double*)imWindow->imageData)[(2 * WSIZE - y)*imWindow->width + x] = value;
			((double*)imWindow->imageData)[x*imWindow->width + y] = value;
			((double*)imWindow->imageData)[x*imWindow->width + (2 * WSIZE - y)] = value;

		}
	}
}

//myRound
//-------
int myRound(double d)
{
	return (int)d;
}

//Spektrum image
//--------------
void showSpectrum(int x, int y)
{
	const int z = WSIZE;
	int i, j;
	IplImage *im = cvCreateImage(cvSize(z*DWSIZE, z*DWSIZE), IPL_DEPTH_64F, 1);
	for (i = 0; i<DWSIZE; i++) for (j = 0; j<DWSIZE; j++)
	{
		cvSetImageROI(im, cvRect(i*z, j*z, z, z));
		cvSet(im, cvScalar(3.0*getColor2(imFsqr, x*DWSIZE + i, y*DWSIZE + j), 0, 0, 0));
	}
	cvResetImageROI(im);
	cvNamedWindow("Spektrum", CV_WINDOW_AUTOSIZE);
	cvShowImage("Spektrum", im);
	cvSaveImage("spektrum.bmp", im);
	cvReleaseImage(&im);
	cvWaitKey();
}

//getEnergyTreshold 
//-----------------
double getEnergyThreshold(int bigX, int bigY)
{
	int maxSteps = bigX>bigY ? bigX / 2 : bigY / 2;
	double thresh[4] = { 0 };

	for (int i = 0; i<4; i++)
	{
		int actStep = 1;
		double maxRise = 0;
		int maxRisePos = 0;

		switch (i)
		{
		case 0:
			while (hfEnergy[actStep][actStep]>hfEnergy[actStep - 1][actStep - 1] && actStep<maxSteps)
			{
				if (hfEnergy[actStep][actStep] - hfEnergy[actStep - 1][actStep - 1] > maxRise)
				{
					maxRise = hfEnergy[actStep][actStep] - hfEnergy[actStep - 1][actStep - 1];
					maxRisePos = actStep;
				}
				actStep++;
			}
			if (maxRisePos>0) thresh[i] = (hfEnergy[maxRisePos][maxRisePos] + hfEnergy[maxRisePos - 1][maxRisePos - 1]) * 0.5;
			break;
		case 1:
			while (hfEnergy[actStep][(bigY - 1) - actStep]>hfEnergy[actStep - 1][(bigY - 1) - actStep + 1] && actStep<maxSteps)
			{
				if (hfEnergy[actStep][(bigY - 1) - actStep] - hfEnergy[actStep - 1][(bigY - 1) - actStep + 1] > maxRise)
				{
					maxRise = hfEnergy[actStep][(bigY - 1) - actStep] - hfEnergy[actStep - 1][(bigY - 1) - actStep + 1];
					maxRisePos = actStep;
				}
				actStep++;
			}
			if (maxRisePos>0) thresh[i] = (hfEnergy[maxRisePos][(bigY - 1) - maxRisePos] + hfEnergy[maxRisePos - 1][(bigY - 1) - maxRisePos + 1]) * 0.5;
			break;
		case 2:
			while (hfEnergy[(bigX - 1) - actStep][actStep] > hfEnergy[(bigX - 1) - actStep + 1][actStep - 1] && actStep<maxSteps)
			{
				if (hfEnergy[(bigX - 1) - actStep][actStep] - hfEnergy[(bigX - 1) - actStep + 1][actStep - 1] > maxRise)
				{
					maxRise = hfEnergy[(bigX - 1) - actStep][actStep] - hfEnergy[(bigX - 1) - actStep + 1][actStep - 1];
					maxRisePos = actStep;
				}
				actStep++;
			}
			if (maxRisePos>0) thresh[i] = (hfEnergy[(bigX - 1) - maxRisePos][maxRisePos] + hfEnergy[(bigX - 1) - maxRisePos + 1][maxRisePos - 1]) * 0.5;
			break;
		case 3:
			while (hfEnergy[(bigX - 1) - actStep][(bigY - 1) - actStep]>hfEnergy[(bigX - 1) - actStep + 1][(bigY - 1) - actStep + 1] && actStep<maxSteps)
			{
				if (hfEnergy[(bigX - 1) - actStep][(bigY - 1) - actStep] - hfEnergy[(bigX - 1) - actStep + 1][(bigY - 1) - actStep + 1] > maxRise)
				{
					maxRise = hfEnergy[(bigX - 1) - actStep][(bigY - 1) - actStep] - hfEnergy[(bigX - 1) - actStep + 1][(bigY - 1) - actStep + 1];
					maxRisePos = actStep;
				}
				actStep++;
			}
			if (maxRisePos>0) thresh[i] = (hfEnergy[(bigX - 1) - maxRisePos][(bigY - 1) - maxRisePos] + hfEnergy[(bigX - 1) - maxRisePos + 1][(bigY - 1) - maxRisePos + 1]) * 0.5;
			break;
		}
	}

	return (thresh[0] + thresh[1] + thresh[2] + thresh[3]) / ((thresh[0]>0) + (thresh[1]>0) + (thresh[2]>0) + (thresh[3]>0));
}

//createEnergyMask
//----------------
void createEnergyMask(int bigX, int bigY, double thresh)
{
	int i, j;
	for (i = 0; i<bigX; i++) for (j = 0; j<bigY; j++) mask[i][j] = 1;

	i = 0; j = 0;
	while (hfEnergy[i][j] < thresh)
	{
		while (hfEnergy[i][j] < thresh)
		{
			mask[i][j] = 0; i++;
		}
		i = 0; j++;
	}

	i = 0; j = bigY - 1;
	while (hfEnergy[i][j] < thresh)
	{
		while (hfEnergy[i][j] < thresh)
		{
			mask[i][j] = 0; i++;
		}
		i = 0; j--;
	}

	i = bigX - 1; j = 0;
	while (hfEnergy[i][j] < thresh)
	{
		while (hfEnergy[i][j] < thresh)
		{
			mask[i][j] = 0; i--;
		}
		i = bigX - 1; j++;
	}

	i = bigX - 1; j = bigY - 1;
	while (hfEnergy[i][j] < thresh)
	{
		while (hfEnergy[i][j] < thresh)
		{
			mask[i][j] = 0; i--;
		}
		i = bigX - 1; j--;
	}
}



int main()
{

	//Create the sinus and cosinus table
	//----------------------------------
	createSinCosTable();
	
	int x, y, i, j;
	int tau1, tau2;

	double maxenergy = 0;

	//Open and load the input original fingerprint image
	//--------------------------------------------------
	IplImage* im = cvLoadImage("image.bmp", 0);
	cvNamedWindow("Original Input Image", CV_WINDOW_AUTOSIZE);
	cvShowImage("Original Input Image", im);
	cvWaitKey();

	//Make clones
	//-----------
	imU = cvCloneImage(im);
	imOrient = cvCloneImage(im);
	imSOrient = cvCloneImage(im);
	imEnergy = cvCloneImage(im);
	imhfEnergy = cvCloneImage(im);
	imCoherence = cvCloneImage(im);
	cvSet(imOrient, cvScalar(255, 255, 255, 0.0));
	cvSet(imSOrient, cvScalar(255, 255, 255, 0.0));
	cvSet(imEnergy, cvScalar(255, 255, 255, 0.0));
	cvSet(imCoherence, cvScalar(255, 255, 255, 0.0));

	//Create the imD image 
	//--------------------
	imD = cvCreateImage(cvSize(im->width, im->height), IPL_DEPTH_64F, 1);
	cvConvertScale(imU, imD, 1.0 / 255.0);
	CvScalar avg, sdv;
	cvAvgSdv(imD, &avg, &sdv);
	cvAddWeighted(imD, 1.0, imD, 0.0, -avg.val[0], imD);

	//Show the imD image 
	//------------------
	cvNamedWindow("imD", CV_WINDOW_AUTOSIZE);
	cvShowImage("imD", imD);
	cvSaveImage("imD.bmp", imD);
	cvWaitKey();

	//Create the big window for cosinus and sinus image
	//-------------------------------------------------
	createWindowImage();

	int bigX, bigY;
	bigX = (im->width - (2 * WSIZE + 1)) / WSIZE + 1;
	bigY = (im->height - (2 * WSIZE + 1)) / WSIZE + 1;
	imBigWindow = cvCreateImage(cvSize(bigX*(2 * WSIZE + 1), bigY*(2 * WSIZE + 1)), IPL_DEPTH_64F, 1);

	for (tau1 = WSIZE; tau1<im->width - WSIZE; tau1 += WSIZE) for (tau2 = WSIZE; tau2<im->height - WSIZE; tau2 += WSIZE)
	{
		cvSetImageROI(imD, cvRect(tau1 - WSIZE, tau2 - WSIZE, 2 * WSIZE + 1, 2 * WSIZE + 1));
		cvSetImageROI(imBigWindow, cvRect(((tau1 - WSIZE) / WSIZE)*(2 * WSIZE + 1), ((tau2 - WSIZE) / WSIZE)*(2 * WSIZE + 1), 2 * WSIZE + 1, 2 * WSIZE + 1));
		cvMul(imD, imWindow, imBigWindow);
	}

	cvResetImageROI(imD);
	cvResetImageROI(imBigWindow);

	//Show the cosinus image
	//----------------------
	cvNamedWindow("Cosinus", CV_WINDOW_AUTOSIZE);
	cvShowImage("Cosinus", imCos);
	cvSaveImage("cosinus.bmp", imCos);
	cvWaitKey();

	//Show the sinus image
	//----------------------
	cvNamedWindow("Sinus", CV_WINDOW_AUTOSIZE);
	cvShowImage("Sinus", imSin);
	cvSaveImage("sinus.bmp", imSin);
	cvWaitKey();

	//Create imFcos, imFsin and imFsqr images
	//---------------------------------------
	imFcos = cvCreateImage(cvSize(bigX*DWSIZE, bigY*DWSIZE), IPL_DEPTH_64F, 1);
	imFsin = cvCreateImage(cvSize(bigX*DWSIZE, bigY*DWSIZE), IPL_DEPTH_64F, 1);
	imFsqr = cvCreateImage(cvSize(bigX*DWSIZE, bigY*DWSIZE), IPL_DEPTH_64F, 1);
	cvSet(imFcos, cvScalar(0, 0, 0, 0));
	cvSet(imFsin, cvScalar(0, 0, 0, 0));

	double pRadius[32][48][WSIZE] = { 0 };
	double pTheta[32][48][24] = { 0 };

	
	IplImage* imAux = cvCloneImage(imWindow);
	double maxTotalEnergy = 0;
	for (x = 0; x<bigX; x++) for (y = 0; y<bigY; y++)
	{
		CvRect roi = cvRect(x*DWSIZE, y*DWSIZE, DWSIZE, DWSIZE);
		cvSetImageROI(imBigWindow, roi);
		cvSetImageROI(imFcos, roi);
		cvSetImageROI(imFsin, roi);
		double totalEnergy = 0;
		double totalhfEnergy = 0;
		for (i = 0; i<DWSIZE; i++) for (j = 0; j<DWSIZE; j++)
		{
			double sinX, cosX;
			cvSetImageROI(imCos, cvRect(i*DWSIZE, j*DWSIZE, DWSIZE, DWSIZE));
			cvSetImageROI(imSin, cvRect(i*DWSIZE, j*DWSIZE, DWSIZE, DWSIZE));
			cvMul(imCos, imBigWindow, imAux);
			cvAvgSdv(imAux, &avg, &sdv);
			cosX = avg.val[0] * DWSIZE;
			setColor2(imFcos, x*DWSIZE + i, y*DWSIZE + j, cosX);

			cvMul(imSin, imBigWindow, imAux);
			cvAvgSdv(imAux, &avg, &sdv);
			sinX = -avg.val[0] * DWSIZE;
			setColor2(imFsin, x*DWSIZE + i, y*DWSIZE + j, sinX);
			setColor2(imFsqr, x*DWSIZE + i, y*DWSIZE + j, (pow(sinX, 2) + pow(cosX, 2)));
			totalEnergy += pow(sinX, 2) + pow(cosX, 2);

			int radius, theta;
			radius = (int)sqrt((float)((i - WSIZE)*(i - WSIZE) + (j - WSIZE)*(j - WSIZE)));

			if (radius>WSIZE / 4) totalhfEnergy += pow(sinX, 2) + pow(cosX, 2);

			if (radius >= 0 && radius<WSIZE)
			{
				pRadius[x][y][radius] += pow(sinX, 2) + pow(cosX, 2);

				if (i == 0)
				{
					theta = j>0 ? 6 : 18;
				}
				else
				{
					double szog = atan2((float)(j - WSIZE), (float)(i - WSIZE));
					if (szog<0) szog += 2 * PI;
					theta = myRound(12 * szog / PI);
					if (theta>23) theta = 0;
				}
				pTheta[x][y][theta] += pow(sinX, 2) + pow(cosX, 2);
			}
		}
		
		radius[x][y] = WSIZE;
		if (pRadius[x][y][WSIZE - 1]>pRadius[x][y][WSIZE - 2]) radius[x][y] = WSIZE - 1;
		for (i = WSIZE - 2; i>2; i--)
		if (pRadius[x][y][i] > pRadius[x][y][i - 1] && pRadius[x][y][i] > pRadius[x][y][i + 1] && pRadius[x][y][i] > pRadius[x][y][radius[x][y]]) radius[x][y] = i;

		for (i = 0; i<WSIZE; i++) pRadius[x][y][i] /= totalEnergy;
		for (i = 0; i<24; i++) pTheta[x][y][i] /= totalEnergy;

		double eSum1 = 0.0, eSum2 = 0.0;
		for (i = 0; i<24; i++)
		{
			eSum1 += pTheta[x][y][i] * sin(2.0*i*PI / 12.0);
			eSum2 += pTheta[x][y][i] * cos(2.0*i*PI / 12.0);
		}
		double irany = 0.5 * atan2(eSum1, eSum2);
		direction[x][y] = irany;
		irany += PI / 2;
		if (totalEnergy>maxTotalEnergy) maxTotalEnergy = totalEnergy;
		energy[x][y] = totalEnergy;
		hfEnergy[x][y] = totalhfEnergy;
		
		cvLine(imOrient, cvPoint(myRound(WSIZE + x*WSIZE + (cos(irany)*WSIZE / 2.0)), myRound(WSIZE + y*WSIZE + (sin(irany)*WSIZE / 2.0))), cvPoint(myRound(WSIZE + x*WSIZE - (cos(irany)*WSIZE / 2.0)), myRound(WSIZE + y*WSIZE - (sin(irany)*WSIZE / 2.0))), cvScalar(0, 0, 0, 0));
		cvCircle(imOrient, cvPoint(myRound(WSIZE + x*WSIZE + (cos(irany)*WSIZE / 2.0)), myRound(WSIZE + y*WSIZE + (sin(irany)*WSIZE / 2.0))), 2, cvScalar(0, 0, 0, 0));
	}

	//Energy threshold
	//----------------
	double threshold = getEnergyThreshold(bigX, bigY);
	createEnergyMask(bigX, bigY, threshold);

	//Show spectrum
	//-------------
	showSpectrum(6, 13);

	//Show orientation fingerprint image
	//----------------------------------
	cvResetImageROI(imBigWindow);
	cvResetImageROI(imFcos);
	cvResetImageROI(imFsin);
	cvNamedWindow("Orientation", CV_WINDOW_AUTOSIZE);
	cvShowImage("Orientation", imOrient);
	cvSaveImage("orientation.bmp", imOrient);
	cvWaitKey();

	double gauss[3] = { 0.25, 0.125, 0.0625 };

	//Calculate coherence
	//-------------------
	for (x = 0; x<bigX; x++) for (y = 0; y<bigY; y++)
	{
		double sum1 = 0, sum2 = 0, sum = 0, cohsum = 0;
		for (i = -1; i <= 1; i++) for (j = -1; j <= 1; j++)
		if (x + i >= 1 && y + j >= 1 && x + i<bigX - 1 && y + j<bigY - 1)
		{
			sum += gauss[abs(i) + abs(j)];
			sum1 += gauss[abs(i) + abs(j)] * sin(2 * direction[x + i][y + j]);
			sum2 += gauss[abs(i) + abs(j)] * cos(2 * direction[x + i][y + j]);
			cohsum += cos(direction[x][y] - direction[x + i][y + j]);
		}
		double irany = 0.5 * atan2(sum1, sum2);

		smdir[x][y] = irany;

		double coherence = cohsum / (sum > 0.99 ? 9.0 : 6.0);
		printf("coherence %f\n", coherence);

		cvLine(imSOrient, cvPoint(myRound(WSIZE + x*WSIZE + (cos(irany)*WSIZE / 2.0)), myRound(WSIZE + y*WSIZE + (sin(irany)*WSIZE / 2.0))), cvPoint(myRound(WSIZE + x*WSIZE - (cos(irany)*WSIZE / 2.0)), myRound(WSIZE + y*WSIZE - (sin(irany)*WSIZE / 2.0))), cvScalar(0, 0, 0, 0));
		cvCircle(imSOrient, cvPoint(myRound(WSIZE + x*WSIZE + (cos(irany)*WSIZE / 2.0)), myRound(WSIZE + y*WSIZE + (sin(irany)*WSIZE / 2.0))), 2, cvScalar(0, 0, 0, 0));

		for (i = -1; i <= 1; i++) for (j = -1; j <= 1; j++)
		if (x + i >= 1 && y + j >= 1 && x + i<bigX - 1 && y + j<bigY - 1)
		{
			sum += gauss[abs(i) + abs(j)] * (radius[x + i][y + j] >= 3 ? 1.0 : 0.01);
			sum1 += energy[x][y] * gauss[abs(i) + abs(j)] * (radius[x + i][y + j] >= 3 ? 1.0 : 0.01);
		}

		cvSetImageROI(imEnergy, cvRect(x*WSIZE + WSIZE / 2, y*WSIZE + WSIZE / 2, WSIZE, WSIZE));
		cvSet(imEnergy, cvScalar(myRound(255.0 - (255.0*(sum1 / sum) / maxTotalEnergy)), 0, 0, 0));

		if (mask[x][y])
		{
			cvSetImageROI(imhfEnergy, cvRect(x*WSIZE + WSIZE / 2, y*WSIZE + WSIZE / 2, WSIZE, WSIZE));
			cvSet(imhfEnergy, cvScalar(myRound(255.0 - (255.0*hfEnergy[x][y] / maxTotalEnergy)), 0, 0, 0));
		}

		cvSetImageROI(imCoherence, cvRect(x*WSIZE + WSIZE / 2, y*WSIZE + WSIZE / 2, WSIZE, WSIZE));
		cvSet(imCoherence, cvScalar(myRound(255.0*(1.0 - coherence)), 0, 0, 0));

	}

	cvResetImageROI(imEnergy);
	cvResetImageROI(imhfEnergy);
	cvResetImageROI(imCoherence);

	//Show SOrientation image
	//-----------------------
	cvNamedWindow("SOrientation", CV_WINDOW_AUTOSIZE);
	cvShowImage("SOrientation", imSOrient);
	cvSaveImage("SOrientation.bmp", imSOrient);
	cvWaitKey();

	//Show energy image
	//-----------------
	cvNamedWindow("hfEnergy", CV_WINDOW_AUTOSIZE);
	cvShowImage("hfEnergy", imhfEnergy);
	cvSaveImage("hfEnergy.bmp", imhfEnergy);
	cvWaitKey();

	//Show coherence image
	//--------------------
	cvNamedWindow("imCoherence", CV_WINDOW_AUTOSIZE);
	cvShowImage("imCoherence", imCoherence);
	cvSaveImage("imCoherence.bmp", imCoherence);
	cvWaitKey();

	//Create imBigEnhanced, imEnhanced and imFinal images
	//---------------------------------------------------
	imBigEnhanced = cvCreateImage(cvSize(bigX*DWSIZE, bigY*DWSIZE), IPL_DEPTH_64F, 1);
	imEnhanced = cvCreateImage(cvSize(bigX*(WSIZE), bigY*(WSIZE)), IPL_DEPTH_64F, 1);
	imFinal = cvCreateImage(cvSize(bigX*(WSIZE), bigY*(WSIZE)), IPL_DEPTH_8U, 1);
	cvSet(imBigEnhanced, cvScalar(0, 0, 0, 0));
	cvSet(imEnhanced, cvScalar(1, 0, 0, 0));

	for (x = 0; x<bigX; x++) for (y = 0; y<bigY; y++)
	{
		for (i = -WSIZE; i <= WSIZE; i++) for (j = -WSIZE; j <= WSIZE; j++)
		{
			double rr = sqrt((double)(i*i + j*j));
			double iranyszorzo = exp(-pow(atan(tan(atan2((float)j, (float)i) - smdir[x][y])), 2) * 16);
			double sugarszorzo = exp(-pow(rr - radius[x][y], 2) * 16);

			setColor2(imFcos, x*DWSIZE + i + WSIZE, y*DWSIZE + j + WSIZE, getColor2(imFcos, x*DWSIZE + i + WSIZE, y*DWSIZE + j + WSIZE)*iranyszorzo*sugarszorzo);
			setColor2(imFsin, x*DWSIZE + i + WSIZE, y*DWSIZE + j + WSIZE, getColor2(imFsin, x*DWSIZE + i + WSIZE, y*DWSIZE + j + WSIZE)*iranyszorzo*sugarszorzo);

		}

		CvRect roi = cvRect(x*DWSIZE, y*DWSIZE, DWSIZE, DWSIZE);
		cvSetImageROI(imBigEnhanced, roi);
		cvSetImageROI(imFcos, roi);
		cvSetImageROI(imFsin, roi);


		for (i = 0; i<DWSIZE; i++) for (j = 0; j<DWSIZE; j++)
		{
			cvSetImageROI(imCos, cvRect(i*DWSIZE, j*DWSIZE, DWSIZE, DWSIZE));
			cvSetImageROI(imSin, cvRect(i*DWSIZE, j*DWSIZE, DWSIZE, DWSIZE));
			cvAddWeighted(imCos, getColor2(imFcos, x*DWSIZE + i, y*DWSIZE + j), imSin, -getColor2(imFsin, x*DWSIZE + i, y*DWSIZE + j), 0, imAux);
			cvAdd(imBigEnhanced, imAux, imBigEnhanced);
		}

		cvSetImageROI(imBigEnhanced, cvRect(x*DWSIZE + WSIZE / 2, y*DWSIZE + WSIZE / 2, WSIZE, WSIZE));
		cvSetImageROI(imEnhanced, cvRect(x*(WSIZE), y*(WSIZE), WSIZE, WSIZE));

		if (mask[x][y]) cvCopy(imBigEnhanced, imEnhanced);
		else cvSet(imEnhanced, cvScalar(0.75, 0, 0, 0));

		cvResetImageROI(imBigEnhanced);
		cvResetImageROI(imEnhanced);
		cvResetImageROI(imFcos);
		cvResetImageROI(imFsin);

	}

	cvConvertScale(imEnhanced, imFinal, 255.0);

	//Show the enhanced image
	//-----------------------
	cvNamedWindow("imBigEnhanced", CV_WINDOW_AUTOSIZE);
	cvShowImage("imBigEnhanced", imBigEnhanced);
	cvSaveImage("imBigEnhanced.bmp", imBigEnhanced);
	cvWaitKey();

	//Show the final image
	//--------------------
	cvNamedWindow("imFinal", CV_WINDOW_AUTOSIZE);
	cvShowImage("imFinal", imFinal);
	cvSaveImage("imFinal.bmp", imFinal);
	cvWaitKey();

	//Create the output image
	//-----------------------
	cvAddWeighted(imFinal, 50, imFinal, 0, 0, imFinal);
	imOutput = cvCloneImage(imFinal);
	cvSmooth(imFinal, imOutput, CV_MEDIAN, 3);
	cvCopy(imOutput, imFinal);

	cvCmpS(imOutput, 127, imOutput, CV_CMP_GT);

	for (x = 0; x<bigX; x++) for (y = 0; y<bigY; y++) if (!mask[x][y])
	{
		cvSetImageROI(imOutput, cvRect(x*(WSIZE), y*(WSIZE), WSIZE, WSIZE));
		cvSet(imOutput, cvScalar(192, 0, 0, 0));
	}

	cvResetImageROI(imOutput);

	//Show the output image
	//---------------------
	cvNamedWindow("imOutput", CV_WINDOW_AUTOSIZE);
	cvShowImage("imOutput", imOutput);
	cvSaveImage("output.bmp", imOutput);
	cvWaitKey();


	//Open and Read the Image
	//----------------------- 
	cv::Mat imInput = cv::imread("output.bmp", CV_LOAD_IMAGE_COLOR);
	cv::imshow("The Fingerprint",imInput);
	cv::waitKey();

	//Remove Noise by Blurring with a Gaussian Filter
	//-----------------------------------------------
	cv::Mat img_filter;
	cv::GaussianBlur(imInput, img_filter, Size(3, 3), 0, 0, BORDER_DEFAULT);
	
	//Grayscale matrix
	//----------------
	cv::Mat grayscaleMat(img_filter.size(), CV_8U);

	//Convert BGR to Gray
	//-------------------
	cv::cvtColor(img_filter, grayscaleMat, CV_BGR2GRAY);
	
	//Equalize the histogram
	//----------------------
	cv::Mat img_hist_equalized;
	cv::equalizeHist(grayscaleMat, img_hist_equalized);
	
	//Binary image
	//------------
	cv::Mat binaryMat(img_hist_equalized.size(), img_hist_equalized.type());

	//Apply thresholding
	//------------------
	cv::threshold(img_hist_equalized, binaryMat, 100, 255, cv::THRESH_BINARY);

	//Save image
	//----------
	cv::imwrite("binary.bmp", binaryMat);
	
	//Load the saved binary image
	//---------------------------
	IplImage *im_binary_input, *im_binary_input_clone, *im_clean;
	im_binary_input = cvLoadImage("binary.bmp", 0);

	//Clone and show the saved binary image
	//-------------------------------------
	im_binary_input_clone = cvCloneImage(im_binary_input);
	cvShowImage("The Fingerprint", im_binary_input);

	//Thinning the fingerprint image with Golay ABC L mask algorithm
	//--------------------------------------------------------------
	GolayL(im_binary_input, im_binary_input_clone);

	//Save the thinned image
	//-----------------------
	cvSaveImage("thinning.bmp", im_binary_input_clone);

	//The im_binary_input image will be the thinned image
	//----------------------------------------------------
	cvCopy(im_binary_input_clone, im_binary_input);

	//Find the ridge ending in fingerprint with Golay ABC E mask algorithm
	//--------------------------------------------------------------------
	GolayE(im_binary_input, im_binary_input_clone);

	//Copy the im_binary_input_clone image in the im_binary_input image
	//-----------------------------------------------------------------
	cvCopy(im_binary_input_clone, im_binary_input);

	//Make a ROI on the im_binary_input image and on the im_binary_input_clone image
	//------------------------------------------------------------------------------
	cvSet(im_binary_input_clone, cvScalar(255, 0, 0, 0));
	CvRect roi = cvRect(2, 2, im_binary_input_clone->width - 4, im_binary_input_clone->height - 4);
	cvSetImageROI(im_binary_input, roi);
	cvSetImageROI(im_binary_input_clone, roi);
	cvCopy(im_binary_input, im_binary_input_clone);
	cvResetImageROI(im_binary_input);
	cvResetImageROI(im_binary_input_clone);

	//Delete the thorns from to the binary fingerprint image
	//------------------------------------------------------
	int k, l, count;

	for (i = 0; i<nrErased[0]; i++)
	{
		thorns[i][0] = erasedPixels[0][i];
		thornLength[i] = 1;
	}

	nrThorns = nrErased[0];

	for (k = 1; k<maxCycle; k++)
	{
		count = 0;
		for (j = 0; j<nrErased[k]; j++)
		{
			bool found = false;
			for (i = 0; i<nrThorns; i++)
			{
				if (abso(erasedPixels[k][j].x - thorns[i][k - 1].x) <= 1 && abso(erasedPixels[k][j].y - thorns[i][k - 1].y) <= 1)
				{
					thorns[i][k] = erasedPixels[k][j];
					thornLength[i]++;
					found = true;
				}
			}
			if (!found) count++;
		}
		printf("Thorn report: %d %d\n", k, count);
	}

	for (i = 0; i<nrThorns; i++) if (thornLength[i] >= 5)
	{
		for (j = 0; j<thornLength[i]; j++)
		{
			SetValue(im_binary_input_clone, thorns[i][j].x, thorns[i][j].y, true);
		}
	}

	//Show the clean fingerprint image (anti thorn fingerprint image)
	//---------------------------------------------------------------
	cvShowImage("The Fingerprint", im_binary_input_clone);
	cvSaveImage("clean.bmp", im_binary_input_clone);

	//Open the anti thorn fingerprint image
	//-------------------------------------
	im_clean = cvLoadImage("clean.bmp", 1);

	for (i = 2; i<im_binary_input_clone->width - 2; i++) 
	for (j = 2; j<im_binary_input_clone->height - 2; j++)
	if (GetValue(im_binary_input_clone, i, j))
	{
		count = 0;
		for (k = i - 1; k <= i + 1; k++) 
		for (l = j - 1; l <= j + 1; l++)
		if ((k != i || l != j) && GetValue(im_binary_input_clone, k, l)) 
			count++;
		if (count == 1) 
			setRGB(im_clean, i, j, 0, 255, 0);
	}

	//Find the ridge bifurcation in fingerprint
	//-----------------------------------------
	GolayQ(im_binary_input_clone, im_clean);

	//Mark the minutiae points
	//(Red -> Ridge Bifurcation); (Green -> Ridge Ending); (Blue ->Margin of Fingerprint)
	//-----------------------------------------------------------------------------------
	Minutia minu;

	FILE* f;
	f = fopen("minutiae.min", "wb+");

	for (i = 2; i<im_clean->width - 2; i++) 
	for (j = 2; j<im_clean->height - 2; j++)
	if (isRed(im_clean, i, j) || isGreen(im_clean, i, j))
	{
		bool kill = false;

		if (i<WSIZE || i >= im_clean->width - WSIZE) 
			kill = true;
		if (j<WSIZE || j >= im_clean->height - WSIZE) 
			kill = true;
		
		if (!kill)
		{
			if (isGray(im_clean, i + WSIZE, j))	
				kill = true;
			if (isGray(im_clean, i - WSIZE, j))	
				kill = true;
			if (isGray(im_clean, i, j + WSIZE))	
				kill = true;
			if (isGray(im_clean, i, j - WSIZE))	
				kill = true;
		}

		if (kill) 
			setRGB(im_clean, i, j, 0, 0, 255);
		else
		{
			minu.event = isRed(im_clean, i, j) ? 1 : 2;
			minu.x = (double)i;
			minu.y = (double)j;
			fwrite(&minu, 1, sizeof(minu), f);
		}
	}

	fclose(f);

	//Save and show the clean fingerprint image with minutiae
	//-------------------------------------------------------
	cvSaveImage("minutiae.bmp", im_clean);
	cvShowImage("The Fingerprint", im_clean);

	
	cv::waitKey();
	return 0;
	
}