#define _MAIN_
#ifdef _MAIN_

#include "main.h"

//// zSpace Core Headers
#include <headers/zApp/include/zObjects.h>
#include <headers/zApp/include/zFnSets.h>
#include <headers/zApp/include/zViewer.h>
#include <vector>
#include <limits>
#include <cmath>

using namespace zSpace;



void setup()
{
}

void update(int value)
{
    // Placeholder for future update logic
}

void draw()
{
    backGround(0.9);
    drawGrid(50);
  
}

void keyPress(unsigned char k, int xm, int ym)
{
   
}

void mousePress(int b, int state, int x, int y)
{
    // Placeholder for future mouse press interactions
}

void mouseMotion(int x, int y)
{
    // Placeholder for future motion-based interactions
}

#endif // _MAIN_