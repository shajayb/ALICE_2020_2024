class landValueMap
{
public:
    float lv[RES][RES];
    zVector lvec[RES][RES];
    std::vector<zVector> lastShortestPath;

    void updateLVMap(const std::vector<Employment_center>& ECenters)
    {
        for (int i = 0; i < RES; i++)
        {
            for (int j = 0; j < RES; j++)
            {
                zVector pt(i, j, 0);
                float weightedSum = 0.0f;
                float totalWeight = 0.0f;
                for (const auto& EC : ECenters)
                {
                    zVector cen = EC.cen;
                    float distance = pt.distanceTo(cen);
                    float weight = (distance > 0.0f) ? 1.0f / distance : 1.0f;
                    weightedSum += weight * distance;
                    totalWeight += weight;
                }
                lv[i][j] = (totalWeight > 0.0f) ? weightedSum / totalWeight : 0.0f;
            }
        }
    }

    void computeVectorField()
    {
        for (int i = 1; i < RES - 1; i++)
        {
            for (int j = 1; j < RES - 1; j++)
            {
                float dx = (lv[i + 1][j] - lv[i - 1][j]) * 0.5f;
                float dy = (lv[i][j + 1] - lv[i][j - 1]) * 0.5f;
                lvec[i][j] = zVector(dx, dy, 0);
            }
        }
    }

    void findShortestPath(zVector start, zVector end)
    {
        lastShortestPath.clear();

        auto heuristic = [](int x1, int y1, int x2, int y2) {
            return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
            };

        int sx = std::round(start.x), sy = std::round(start.y);
        int ex = std::round(end.x), ey = std::round(end.y);

        std::priority_queue<Node*, std::vector<Node*>, CompareNode> openSet;
        std::unordered_map<int, Node*> visited;

        Node* startNode = new Node(sx, sy, 0, heuristic(sx, sy, ex, ey), nullptr);
        openSet.push(startNode);

        int directions[4][2] = { {0,1}, {1,0}, {0,-1}, {-1,0} };

        while (!openSet.empty()) {
            Node* current = openSet.top();
            openSet.pop();

            if (current->x == ex && current->y == ey) {
                while (current) {
                    lastShortestPath.push_back(zVector(current->x, current->y, 0));
                    current = current->parent;
                }
                std::reverse(lastShortestPath.begin(), lastShortestPath.end());
                return;
            }

            int key = current->x * RES + current->y;
            if (visited.count(key)) continue;
            visited[key] = current;

            for (auto& dir : directions) {
                int nx = current->x + dir[0];
                int ny = current->y + dir[1];

                if (nx >= 0 && nx < RES && ny >= 0 && ny < RES) {
                    float newCost = current->cost + lv[nx][ny] + 1;
                    openSet.push(new Node(nx, ny, newCost, heuristic(nx, ny, ex, ey), current));
                }
            }
        }


    }

    void smoothPath()
    {
        if (lastShortestPath.size() < 3) return; // No smoothing if too few points

        std::vector<zVector> smoothedPath = lastShortestPath;

        for (size_t i = 1; i < lastShortestPath.size() - 1; i++)
        {
            smoothedPath[i] = lastShortestPath[i - 1] * 0.3f + lastShortestPath[i] * 0.4f + lastShortestPath[i + 1] * 0.3f;
        }

        lastShortestPath = smoothedPath;
    }

    void drawPth(vector<zVector>& path)
    {
        if (!path.empty())
        {
            glColor3f(0, 0, 1);
            for (size_t i = 0; i < path.size() - 1; i++)
            {
                drawLine(zVecToAliceVec(path[i]), zVecToAliceVec(path[i + 1]));
            }
        }
    }

    void processTriangle(zVector pts[4], float values[4], float thresholdValue, std::vector<std::pair<zVector, zVector>>& contour)
    {
        zVector newPts[4];
        bool addedPtOnEdge[3] = { false, false, false };
        int edgeNum = 0, numNewPt = 0;

        for (int nv = 0; nv < 3; nv++)
        {
            int curPt = nv, nextPt = nv + 1;
            float val = (thresholdValue - values[curPt]) / (values[nextPt] - values[curPt]);
            if (val > 0 && val < 1)
            {
                newPts[edgeNum] = pts[curPt] + (pts[nextPt] - pts[curPt]) * val;
                addedPtOnEdge[edgeNum++] = true;
                numNewPt++;
            }
        }

        newPts[3] = newPts[0];
        for (int n = 0; n < 3; n++)
        {
            if (addedPtOnEdge[n] && addedPtOnEdge[n + 1])
            {
                contour.push_back({ newPts[n], newPts[n + 1] });
            }
        }
    }

    void processAllTriangles(float thresholdValue, std::vector<std::pair<zVector, zVector>>& contours)
    {
        for (int i = 0; i < 49; i++)
        {
            for (int j = 0; j < 49; j++)
            {
                zVector pts[4] = { zVector(i, j, 0), zVector(i, j + 1, 0), zVector(i + 1, j, 0), zVector(i + 1, j + 1, 0) };
                float values[4] = { lv[i][j], lv[i][j + 1], lv[i + 1][j], lv[i + 1][j + 1] };
                processTriangle(pts, values, thresholdValue, contours);
            }
        }
    }

    void drawIsoline(float thresholdValue)
    {
        isolines.clear();
        processAllTriangles(thresholdValue, isolines);

        glColor3f(1, 1, 0); // Yellow isoline
        for (const auto& segment : isolines)
        {
            drawLine(zVecToAliceVec(segment.first), zVecToAliceVec(segment.second));
        }
    }

    void draw()
    {
        if (!lastShortestPath.empty())
        {
            glColor3f(0, 0, 1);
            for (size_t i = 0; i < lastShortestPath.size() - 1; i++)
            {
                drawLine(zVecToAliceVec(lastShortestPath[i]), zVecToAliceVec(lastShortestPath[i + 1]));
            }
        }

        glColor3f(0, 0, 0);
        for (int i = 1; i < RES; i++)
        {
            for (int j = 1; j < RES; j++)
            {

                zVector pt, dir;
                pt = zVector(i, j, 0);
                dir = lvec[i][j];
                dir.normalize();

                drawLine(zVecToAliceVec(pt), zVecToAliceVec(pt + dir));


                //
                // zPoint pts[4] = { my_grid_pts[i][j], my_grid_pts[i][j + 1], my_grid_pts[i + 1][j], my_grid_pts[i][j] };
                //float values[4] = { my_grid_values[i][j], my_grid_values[i][j + 1], my_grid_values[i + 1][j], my_grid_values[i][j] };
                // drawPt()
                zPoint pts[4] = { zVector(i, j, 0), zVector(i, j + 1, 0), zVector(i + 1, j, 0), zVector(i, j, 0) };
                float values[4] = { lv[i][j], lv[i][j + 1],  lv[i + 1][j], lv[i][j] };


                processTriangle(pts, values, 0.5);
            }
        }
    }
};




class landValueMap
{
public:
    float lv[50][50];
    zVector vf[50][50];
    std::vector<zVector> path;
    std::vector<std::pair<zVector, zVector>> isolines;

    void smoothPath()
    {
        if (path.size() < 3) return;
        std::vector<zVector> smoothedPath = path;
        for (size_t i = 1; i < path.size() - 1; ++i)
        {
            smoothedPath[i] = path[i - 1] * 0.3f + path[i] * 0.4f + path[i + 1] * 0.3f;
        }
        path = smoothedPath;
    }

    void processTriangle(zVector pts[4], float values[4], float thresholdValue, std::vector<std::pair<zVector, zVector>>& contour)
    {
        zVector newPts[4];
        bool addedPtOnEdge[3] = { false, false, false };
        int edgeNum = 0, numNewPt = 0;

        for (int nv = 0; nv < 3; nv++)
        {
            int curPt = nv, nextPt = nv + 1;
            float val = (thresholdValue - values[curPt]) / (values[nextPt] - values[curPt]);
            if (val > 0 && val < 1)
            {
                newPts[edgeNum] = pts[curPt] + (pts[nextPt] - pts[curPt]) * val;
                addedPtOnEdge[edgeNum++] = true;
                numNewPt++;
            }
        }

        newPts[3] = newPts[0];
        for (int n = 0; n < 3; n++)
        {
            if (addedPtOnEdge[n] && addedPtOnEdge[n + 1])
            {
                contour.push_back({ newPts[n], newPts[n + 1] });
            }
        }
    }

    void processAllTriangles(float thresholdValue, std::vector<std::pair<zVector, zVector>>& contours)
    {
        for (int i = 0; i < 49; i++)
        {
            for (int j = 0; j < 49; j++)
            {
                zVector pts[4] = { zVector(i, j, 0), zVector(i, j + 1, 0), zVector(i + 1, j, 0), zVector(i + 1, j + 1, 0) };
                float values[4] = { lv[i][j], lv[i][j + 1], lv[i + 1][j], lv[i + 1][j + 1] };
                processTriangle(pts, values, thresholdValue, contours);
            }
        }
    }

    void drawIsoline(float thresholdValue)
    {
        isolines.clear();
        processAllTriangles(thresholdValue, isolines);

        glColor3f(1, 1, 0); // Yellow isoline
        for (const auto& segment : isolines)
        {
            drawLine(zVecToAliceVec(segment.first), zVecToAliceVec(segment.second));
        }
    }

    void draw()
    {
        for (auto& pt : path)
        {
            glColor3f(0, 1, 0);
            glPointSize(4);
            drawPoint(zVecToAliceVec(pt));
        }
    }
};

