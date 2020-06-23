
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void
clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor,
                    cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT) {
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1) {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2) {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt)) {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1) {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);

        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait) {
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1) {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2) {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int) it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i) {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 0);
    cv::imshow(windowName, topviewImg);

    if (bWait) {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev,
                              std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches) {

    double mean_distance = 0.0;
    vector<cv::DMatch> inside_matches;
    for (auto match: kptMatches) {
        cv::KeyPoint currPoints = kptsCurr[match.trainIdx];
        if (boundingBox.roi.contains(currPoints.pt)) {
            inside_matches.push_back(match);
        }
    }
//If the Dmatch distance between the descriptors is less, then their similarity is high.
// If the Dmatch distance between the descriptors is more, then their similarity is low.
// Using mean of Dmatch distance of all matches as threshold to filter matches which have low similarity.
    for (auto inside_match:inside_matches) {
        mean_distance += inside_match.distance;
    }
    if (inside_matches.size() > 0) {
        mean_distance = mean_distance / inside_matches.size();
    } else {
        return;
    }
    //
    for (auto inside_match:inside_matches) {
        if (inside_match.distance < mean_distance) {
            boundingBox.kptMatches.push_back(inside_match);
        }
    }

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg) {
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1) { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0) {
        TTC = NAN;
        return;
    }
    std::sort(distRatios.begin(), distRatios.end());
    int distRatioSize = distRatios.size();
    double medianDistRatio =
            distRatioSize % 2 == 0 ? (distRatios[distRatioSize / 2 - 1] + distRatios[distRatioSize / 2]) / 2
                                   : distRatios[distRatioSize / 2];

    // Finally, calculate a TTC estimate based on these 2D camera features
    TTC = (-1.0 / frameRate) / (1 - medianDistRatio);

}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images(constant acceleration model)
void computeTTCCameraCAM(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC,double &TmPrev, cv::Mat *visImg) {
    cout << "TmPrev: " <<TmPrev<<endl;
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1) { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0) {
        TTC = NAN;
        return;
    }
    std::sort(distRatios.begin(), distRatios.end());
    int distRatioSize = distRatios.size();
    double medianDistRatio =
            distRatioSize % 2 == 0 ? (distRatios[distRatioSize / 2 - 1] + distRatios[distRatioSize / 2]) / 2
                                   : distRatios[distRatioSize / 2];

    // Finally, calculate a TTC estimate based on these 2D camera features
    // Constant acceleration model
    double Tm=(1/frameRate)/(medianDistRatio-1);
    double C=(TmPrev-Tm)/(1/frameRate)+1;
    double C2=1+2*C;
    if(C2<0){
        cout << "Error of CAM based camera TTC calculation"<<endl;
    }
    else{
        TmPrev = Tm;
        TTC =  Tm*(1-sqrt(1+2*C))/C;
        cout << "C: " <<C<< "  Tm: " <<Tm<<"  TTC_CAM: "<< TTC <<endl;
    }
    }


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC) {
    double dT = 1 / frameRate;
    double laneWidth = 4.0;
    vector<double> lidarPointsCurrX, lidarPointsPrevX;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it) {
        if (abs(it->y) <= laneWidth / 2.0) {
            lidarPointsPrevX.push_back(it->x);
        }
    }
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it) {

        if (abs(it->y) <= laneWidth / 2.0) {
            lidarPointsCurrX.push_back(it->x);
        }
    }
    bool is_median = false;
    if (is_median) {
        // calculate median value
        sort(lidarPointsCurrX.begin(), lidarPointsCurrX.end());
        sort(lidarPointsPrevX.begin(), lidarPointsPrevX.end());
        int lidarPtCurrSize = lidarPointsCurrX.size();
        int lidarPtPrevSize = lidarPointsPrevX.size();

        double d1 = lidarPtCurrSize % 2 == 0 ?
                    (lidarPointsCurrX[lidarPtCurrSize / 2 - 1] + lidarPointsCurrX[lidarPtCurrSize / 2]) / 2
                                             : lidarPointsCurrX[lidarPtCurrSize / 2];
        double d0 = lidarPtPrevSize % 2 == 0 ?
                    (lidarPointsPrevX[lidarPtPrevSize / 2 - 1] + lidarPointsPrevX[lidarPtPrevSize / 2]) / 2
                                             : lidarPointsPrevX[lidarPtPrevSize / 2];
        TTC = d1 * dT / (d0 - d1);
    } else {
        // 6 sigma principle
        double XCurrSum = accumulate(lidarPointsCurrX.begin(), lidarPointsCurrX.end(), 0.0);
        double XPrevSum = accumulate(lidarPointsPrevX.begin(), lidarPointsPrevX.end(), 0.0);
        double CurrMean = XCurrSum / lidarPointsCurrX.size();
        double PrevMean = XPrevSum / lidarPointsPrevX.size();
        double Curraccum = 0.0;
        double Prevaccum = 0.0;
        for_each(begin(lidarPointsCurrX), std::end(lidarPointsCurrX), [&](const double d) {
            Curraccum += (d - XCurrSum) * (d - XCurrSum);
        });

        double CurrStd = sqrt(Curraccum / (lidarPointsCurrX.size() - 1));

        for_each(begin(lidarPointsPrevX), std::end(lidarPointsPrevX), [&](const double d) {
            Prevaccum += (d - XPrevSum) * (d - XPrevSum);
        });

        double PrevStd = sqrt(Prevaccum / (lidarPointsPrevX.size() - 1));
        int CurrCount = 0;
        int PrevCount = 0;
        double CurrAns = 0;
        double PrevAns = 0;
        for (int i = 0; i < lidarPointsCurrX.size(); ++i) {
            if (abs(lidarPointsCurrX[i] - CurrMean) < 3 * CurrStd) {
                CurrAns += lidarPointsCurrX[i];
                ++CurrCount;
            }
        }

        for (int i = 0; i < lidarPointsPrevX.size(); ++i) {
            if (abs(lidarPointsPrevX[i] - PrevMean) < 3 * PrevStd) {
                PrevAns += lidarPointsPrevX[i];
                ++PrevCount;
            }
        }

        double d1 = CurrAns / CurrCount;
        double d0 = PrevAns / PrevCount;
        TTC = d1 * dT / (d0 - d1);
    }

}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame,
                        DataFrame &currFrame) {
    multimap<int, int> boxmap{};
    for (auto match:matches) {
        cv::KeyPoint prevPoints = prevFrame.keypoints[match.queryIdx];
        cv::KeyPoint currPoints = currFrame.keypoints[match.trainIdx];
        int prevBoxId = -1;
        int currBoxId = -1;

        for (auto box:prevFrame.boundingBoxes) {
            if (box.roi.contains(prevPoints.pt)) prevBoxId = box.boxID;
        }
        for (auto box:currFrame.boundingBoxes) {
            if (box.roi.contains(currPoints.pt)) currBoxId = box.boxID;
        }
        // generate currBoxId-prevBoxId map pair
        boxmap.insert({currBoxId, prevBoxId});

    }
    int CurrBoxSize = currFrame.boundingBoxes.size();
    int prevBoxSize = prevFrame.boundingBoxes.size();
    // find the best matched previous boundingbox for each current boudingbox
    for (int i = 0; i < CurrBoxSize; ++i) {
        auto boxmapPair = boxmap.equal_range(i);
        vector<int> currBoxCount(prevBoxSize, 0);
        for (auto pr = boxmapPair.first; pr != boxmapPair.second; ++pr) {
            if (-1 != (*pr).second) currBoxCount[(*pr).second] += 1;
        }
        // find the position of best prev box which has highest number of keypoint correspondences.
        int maxPosition = std::distance(currBoxCount.begin(),
                                        std::max_element(currBoxCount.begin(), currBoxCount.end()));
        bbBestMatches.insert({maxPosition, i});
        cout<<"Current BoxID: "<<i<<" match Previous BoxID: "<<maxPosition<<endl;
    }

}
