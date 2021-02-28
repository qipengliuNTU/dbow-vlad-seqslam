#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{

    string voc_path = argv[1];
    string image_folder = argv[2];
    string output_file = argv[3];

    vector<cv::String> images_path;
    glob(image_folder + "/*", images_path, false);

    cout << "reading database" << endl;
    DBoW3::Vocabulary vocab(voc_path);
    if (vocab.empty())
    {
        cerr << "Vocabulary does not exist." << endl;
        return 1;
    }

    cout << "reading images... " << endl;
    vector<Mat> images;
    for (int i = 0; i < images_path.size(); i += 2)
    {
        images.push_back(imread(images_path[i]));
    }

    cout << "convert image to bow ... " << endl;
    Ptr<Feature2D> detector = ORB::create();
    vector<DBoW3::BowVector> bow_vecs;
    bow_vecs.reserve(images.size());
    for (Mat &image : images)
    {
        vector<KeyPoint> keypoints;
        Mat descriptor;
        detector->detectAndCompute(image, Mat(), keypoints, descriptor);
        DBoW3::BowVector v;
        vocab.transform(descriptor, v);
        bow_vecs.push_back(v);
    }

    cout << "generate distance matrix " << endl;
    ofstream of;
    of.open(output_file);
    if (of.fail())
    {
        std::cerr << "Failed to open output file " << output_file << std::endl;
        exit(1);
    }
    for (int i = 0; i < bow_vecs.size(); i++)
    {
        for (int j = 0; j < bow_vecs.size(); j++)
        {
            of << vocab.score(bow_vecs[i], bow_vecs[j]) << " ";
        }
        of << "\n";
    }

    of.close();
    cout << "Output done\n";
}