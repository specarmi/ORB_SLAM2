#include <iostream>
#include <vector>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/persistence.hpp>

#include "ORBVocabulary.h"

using namespace DBoW2;
using namespace std;
using namespace ORB_SLAM2;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(vector<vector<cv::Mat > > &features, const string &feature_path);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 5;

// ----------------------------------------------------------------------------

int main(int argc, char **argv)
{
  if(argc != 3)
  {
      cerr << endl << "Usage: ./build_superpt_vocab path_to_superpoint_features "
        "path_to_superpoint_vocabulary" << endl;
      return 1;
  }
  
  // load the vocabulary from disk
  ORBVocabulary* voc;
  voc = new ORBVocabulary();
  bool vocload = voc->loadFromTextFile(argv[2]);
  if(!vocload)
  {
    cout << "Failed to load vocabulary" << endl;
    return 1;
  }

  // get features for a small number of images
  vector<vector<cv::Mat > > features;
  loadFeatures(features, argv[1]);

  // score images to test vocab
  cout << "Scoring image matches (0 low, 1 high): " << endl;
  cout << "Images from FLIR ADAS videos frames: 1, 2, 15, 40, 4224" << endl;
  BowVector v1, v2;
  for(int i = 0; i < NIMAGES; i++)
  {
    voc->transform(features[i], v1);
    for(int j = 0; j < NIMAGES; j++)
    {
      voc->transform(features[j], v2);
      
      double score = voc->score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<cv::Mat > > &features, const string &feature_path)
{
  features.clear();
  features.reserve(NIMAGES);

  cout << "Importing SuperPoint features..." << endl;
  for(int i = 1; i < NIMAGES + 1; ++i)
  {
    stringstream ss;
    ss << feature_path << i << ".yaml";
    cv::FileStorage pts_log(ss.str(), cv::FileStorage::READ);

    cv::Mat descriptors;
    pts_log["descriptors"] >> descriptors;

    features.push_back(vector<cv::Mat >());
    changeStructure(descriptors, features.back());
  }
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}