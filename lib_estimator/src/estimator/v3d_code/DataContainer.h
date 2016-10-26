#ifndef SFMDATA_DATACONTAINTER_H
#define SFMDATA_DATACONTAINTER_H

#ifdef USE_BIAS_WIN32_ONLY
#include "FundMatrixEstimator.h"
#endif

//#include <estimator/v3d_code/Simple2DContainerUT.hpp>
//#include <estimator/v3d_code/MatchingTask.h>
#include <estimator/v3d_code/ocv_arrsac_E_5pts.h>
//#include <estimator/v3d_code/ocv_arrsac_F_7pts.h>
//#include <estimator/v3d_code/ocv_arrsac_homography.h>
#include <estimator/v3d_code/Matcher.h>

namespace sfmData {

  typedef struct {
    //matcher::MatcherWrapper matcherWrapper;
	
#ifdef USE_BIAS_WIN32_ONLY	
    FundMatrixEstimator F_Estimator;
    EssentialMatrixEstimator E_Estimator;
#endif
		
    
    ocv_lib::ARRSAC_essential E_Estimator_ocv;
    //ocv_lib::ARRSAC_fundamental F_Estimator_ocv;
    //ocv_lib::ARRSAC_homography H_Estimator_ocv;

		std::vector<std::pair<int, int> > matches;
		std::vector<std::pair<int, int> > inlier_matches;
		//std::vector<std::pair<int, int> > inlier_matches_H;
    //featSet featureSet;
		
		std::vector<double> x1;
		std::vector<double> x2;
		std::vector<double> y1;
		std::vector<double> y2;
		
		//N2DC::Simple2DContainerUT<int> match_matrix;
		
		matcher::MatcherContainer matcherContainer;

    // NOTE: jheinly - this was added to enable the printing of the indices of the images
    //                 where a watermark matching was detected
    size_t imageIndex1;
    size_t imageIndex2;
	} DataContainer;
}


#endif //SFMDATA_DATACONTAINTER_H
