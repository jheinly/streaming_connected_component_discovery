#include <estimator/v3d_code/ocv_arrsac.h>

//using namespace ocv_lib;

void ocv_lib::ARRSAC::setInlierThreshold(double inlierThreshold)
{
  m_inlierThreshold = inlierThreshold;   
}

void ocv_lib::ARRSAC::init(size_t minSampleSize, double inlierThreshold, size_t maxHypotheses, 
                           size_t maxSolutionsPerSample, size_t blockSize)
{
  // initialize common parameters
  m_minSampleSize = minSampleSize;  
  setInlierThreshold(inlierThreshold);
  m_maxHypotheses = maxHypotheses;	
  m_maxSolutionsPerSample= maxSolutionsPerSample; 
  m_blockSize = blockSize;
  m_confThreshold = 0.95;

  // initialize data structures
  m_sample.resize(m_minSampleSize);
}


size_t ocv_lib::ARRSAC::solveMaster(cv::Mat& solution, 
                                    std::vector<bool>& inliers)
{
  inliers.clear();

  // initialize sprt parameters
  m_SPRT_tM = 200;//20
  m_SPRT_mS = 4;//1;
  m_SPRT_delta = 0.05;
  m_SPRT_epsilon = 0.01;//0.1;

  const int max_num_degenerate_sample_tries = 100;

  // initialize data structures
  std::vector<cv::Mat> multisoln_table; 
  std::vector<cv::Mat> mainHypSet;
  std::vector<int> mainInlierSet;
  std::vector<size_t> which_samples; which_samples.resize(m_minSampleSize);

  // design the first sprt test using the default values
  double An = designSPRTTest(); //todo make this a call done before.

  // initialize local variables
  size_t hypCount = 0; 
  bool good_flag = false;
  int numSolns = 0;
  int ninliers = 0;
  size_t numPointsTested = 0;
  size_t solution_index = 0;
  cv::Mat* active_solution_ptr = NULL;
  int best_inlier_score = 0;
  unsigned int num_tries = 0;

  // backup solution in case all solutions are rejected by the sprt
  cv::Mat backupSolution;
  int backupInliers = 0;

  // process data in rounds
  int roundNum = 0;  // first round is 0th
  /*int dataBegin = roundNum * m_blockSize; //not implemented 
  int dataEnd = min(dataBegin + m_blockSize, m_numDataPoints); //not implemented*/
  size_t numHypInitialSet = m_maxHypotheses;

  while (hypCount < numHypInitialSet)
  {
    if (generateRandomSample(which_samples))
    {
      numSolns = getSampleSolutions(which_samples, multisoln_table);
      // check to handle the case where the points generate only 'bad' models
      if (numSolns < 1)
      {
        ++num_tries;
        if (num_tries > max_num_degenerate_sample_tries) {
          return 0;
        }
      }
      else
      {
        num_tries = 0;
      }

      for(solution_index = 0; solution_index < static_cast<size_t>(numSolns); solution_index++)
      {
        if (hypCount == m_maxHypotheses)
          break;

        hypCount = hypCount + 1;

        active_solution_ptr = &multisoln_table[solution_index];

        // evaluate solution with sprt (set last argument to 1)
        good_flag = evaluateSolution(*active_solution_ptr, 
          An, ninliers, numPointsTested, 1);

        if (ninliers > backupInliers)
        {
          backupInliers = ninliers;
          backupSolution = *active_solution_ptr;
        }

        if (good_flag == 0)   // Model rejected
        {
          double deltaNew = static_cast<double>(ninliers) / static_cast<double>(numPointsTested);   // CHECK: divide by npts?
          if ((deltaNew - m_SPRT_delta)/m_SPRT_delta > 0.05)      // Only change if delta increases!
          {
            m_SPRT_delta = deltaNew;
            An = designSPRTTest();
          }
        }

        else    // Model accepted
        {
          // push back only if model accepted
          mainHypSet.push_back(multisoln_table[solution_index]);
          mainInlierSet.push_back(ninliers);

          if (ninliers > best_inlier_score)   // ...and best support so far
          {
            best_inlier_score = ninliers;
            m_SPRT_epsilon = double(ninliers)/numPointsTested;
            An = designSPRTTest();

            double pNoInliers = 1 -  pow(m_SPRT_epsilon,int(m_minSampleSize));
            double eps = pow(10.0,-15.0);
            pNoInliers = std::max(eps, pNoInliers);      // Avoid division by -Inf
            pNoInliers = std::min(1-eps, pNoInliers);    // Avoid division by 0.
            numHypInitialSet = (unsigned int)std::min(double(m_maxHypotheses), ceil(log(1-m_confThreshold)/log(pNoInliers)));
          }
        } 

      } // end evaluation of hypotheses for one sample
    } // end if - check here for error in generating solution?

  } // finished generating all hypotheses in the initial set


  // at this point, we have an initial set of hypotheses in mainHypSet, with corresponding
  // inlier counts in mainInlierSet

  //	unsigned int totalNumHypGenerated = hypCount; //not used PFG april 2010
  size_t numHypCurrent = mainHypSet.size();

  // check for no good solutions
  bool noGoodSolutions = false;
  if (numHypCurrent == 0)
  {
    noGoodSolutions = true;
  }

  size_t numHypPreemptive = m_maxHypotheses / 2;
  // Preemptive evaluation rounds begin

  roundNum = 1;
  bool repeatTesting = true;
  int numRounds = (int)ceil(double(m_numDataPoints)/m_blockSize);
  double temp_an = 0.0;
  size_t temp_pts_tested = 0;

  if (numRounds > 1 && !noGoodSolutions)
  {
    while (repeatTesting)
    {
      // Select data block for the current round
      //dataBegin = (roundNum) * m_blockSize; //not implemented
      //dataEnd = min(dataBegin + m_blockSize, m_numDataPoints); //not implemented

      for (solution_index = 0; solution_index < numHypCurrent; solution_index++)
      {
        active_solution_ptr = &mainHypSet[solution_index];

        evaluateSolution(*active_solution_ptr, temp_an, ninliers, temp_pts_tested, 0);
        mainInlierSet[solution_index]+=ninliers;
      }

      // Instead of sorting, get rid of hypotheses that rank below the median inlier score
      // Does this need to be made more efficient?
      std::vector<cv::Mat>::iterator itHyp;
      std::vector<int>::iterator itInliers;



      std::vector<int> tempInlierSet(mainInlierSet);
      std::sort(tempInlierSet.begin(), tempInlierSet.end());

      int thresh = tempInlierSet[tempInlierSet.size() / 2]; //med.GetMedian();

      itHyp = mainHypSet.begin();
      for (itInliers=mainInlierSet.begin(); itInliers!=mainInlierSet.end();)
      {
        if (*itInliers < thresh)
        {
          itInliers = mainInlierSet.erase(itInliers);
          itHyp = mainHypSet.erase(itHyp);
        } 
        else 
        {
          itInliers++;
          itHyp++;
        }
      }

      roundNum++;
      if (roundNum < numRounds)
      {
        numHypCurrent = mainHypSet.size();
        numHypPreemptive = numHypPreemptive / 2 +  numHypPreemptive % 2; //use to be ceil which should do the trick PFG April 2010
        numHypCurrent = std::min(numHypPreemptive, numHypCurrent);

        if (numHypCurrent > 1)
          repeatTesting = 1;
        else
          repeatTesting = 0;
      }
      else
        repeatTesting = 0;

    } // Finished evaluation, we now have a valid solution

  } // end check for number of rounds = 1

  // Find best hypotheses
  if (!noGoodSolutions)
  {
    mainHypSet[0].copyTo(solution);
    best_inlier_score = mainInlierSet[0];
    for (solution_index = 0; solution_index < mainHypSet.size(); solution_index++)
    {
      if (mainInlierSet[solution_index] > best_inlier_score)
      {
        best_inlier_score = mainInlierSet[solution_index];
        mainHypSet[solution_index].copyTo(solution);
      }

    }
    ninliers = best_inlier_score;
  }
  else {
    ninliers = backupInliers;
    backupSolution.copyTo(solution);
  }
  // find all inliers
  if (ninliers)
    ninliers = findInliers(solution, inliers);
  /*else 
  std::cerr << "RANSAC did not find any good solution" << std::endl;*/
  return ninliers;
}

inline bool ocv_lib::ARRSAC::generateRandomSample(std::vector<size_t>& sample)
{
  unsigned int count=0;
  unsigned int index;
  std::vector<size_t>::iterator pos;
  pos = sample.begin();
  do {
    index = rand() % m_numDataPoints;
    if (find(sample.begin(), pos, index) == pos)
    {
      sample[count] = index;
      ++count;
      ++pos;
    }
  } while (count < m_minSampleSize);

  return true;
}

inline double ocv_lib::ARRSAC::designSPRTTest() 
{
  double An_1, C, K;

  C = (1 - m_SPRT_delta)*log( (1 - m_SPRT_delta)/(1-m_SPRT_epsilon) ) 
    + m_SPRT_delta*(log( m_SPRT_delta/m_SPRT_epsilon ));
  K = (m_SPRT_tM*C)/m_SPRT_mS + 1;
  An_1 = K;

  // compute A using a recursive relation
  // A* = lim(n->inf)(An), the series typically converges within 4 iterations
  for (unsigned int i = 0; i < 10; ++i)
  {
    double An = K + log(An_1);
    if (An - An_1 < 1.5e-8) 
    {
      break;
    }
    An_1 = An;
  }

  return An_1;
}
