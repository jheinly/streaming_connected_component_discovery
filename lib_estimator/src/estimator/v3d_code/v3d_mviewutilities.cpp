#include <estimator/v3d_code/v3d_linear_lu.h>
#include <estimator/v3d_code/v3d_mathutilities.h>
#include <estimator/v3d_code/v3d_optimization.h>
#include <estimator/v3d_code/v3d_mviewutilities.h>

#include <map>
#include <cstdlib>
#include <iostream>

namespace
{

   using namespace std;
   using namespace V3D;

   // Struct Link connects an image measurement with a TriangulatedPoint.
   struct Link
   {
         int id, view;
         int point; // Id of newly created TriangulatedPoint

         Link(PointMeasurement const& m)
            : id(m.id), view(m.view), point(-1)
         { }

         bool matches(PointMeasurement const& m) const
         {
            return id == m.id && view == m.view;
         }
   };

   inline Link&
   findMatchingLink(PointMeasurement const& m, vector<Link>& links)
   {
      for (unsigned i = 0; i < links.size(); ++i)
      {
         if (links[i].matches(m))
            return links[i];
      }

      std::cerr << "Internal error: cannot find matching link!" << std::endl;
      exit(-1);
   }

   struct CorrespondenceLess
   {
         bool operator()(PointCorrespondence const& m1,
                         PointCorrespondence const& m2) const
         {
            if (m1.left.view < m2.left.view) return true;
            if (m1.left.view > m2.left.view) return false;

            return m1.right.view < m2.right.view;
         }
   };

} // end namespace <>

namespace V3D
{

   void 
   TriangulatedPoint::connectTracks(std::vector<PointCorrespondence> const& corrs_,
                                    std::vector<TriangulatedPoint>& points,
                                    size_t nRequiredMeasurements)
   {
      points.clear();
      if (corrs_.empty()) return;

      // Change the image correspondences, s.t. the left view has smaller id
      vector<PointCorrespondence> corrs(corrs_);
      for (unsigned i = 0; i < corrs.size(); ++i)
      {
         if (corrs[i].left.view > corrs[i].right.view)
            swap(corrs[i].left, corrs[i].right);
      }

      // Sort the correspondences, s.t. corresponding points with smaller left
      // view id come first
      sort(corrs.begin(), corrs.end(), CorrespondenceLess());

//    for (size_t i = 0; i < corrs.size(); ++i)
//    {
//       PointCorrespondence const& c = corrs[i];
//       cout << "left view = " << c.left.view << " left id = " << c.left.id
//            << "  right view = " << c.right.view << " right id = " << c.right.id << endl;
//    }

      // map from tight index [0..nViews) -> camera id
      map<int, int> cameraIdMap;
      for (unsigned i = 0; i < corrs.size(); ++i)
      {
         // Skip measurements with no valid view (camera) id
         if (corrs[i].left.view < 0 || corrs[i].right.view < 0) continue;

         if (cameraIdMap.find(corrs[i].left.view) == cameraIdMap.end())
            cameraIdMap.insert(make_pair(corrs[i].left.view, cameraIdMap.size()));

         if (cameraIdMap.find(corrs[i].right.view) == cameraIdMap.end())
            cameraIdMap.insert(make_pair(corrs[i].right.view, cameraIdMap.size()));
      }

      const unsigned int nViews = static_cast<unsigned int>(cameraIdMap.size());

      // Maps view id -> vector of associated Links
      vector<vector<Link> > distributedLinks(nViews);

      for (unsigned i = 0; i < corrs.size(); ++i)
      {
         PointCorrespondence const& corr = corrs[i];
         if (corr.left.view < 0 || corr.right.view < 0) continue;

         int l = cameraIdMap[corr.left.view];
         int r = cameraIdMap[corr.right.view];

         distributedLinks[l].push_back(Link(corr.left));
         distributedLinks[r].push_back(Link(corr.right));
      }

      vector<TriangulatedPoint> tmpPoints;

      vector<pair<int, int> > equivalences;

      for (unsigned i = 0; i < corrs.size(); ++i)
      {
         PointCorrespondence const& corr = corrs[i];

         if (corr.left.id < 0 || corr.right.id < 0) continue;
         if (corrs[i].left.view < 0 || corrs[i].right.view < 0) continue;

         int l = cameraIdMap[corr.left.view];
         int r = cameraIdMap[corr.right.view];

         Link& left  = findMatchingLink(corr.left, distributedLinks[l]);
         Link& right = findMatchingLink(corr.right, distributedLinks[r]);

         if (left.point < 0 && right.point < 0)
         {
            tmpPoints.push_back(TriangulatedPoint());
            tmpPoints.back().measurements.push_back(corr.left);
            tmpPoints.back().measurements.push_back(corr.right);
            left.point  = static_cast<int>(tmpPoints.size()) - 1;
            right.point = left.point;
         }
         else if (left.point >= 0 && right.point < 0)
         {
            tmpPoints[left.point].measurements.push_back(corr.right);
            right.point = left.point;
         }
         else if (left.point < 0 && right.point >= 0)
         {
            tmpPoints[right.point].measurements.push_back(corr.left);
            left.point = right.point;
         }
         else
         {
#if 1
            // This case indicates a loop in the view configuration.
            // Register equivalent triangulated points for now.
            if (left.point != right.point)
            {
               if (left.point < right.point)
                  equivalences.push_back(make_pair(left.point, right.point));
               else
                  equivalences.push_back(make_pair(right.point, left.point));
            }
#else
            VRException::vrerror("VRTriangulatedPoint::linkMeasurements()", __FILE__, __LINE__,
                                 "Internal error: need to unify two triangulated points!");
#endif
         }
      } // end for (i)

      // Handle registered equivalences.
      map<int, int> substitutions;
      for (unsigned i = 0; i < equivalences.size(); ++i)
      {
         pair<int, int> const& assoc = equivalences[i];
         // Note: assoc.second > assoc.first.
         int target = assoc.first;
         map<int, int>::const_iterator p = substitutions.find(assoc.second);
         // Follow the chain of substitutions until the end
         while (p != substitutions.end())
         {
            target = (*p).second;
            p = substitutions.find(target);
         }
         substitutions.insert(make_pair(assoc.second, target));
         // Copy the image measurements to the target TriangulatedPoint
         TriangulatedPoint const& src  = tmpPoints[assoc.second];
         TriangulatedPoint&       dest = tmpPoints[target];
         for (unsigned j = 0; j < src.measurements.size(); ++j)
            dest.measurements.push_back(src.measurements[j]);
      } // end for (i)

      for (unsigned i = 0; i < tmpPoints.size(); ++i)
      {
         // Ignore all points which are substituted (unified) and points with
         // too few views.
         if (substitutions.find(i) == substitutions.end() &&
             tmpPoints[i].measurements.size() >= nRequiredMeasurements)
         {
            points.push_back(tmpPoints[i]);
         }
      }
   } // end TriangulatedPoint::connectTracks()

   void
   computeConsistentRotations(int const nViews,
                              std::vector<Matrix3x3d> const& relativeRotations,
                              std::vector<std::pair<int, int> > const& viewPairs,
                              std::vector<Matrix3x3d>& rotations, int method)
   {
#if !defined(V3DLIB_ENABLE_ARPACK)
      if (method == V3D_CONSISTENT_ROTATION_METHOD_SPARSE_EIG)
         method = V3D_CONSISTENT_ROTATION_METHOD_EIG_ATA;
#endif

      int const nRelPoses = static_cast<int>(relativeRotations.size());

      rotations.resize(nViews);

      switch (method)
      {
         case V3D_CONSISTENT_ROTATION_METHOD_SVD:
         {
            Matrix<double> A(3*nRelPoses, 3*nViews, 0.0);
            Matrix3x3d I;
            makeIdentityMatrix(I);
            scaleMatrixIP(-1.0, I);

            for (int i = 0; i < nRelPoses; ++i)
            {
               int const view1 = viewPairs[i].first;
               int const view2 = viewPairs[i].second;

               Matrix3x3d const& Rrel = relativeRotations[i];

               copyMatrixSlice(Rrel, 0, 0, 3, 3, A, 3*i, 3*view1);
               copyMatrixSlice(I,    0, 0, 3, 3, A, 3*i, 3*view2);
            } // end for (i)

            SVD<double> svd(A);
            int const startColumn = A.num_cols()-3; // last columns of right sing. vec for SVD

            Matrix<double> const& V = svd.getV();

            for (int i = 0; i < nViews; ++i)
            {
               copyMatrixSlice(V, 3*i, startColumn, 3, 3, rotations[i], 0, 0);
               enforceRotationMatrix(rotations[i]);
            }
            break;
         }
         case V3D_CONSISTENT_ROTATION_METHOD_SVD_ATA:
         case V3D_CONSISTENT_ROTATION_METHOD_EIG_ATA:
         case V3D_CONSISTENT_ROTATION_METHOD_SPARSE_EIG:
         {
            vector<pair<int, int> > nzA;
            vector<double> valsA;
            nzA.reserve(12*nRelPoses);
            valsA.reserve(12*nRelPoses);

            for (int i = 0; i < nRelPoses; ++i)
            {
               int const view1 = viewPairs[i].first;
               int const view2 = viewPairs[i].second;

               Matrix3x3d const& Rrel = relativeRotations[i];

               nzA.push_back(make_pair(3*i+0, 3*view1+0)); valsA.push_back(Rrel[0][0]);
               nzA.push_back(make_pair(3*i+0, 3*view1+1)); valsA.push_back(Rrel[0][1]);
               nzA.push_back(make_pair(3*i+0, 3*view1+2)); valsA.push_back(Rrel[0][2]);
               nzA.push_back(make_pair(3*i+1, 3*view1+0)); valsA.push_back(Rrel[1][0]);
               nzA.push_back(make_pair(3*i+1, 3*view1+1)); valsA.push_back(Rrel[1][1]);
               nzA.push_back(make_pair(3*i+1, 3*view1+2)); valsA.push_back(Rrel[1][2]);
               nzA.push_back(make_pair(3*i+2, 3*view1+0)); valsA.push_back(Rrel[2][0]);
               nzA.push_back(make_pair(3*i+2, 3*view1+1)); valsA.push_back(Rrel[2][1]);
               nzA.push_back(make_pair(3*i+2, 3*view1+2)); valsA.push_back(Rrel[2][2]);

               nzA.push_back(make_pair(3*i+0, 3*view2+0)); valsA.push_back(-1.0);
               nzA.push_back(make_pair(3*i+1, 3*view2+1)); valsA.push_back(-1.0);
               nzA.push_back(make_pair(3*i+2, 3*view2+2)); valsA.push_back(-1.0);
            } // end for (i)

            CCS_Matrix<double> A(3*nRelPoses, 3*nViews, nzA, valsA);

            if (method == V3D_CONSISTENT_ROTATION_METHOD_SPARSE_EIG)
            {
#if defined(V3DLIB_ENABLE_ARPACK)
               Vector<double> sigma;
               Matrix<double> V;
               SparseSymmetricEigConfig cfg;
               cfg.maxArnoldiIterations = 100000;
               computeSparseSVD(A, V3D_ARPACK_SMALLEST_MAGNITUDE_EIGENVALUES, 3, sigma, V, cfg);
               //computeSparseSVD(A, V3D_ARPACK_SMALLEST_EIGENVALUES, 3, sigma, V, cfg);
               for (int i = 0; i < nViews; ++i)
               {
                  copyMatrixSlice(V, 3*i, 0, 3, 1, rotations[i], 0, 2);
                  copyMatrixSlice(V, 3*i, 1, 3, 1, rotations[i], 0, 1);
                  copyMatrixSlice(V, 3*i, 2, 3, 1, rotations[i], 0, 0);
               }
#endif
            }
            else
            {
               Matrix<double> AtA(3*nViews, 3*nViews);
               multiply_At_A_SparseDense(A, AtA);

               if (method == V3D_CONSISTENT_ROTATION_METHOD_SVD_ATA)
               {
                  SVD<double> svd(AtA);
                  int const startColumn = A.num_cols()-3; // last columns of right sing. vec for SVD
                  Matrix<double> const& V = svd.getV();
                  for (int i = 0; i < nViews; ++i)
                     copyMatrixSlice(V, 3*i, startColumn, 3, 3, rotations[i], 0, 0);
               }
               else
               {
                  Eigenvalue<double> svd(AtA);
                  int const startColumn = 0; // first columns of eigenvector matrix
                  Matrix<double> const& V = svd.getV();
                  for (int i = 0; i < nViews; ++i)
                     copyMatrixSlice(V, 3*i, startColumn, 3, 3, rotations[i], 0, 0);
               } // end if
            } // end if
            break;
         }
         default:
         {
            std::cerr << "Unknown method argument for computeConsistentRotations()." << std::endl;
            exit(-1);
         }
      } // end switch

      for (int i = 0; i < nViews; ++i)
         enforceRotationMatrix(rotations[i]);

      // Remove gauge freedom by setting R[0] = I.
      Matrix3x3d const R0t = rotations[0].transposed();
      for (int i = 0; i < nViews; ++i)
         rotations[i] = rotations[i] * R0t;

      // Note: it seems, that either all Rs have det(R)=1 or all have det(R)=-1.
      // Since we remove the gauge freedem by multiplying all rotations with R_0^t,
      // we always end up with det(R)=1 and any code to enforce a positive determinant
      // is not necessary.
   } // end computeConsistentRotations()

//**********************************************************************

   Matrix3x3d
   computeIntersectionCovariance(vector<Matrix3x4d> const& projections,
                                 vector<PointMeasurement> const& measurements,
                                 double sigma)
   {
      Matrix<double> Cp(3, 3, 0.0);
      Cp[0][0] = Cp[1][1] = sigma;
      Cp[2][2] = 0.0;

      int const N = static_cast<int>(measurements.size());

      Matrix<double> A(2*N, 4, 0.0);
      InlineMatrix<double, 2, 3> Sp;
      makeZeroMatrix(Sp);
      InlineMatrix<double, 2, 4> Sp_P;
      for (int i = 0; i < N; ++i)
      {
         Sp[0][1] = -1; Sp[0][2] = measurements[i].pos[1];
         Sp[1][0] =  1; Sp[1][2] = -measurements[i].pos[0];

         int const view = measurements[i].view;
         multiply_A_B(Sp, projections[view], Sp_P);

         A[2*i+0][0] = Sp_P[0][0]; A[2*i+0][1] = Sp_P[0][1]; A[2*i+0][2] = Sp_P[0][2]; A[2*i+0][3] = Sp_P[0][3];
         A[2*i+1][0] = Sp_P[1][0]; A[2*i+1][1] = Sp_P[1][1]; A[2*i+1][2] = Sp_P[1][2]; A[2*i+1][3] = Sp_P[1][3];
      } // end for (i)

      SVD<double> svd(A);

      Matrix<double> V;
      svd.getV(V);

      Vector4d X;
      X[0] = V[0][3]; X[1] = V[1][3]; X[2] = V[2][3]; X[3] = V[3][3];

      Vector3d P;
      Matrix<double> S(2, 3, 0.0);
      Matrix<double> B(2*N, 3*N, 0.0);

      for (int i = 0; i < N; ++i)
      {
         int const view = measurements[i].view;
         multiply_A_v(projections[view], X, P);
         P[0] /= P[2]; P[1] /= P[2]; P[2] = 1.0;

         S[0][1] = -P[2]; S[0][2] =  P[1];
         S[1][0] =  P[2]; S[1][2] = -P[0];

         B[2*i+0][3*i+0] = -S[0][0]; B[2*i+0][3*i+1] = -S[0][1]; B[2*i+0][3*i+2] = S[0][2];
         B[2*i+1][3*i+0] = -S[1][0]; B[2*i+1][3*i+1] = -S[1][1]; B[2*i+1][3*i+2] = S[1][2];
      } // end for (i)

      Matrix<double> C(3*N, 3*N, 0.0);
      for (int i = 0; i < N; ++i)
      {
         C[3*i+0][3*i+0] = Cp[0][0]; C[3*i+0][3*i+1] = Cp[0][1]; C[3*i+0][3*i+2] = Cp[0][2];
         C[3*i+1][3*i+0] = Cp[1][0]; C[3*i+1][3*i+1] = Cp[1][1]; C[3*i+1][3*i+2] = Cp[1][2];
         C[3*i+2][3*i+0] = Cp[2][0]; C[3*i+2][3*i+1] = Cp[2][1]; C[3*i+2][3*i+2] = Cp[2][2];
      } // end for (i)

      Matrix<double> B_C(2*N, 3*N);
      multiply_A_B(B, C, B_C);
      Matrix<double> T(2*N, 2*N);
      multiply_A_Bt(B_C, B, T);

      Matrix<double> Tinv;
      invertMatrix(T, Tinv);

      Matrix<double> NN(5, 5), N4(4, 4);
      Matrix<double> At_Tinv(4, 2*N);
      multiply_At_B(A, Tinv, At_Tinv);
      multiply_A_B(At_Tinv, A, N4);

      for (int r = 0; r < 4; ++r)
         for (int c = 0; c < 4; ++c)
            NN[r][c] = N4[r][c];

      NN[0][4] = NN[4][0] = X[0];
      NN[1][4] = NN[4][1] = X[1];
      NN[2][4] = NN[4][2] = X[2];
      NN[3][4] = NN[4][3] = X[3];
      NN[4][4] = 0.0;

      Matrix<double> Ninv(5, 5);
      invertMatrix(NN, Ninv);

      Matrix4x4d sigma_XX;
      for (int r = 0; r < 4; ++r)
         for (int c = 0; c < 4; ++c)
            sigma_XX[r][c] = Ninv[r][c];

      Matrix3x4d Je;
      makeZeroMatrix(Je);
      Je[0][0] = Je[1][1] = Je[2][2] = 1.0 / X[3];
      Je[0][3] = -X[0] / (X[3]*X[3]);
      Je[1][3] = -X[1] / (X[3]*X[3]);
      Je[2][3] = -X[2] / (X[3]*X[3]);

      Matrix3x3d sigma_X = Je * sigma_XX * Je.transposed();
      return sigma_X;
   }

   double
   computeIntersectionRoundness(vector<Matrix3x4d> const& projections,
                                vector<PointMeasurement> const& measurements,
                                double sigma)
   {
      Matrix3x3d const cov_X = computeIntersectionCovariance(projections, measurements, sigma);

      Matrix<double> sigma_X(3, 3);
      copyMatrix(cov_X, sigma_X);

      SVD<double> svd(sigma_X);
      Vector<double> S;
      svd.getSingularValues(S);
      return sqrt(S[2] / S[0]);
   }

} // end namespace V3D

//**********************************************************************

# if defined(V3DLIB_ENABLE_SUITESPARSE)
namespace
{

   struct SparseRotationOptimizer : public V3D::SparseLevenbergOptimizer
   {
         ~SparseRotationOptimizer() { }

         SparseRotationOptimizer(vector<Matrix3x3d>& Rs, vector<Matrix3x3d> const& Rsij, 
                                 vector<int> const& correspondingParam1,
                                 vector<int> const& correspondingParam2)
	   : SparseLevenbergOptimizer(9, Rs.size(), 3, 2, 0, 0, correspondingParam1, correspondingParam2),
              _RsCurrent(Rs), _RsSaved(Rs.size()), _Rsij(Rsij)
         {
            maxIterations = 200;
         }
  

         virtual void evalResidual(VectorArray<double>& e)
         { 
            int const nViewPairs = _Rsij.size();

            for (size_t k = 0; k < e.count(); ++k)
            {
               size_t const k1 = _correspondingParamB[k] ? (k - nViewPairs) : k;
               size_t const k2 = _correspondingParamB[k] ? k : (k + nViewPairs);

               int const i1 = _correspondingParamA[k1];
               int const i2 = _correspondingParamA[k2];

               //cout << "k = " << k << " k1 = " << k1 << " k2 = " << k2 << " j = " << _correspondingParamB[k] << " i1 = " << i1 << " i2 = " << i2 << endl;

               Matrix3x3d const& Ri = _RsCurrent[i1];
               Matrix3x3d const& Rj = _RsCurrent[i2];
               Matrix3x3d const& Rij = _Rsij[k1];

               Matrix3x3d const Rerr = Rj - Rij * Ri;
               e[k][0] = Rerr[0][0]; e[k][1] = Rerr[0][1]; e[k][2] = Rerr[0][2];
               e[k][3] = Rerr[1][0]; e[k][4] = Rerr[1][1]; e[k][5] = Rerr[1][2];
               e[k][6] = Rerr[2][0]; e[k][7] = Rerr[2][1]; e[k][8] = Rerr[2][2];
            }
         } // end evalResidual()

         virtual void fillJacobians(Matrix<double>& Ak, Matrix<double>& Bk, Matrix<double>& Ck,
                                    int i, int j, int k)
         {
            int const nViewPairs = _Rsij.size();

            size_t const k1 = _correspondingParamB[k] ? (k - nViewPairs) : k;
            size_t const k2 = _correspondingParamB[k] ? k : (k + nViewPairs);

            int const i1 = _correspondingParamA[k1];
            int const i2 = _correspondingParamA[k2];

            //cout << "k = " << k << " i = " << i << " j = " << j << " i1 = " << i1 << " i2 = " << i2 << endl;

            Matrix3x3d const& Ri = _RsCurrent[i1];
            Matrix3x3d const& Rj = _RsCurrent[i2];
            Matrix3x3d const& Rij = _Rsij[k1];

            Matrix3x3d A;
            if (j == 0)
            {
               // d(Rj - Rij*Ri)/dRi
               copyMatrix(Rij * Ri, A);
               scaleMatrixIP(-1.0, A);
            }
            else
            {
               // d(Rj - Rij*Ri)/dRj
               copyMatrix(Rj, A);
            } // end if

            Ak[0][0] = 0; Ak[1][0] = A[0][2]; Ak[2][0] = -A[0][1];
            Ak[3][0] = 0; Ak[4][0] = A[1][2]; Ak[5][0] = -A[1][1];
            Ak[6][0] = 0; Ak[7][0] = A[2][2]; Ak[8][0] = -A[2][1];

            Ak[0][1] = -A[0][2]; Ak[1][1] = 0; Ak[2][1] = A[0][0];
            Ak[3][1] = -A[1][2]; Ak[4][1] = 0; Ak[5][1] = A[1][0];
            Ak[6][1] = -A[2][2]; Ak[7][1] = 0; Ak[8][1] = A[2][0];

            Ak[0][2] = A[0][1]; Ak[1][2] = -A[0][0]; Ak[2][2] = 0;
            Ak[3][2] = A[1][1]; Ak[4][2] = -A[1][0]; Ak[5][2] = 0;
            Ak[6][2] = A[2][1]; Ak[7][2] = -A[2][0]; Ak[8][2] = 0;

            //scaleMatrixIP(-1.0, Ak);
            //cout << "Ak = "; displayMatrix(Ak);

            makeZeroMatrix(Bk);
            makeZeroMatrix(Ck);
         }

         virtual double getParameterLength() const
         {
            return sqrt(3.0 * _nParametersA);
         }

         virtual void updateParametersA(VectorArray<double> const& deltaAi)
         {
            Matrix3x3d dR;
            for (int i = 0; i < _nParametersA; ++i) 
            {
               //cout << "deltaAi[" << i << "] = "; displayVector(deltaAi[i]);
               Vector3d omega(deltaAi[i][0], deltaAi[i][1], deltaAi[i][2]);

               Matrix3x3d const oldR(_RsCurrent[i]);

               Matrix3x3d dR;
               createRotationMatrixRodrigues(omega, dR);
               _RsCurrent[i] = oldR * dR;
            }
         } // end updateParametersA()

         virtual void updateParametersB(VectorArray<double> const& deltaBj) { }
         virtual void updateParametersC(Vector<double> const& deltaC)       { }

         virtual void saveAllParameters()    { _RsSaved = _RsCurrent; }
         virtual void restoreAllParameters() { _RsCurrent = _RsSaved; }

         vector<Matrix3x3d>        _RsCurrent, _RsSaved;
         vector<Matrix3x3d> const& _Rsij;
   }; // end struct SparseRotationOptimizer

} // end namespace <>



namespace V3D
{
   void
   refineConsistentRotations(int const nViews,
                             std::vector<Matrix3x3d> const& relativeRotations,
                             std::vector<std::pair<int, int> > const& viewPairs,
                             std::vector<Matrix3x3d>& rotations)
   {
      vector<int> correspondingParam1(2*viewPairs.size());
      vector<int> correspondingParam2(2*viewPairs.size());

      for (size_t i = 0; i < viewPairs.size(); ++i) correspondingParam1[i] = viewPairs[i].first;
      for (size_t i = 0; i < viewPairs.size(); ++i) correspondingParam1[i+viewPairs.size()] = viewPairs[i].second;

      for (size_t i = 0; i < viewPairs.size(); ++i) correspondingParam2[i] = 0;
      for (size_t i = 0; i < viewPairs.size(); ++i) correspondingParam2[i+viewPairs.size()] = 1;

      SparseRotationOptimizer opt(rotations, relativeRotations, correspondingParam1, correspondingParam2);
      optimizerVerbosenessLevel = 1;
      opt.minimize();

      for (int i = 0; i < rotations.size(); ++i) rotations[i] = opt._RsCurrent[i];
   } // end refineConsistentRotations()

} // end namespace V3D
#endif //(V3DLIB_ENABLE_SUITESPARSE)