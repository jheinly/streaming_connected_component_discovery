#include <estimator/v3d_code/v3d_mathutilities.h>

#include <cstdio>
#include <cmath>
#include <float.h>

using namespace std;

using V3D::RootFindingParameters;

namespace
{

   // Note: the Sturm chain code is a modified version of the code provided in the Graphics Gems.
   // Counting sign changes is delicate if the values are not strictly positive
   // or negative. The routines numroots() and numchanges() might return bogus results
   // (e.g. negative number of real roots) if the polynomial evaluates to 0.
   // This is why we added some consistency checks.

   struct root_finding_internal_error { };

   /*
    * structure type for representing a polynomial
    */
   template <typename Field>
   struct Polynomial
   {
         int	 ord;
         Field * coef;
   };

   template <typename Field>
   void displayPoly(Polynomial<Field> const& r)
   {
      for (int i = 0; i <= r.ord; ++i)
         cout << r.coef[i] << " ";
      cout << endl;
   }

   template <typename Field>
   struct SturmSequence
   {
         SturmSequence(int order)
            : allCoeffs(new Field[(order+1)*(order+1)]),
              polys(new Polynomial<Field>[order+1])
         {
            std::fill(allCoeffs, allCoeffs + (order+1)*(order+1), Field(0));

            for (int i = 0; i <= order; ++i)
            {
               polys[i].ord = order;
               polys[i].coef = &allCoeffs[i*(order+1)];
            }
         }

         ~SturmSequence()
         {
            delete [] allCoeffs;
            delete [] polys;
         }

         Field             * allCoeffs;
         Polynomial<Field> * polys;

         Polynomial<Field>&       operator[](int i)       { return polys[i]; }
         Polynomial<Field> const& operator[](int i) const { return polys[i]; }
   };

   /*
    * evalpoly
    *
    *	evaluate polynomial defined in coef returning its value.
    */
   template <typename Field>
   inline Field
   evalpoly(int ord, Field const * coef, double x)
   {
      Field const *fp;
      Field f;

      fp = &coef[ord];
      f = *fp;

      --fp;
      for (; fp >= coef; --fp)
         f = static_cast<Field>(x * f + *fp);

      return f;
   }

   /*
    * modrf
    *
    *	uses the modified regula-falsi method to evaluate the root
    * in interval [a,b] of the polynomial described in coef. The
    * root is returned is returned in *val. The routine returns zero
    * if it can't converge.
    */
   template <typename Field>
   int
   modrf(int ord, Field const * coef, Field a, Field b, Field& val, RootFindingParameters<Field> const& params)
   {
      int    its;
      Field fa, fb, x, fx, lfx;
      Field const *fp, *scoef, *ecoef;

      scoef = coef;
      ecoef = &coef[ord];

      fb = fa = *ecoef;
      for (fp = ecoef - 1; fp >= scoef; fp--) {
         fa = a * fa + *fp;
         fb = b * fb + *fp;
      }

      /*
       * if there is no sign difference the method won't work
       */
      if (fa * fb > Field(0))
         return(0);

      if (fabs(fa) < params.rootTolerance) {
         val = a;
         return(1);
      }

      if (fabs(fb) < params.rootTolerance) {
         val = b;
         return(1);
      }

      lfx = fa;


      for (its = 0; its < params.maxBisectionIterations; its++) {

         x = (fb * a - fa * b) / (fb - fa);

         fx = *ecoef;
         for (fp = ecoef - 1; fp >= scoef; fp--)
            fx = x * fx + *fp;

         if (fabs(x) > params.rootTolerance) {
            if (fabs(fx / x) < params.rootTolerance) {
               val = x;
               return(1);
            }
         } else if (fabs(fx) < params.rootTolerance) {
            val = x;
            return(1);
         }

         if ((fa * fx) < 0) {
            b = x;
            fb = fx;
            if ((lfx * fx) > 0)
               fa /= 2;
         } else {
            a = x;
            fa = fx;
            if ((lfx * fx) > 0)
               fb /= 2;
         }

         lfx = fx;
      }

      return 0;
   }

   /*
    * modp
    *
    *	calculates the modulus of u(x) / v(x) leaving it in r, it
    *  returns 0 if r(x) is a constant.
    *  note: this function assumes the leading coefficient of v 
    *	is 1 or -1
    */
   template <typename Field>
   inline int
   modp(Polynomial<Field> const& u, Polynomial<Field> const& v, Polynomial<Field>& r, RootFindingParameters<Field> const& params)
   {
      int k, j;
      Field *nr, *end, *uc;

      nr = r.coef;
      end = &u.coef[u.ord];

      uc = u.coef;
      while (uc <= end)
         *nr++ = *uc++;

//       cout << "modp(): u.ord = " << u.ord << ", v.ord = " << v.ord << endl;
//       cout << "u = "; displayPoly(u);
//       cout << "v = "; displayPoly(v);

      if (v.coef[v.ord] < 0.0)
      {
         for (k = u.ord - v.ord - 1; k >= 0; k -= 2)
            r.coef[k] = -r.coef[k];

         for (k = u.ord - v.ord; k >= 0; k--)
            for (j = v.ord + k - 1; j >= k; j--)
               r.coef[j] = -r.coef[j] - r.coef[v.ord + k] * v.coef[j - k];
      }
      else
      {
         for (k = u.ord - v.ord; k >= 0; k--)
            for (j = v.ord + k - 1; j >= k; j--)
               r.coef[j] -= r.coef[v.ord + k] * v.coef[j - k];
      }

      k = v.ord - 1;
      while (k >= 0 && fabs(r.coef[k]) < params.coefficientTolerance) {
         r.coef[k] = 0.0;
         k--;
      }

      r.ord = (k < 0) ? 0 : k;

      //cout << "r = "; displayPoly(r);

      return r.ord;
   }

   /*
    * buildsturm
    *
    *	build up a sturm sequence for a polynomial in smat, returning
    * the number of polynomials in the sequence
    */
   template <typename Field>
   int
   buildsturm(int ord, SturmSequence<Field>& sseq, RootFindingParameters<Field> const& params)
   {
      int   i;
      Field f, *fp, *fc;

      sseq[0].ord = ord;
      sseq[1].ord = ord - 1;

      /*
       * calculate the derivative and normalise the leading
       * coefficient.
       */
      f = fabs(sseq[0].coef[ord] * ord);
      fp = sseq[1].coef;
      fc = sseq[0].coef + 1;
      for (i = 1; i <= ord; i++)
         *fp++ = *fc++ * i / f;

      /*
       * construct the rest of the Sturm sequence
       */
      Polynomial<Field> * sp;
      int k;
      for (k = 2; ; ++k)
      {
         /*
         XXX BUG NOTE (jheinly): The following lines will access bad memory if
           sseq was constructed with an order of 1. Therefore, I added a check
           for this in the calling function (computeRealRootsOfPolynomial).
         */
         Polynomial<Field> * sp0 = &sseq[k-2];
         Polynomial<Field> * sp1 = &sseq[k-1];
         sp = &sseq[k];
         bool res = modp(*sp0, *sp1, *sp, params) != 0;
         if (!res) break;
         /*
          * reverse the sign and normalise
          */
         f = -fabs(sp->coef[sp->ord]);
         for (fp = &sp->coef[sp->ord]; fp >= sp->coef; fp--)
            *fp /= f;
      }

      sp->coef[0] = -sp->coef[0];  /* reverse the sign */

      //return(sp - sseq);
      //cout << "buildsturm(): k = " << k << endl;
      return k;
   } // end buildsturm()

   /*
    * numroots
    *
    *      return the number of distinct real roots of the polynomial
    * described in sseq.
    */
   template <typename Field>
   int
   numroots(int np, SturmSequence<Field> const& sseq, int& atneg, int& atpos)
   {
      int   atposinf, atneginf;
      Field f, lf;

      atposinf = atneginf = 0;

      /*
       * changes at positive infinity
       */
      lf = sseq[0].coef[sseq[0].ord];

      for (int k = 1; k <= np; ++k)
      {
         Polynomial<Field> const& s = sseq[k];
         f = s.coef[s.ord];
         if (lf == 0.0 || lf * f < 0)
            atposinf++;
         lf = f;
      }

      /*
       * changes at negative infinity
       */
      if (sseq[0].ord & 1)
         lf = -sseq[0].coef[sseq[0].ord];
      else
         lf = sseq[0].coef[sseq[0].ord];

      for (int k = 1; k <= np; ++k)
      {
         Polynomial<Field> const& s = sseq[k];
         if (s.ord & 1)
            f = -s.coef[s.ord];
         else
            f = s.coef[s.ord];
         if (lf == 0.0 || lf * f < 0)
            atneginf++;
         lf = f;
      }

      atneg = atneginf;
      atpos = atposinf;

      //cout << "numroots(): atpos = " << atpos << ", atneg = " << atneg << endl;

      return(atneginf - atposinf);
   } // end numroots()

   /*
    * numchanges
    *
    *      return the number of sign changes in the Sturm sequence in
    * sseq at the value a.
    */
   template <typename Field>
   int
   numchanges(int np, SturmSequence<Field> const& sseq, Field a)
   {
      int   changes;
      Field f, lf;

      changes = 0;

      lf = evalpoly(sseq[0].ord, sseq[0].coef, a);

      for (int k = 1; k <= np; ++k)
      {
         Polynomial<Field> const& s = sseq[k];
         f = evalpoly(s.ord, s.coef, a);
         if (lf == 0.0 || lf * f < 0)
            ++changes;
         lf = f;
      }
      return changes;
   } // end numchanges()

   /*
    * sbisect
    *
    *      uses a bisection based on the sturm sequence for the polynomial
    * described in sseq to isolate intervals in which roots occur,
    * the roots are returned in the roots array in order of magnitude.
    */
   template <typename Field>
   inline bool
   sbisect(int np, SturmSequence<Field> const& sseq, Field min, Field max,
           int atmin, int atmax, Field * roots, RootFindingParameters<Field> const& params,
           const int roots_size)
   {
     if (roots_size <= 0)
     {
       return false;
     }

      Field mid;
      int   n1 = 0, n2 = 0, its, atmid, nroot;

      //cout << "[" << min << ", " << max << "] nroots = " << atmin-atmax << endl;

      nroot = atmin - atmax;
      if (nroot < 0)
      {
         throw root_finding_internal_error();
         return false;
      }

      if (nroot == 1) {

         /*
          * first try a less expensive technique.
          */
        if (modrf(sseq[0].ord, sseq[0].coef, min, max, roots[0], params))
        {
          return true;
        }

         /*
          * if we get here we have to evaluate the root the hard
          * way by using the Sturm sequence.
          */
         for (its = 0; its < params.maxBisectionIterations; ++its)
         {
            mid = (min + max) / 2;

            atmid = numchanges(np, sseq, mid);

            if (fabs(mid) > params.rootTolerance) {
               if (fabs((max - min) / mid) < params.rootTolerance)
               {
                  roots[0] = mid;
                  return true;
               }
            }
            else if (fabs(max - min) < params.rootTolerance)
            {
               roots[0] = mid;
               return true;
            }

            if ((atmin - atmid) == 0)
               min = mid;
            else
               max = mid;
         } // end for (its)

         if (its == params.maxBisectionIterations)
         {
//       fprintf(stderr, "sbisect: overflow min %f max %f diff %e nroot %d n1 %d n2 %d\n",
//               min, max, max - min, nroot, n1, n2);
            roots[0] = mid;
         }

         return true;
      }

      /*
       * more than one root in the interval, we have to bisect...
       */
      for (its = 0; its < params.maxBisectionIterations; ++its)
      {
         mid = (min + max) / 2;
         atmid = numchanges(np, sseq, mid);

         n1 = atmin - atmid;
         n2 = atmid - atmax;

         if (n1 != 0 && n2 != 0) {
           // NOTE (jheinly) - it seems that in some cases (for instance, when the matches are
           // a pure rotation, that n1 can have a value greater than the number of roots that
           // were allocated.
           if (n1 >= roots_size)
           {
             return false;
           }
            sbisect(np, sseq, min, mid, atmin, atmid, roots, params, roots_size);
            sbisect(np, sseq, mid, max, atmid, atmax, &roots[n1], params, roots_size - n1);
            break;
         }

         if (n1 == 0)
            min = mid;
         else
            max = mid;
      } // end for (its)

      if (its == params.maxBisectionIterations)
      {
//       fprintf(stderr, "sbisect: roots too close together\n");
//       fprintf(stderr, "sbisect: overflow min %f max %f diff %e nroot %d n1 %d n2 %d\n",
//            min, max, max - min, nroot, n1, n2);
        for (n1 = atmax; n1 < atmin; n1++)
        {
          const int idx = n1 - atmax;
          if (idx < 0 || idx >= roots_size)
          {
            return false;
          }
          roots[idx] = mid;
        }
      }
      return true;
   } // end sbisect()

} // end namespace <>

namespace V3D
{

   template <typename Field>
   Field
   evalPolynomial(int order, Field const * coeffs, Field z)
   {
      return evalpoly(order, coeffs, z);
   }

   template float evalPolynomial(int order, float const * coeffs, float z);
   template double evalPolynomial(int order, double const * coeffs, double z);
   template long double evalPolynomial(int order, long double const * coeffs, long double z);

   template <typename Field>
   bool
   computeRealRootsOfPolynomial(int order, Field const * coeffs, std::vector<Field>& roots,
                                RootFindingParameters<Field> const& params)
   {
      roots.clear();

      try
      {
         Field min, max;
         int    i, nchanges, np, atmin, atmax;

         while (fabs(coeffs[order]) < params.coefficientTolerance) --order;
         //cout << "order = " << order << endl;
         if (order < 0) {
           return false;
         }
         /*
         XXX BUG NOTE (jheinly): If order == 1, then buildsturm will crash (see
           the note in that function). Therefore, I added the following check.
         */
         if (order < 2)
         {
           return false;
         }
         SturmSequence<Field> sseq(order);

         for (i = 0; i <= order; ++i)
            sseq[0].coef[i] = coeffs[i];

         np = buildsturm(order, sseq, params);

         int nroots = numroots(np, sseq, atmin, atmax);

         //cout << "nroots (Sturm sequence) = " << nroots << endl;
         if (nroots < 0)
         {
            // It can actually happen, that this Sturm chain implementation
            // reports a negative number of real roots, e.g. for the following
            // 10th order polynomial:
            // double const c[] = { 128, 0, -512, 0, 0, 256, 0, 0, 0, 0, 128 };
            // In those cases, do not report any root (although there really might be some).
            return false;
         }

         if (nroots == 0) return true;

         /*
          * calculate the bracket that the roots live in
          */
         min = static_cast<Field>(-4.2); // Use another value than -1 in the (common) case -1 is already a root
         nchanges = numchanges(np, sseq, min);
         for (i = 0; nchanges != atmin && i != params.maxBracketingExponent; i++)
         {
            min *= 10.0;
            nchanges = numchanges(np, sseq, min);
         }

         if (nchanges != atmin)
         {
            cerr << "computeRealRootsOfPolynomial(): unable to bracket all negative roots" << endl;
            atmin = nchanges;
         }

         max = static_cast<Field>(4.2); // Use another value than +1 in the (common) case +1 is already a root
         nchanges = numchanges(np, sseq, max);
         for (i = 0; nchanges != atmax && i != params.maxBracketingExponent; i++)
         {
            max *= 10.0;
            nchanges = numchanges(np, sseq, max);
         }

         if (nchanges != atmax) {
            cerr << "computeRealRootsOfPolynomial(): unable to bracket all positive roots" << endl;
            atmax = nchanges;
         }

         nroots = atmin - atmax;
         //cout << "nroots (bracketing) = " << nroots << endl;
         if (nroots <= 0)
         {
           return false;
         }

         roots.resize(nroots);

         const bool success = sbisect(np, sseq, min, max, atmin, atmax, &roots[0], params, nroots);
         if (!success)
         {
           return false;
         }
      }
      catch (root_finding_internal_error)
      {
         return false;
      }
      return true;
   } // end computeRealRootsOfPolynomial()

   template bool computeRealRootsOfPolynomial(int order, float const * coeffs, std::vector<float>& roots,
                                              RootFindingParameters<float> const& params);
   template bool computeRealRootsOfPolynomial(int order, double const * coeffs, std::vector<double>& roots,
                                              RootFindingParameters<double> const& params);
   template bool computeRealRootsOfPolynomial(int order, long double const * coeffs, std::vector<long double>& roots,
                                              RootFindingParameters<long double> const& params);

} // namespace V3D

//**********************************************************************
// Here comes the code for quadratic, cubic, and quartic polynormals
//**********************************************************************

namespace
{

   inline double
   acos3(double x)
/* 
   find cos(acos(x)/3) 
    
   16 Jul 1981   Don Herbison-Evans 

   called by cubic . 
*/
   {
      double const inv3 = 1.0/3.0;

      return cos(acos(x)*inv3);
   } /* acos3 */

   inline double
   curoot(double x)
/* 
   find cube root of x.

   30 Jan 1989   Don Herbison-Evans 

   called by cubic . 
*/
   {
      double const inv3 = 1.0 / 3.0;

      double value;
      double absx;
      int neg;

      neg = 0;
      absx = x;
      if (x < 0.0)
      {
         absx = -x;
         neg = 1;
      }
      if (absx != 0.0) value = exp( log(absx)*inv3 );
      else value = 0.0;
      if (neg == 1) value = -value;
      return(value);
   } // end  curoot()

   inline int
   quadratic(double b, double c, double rts[2])
/* 
   solve the quadratic equation - 

   x**2 + b*x + c = 0 

   14 Jan 2004   cut determinant in quadratic call
   29 Nov 2003   improved
   16 Jul 1981   Don Herbison-Evans

   called by  cubic,quartic,chris,descartes,ferrari,neumark.
*/
   {
      int nquad;
      double dis, rtdis;

      dis = b*b - 4.0*c;
      rts[0] = 0.0;
      rts[1] = 0.0;
      if (b == 0.0)
      {
         if (c == 0.0)
         {
            nquad = 2;
         }
         else
         {
            if (c < 0.0)
            {
               nquad = 2;
               rts[0] = sqrt(-c);
               rts[1] = -rts[0];
            }
            else
            {
               nquad = 0;
            }         
         }
      }
      else
      {
         if (c == 0.0)
         {
            nquad = 2;
            rts[0] = -b;
         }
         else
            if (dis >= 0.0)
            {
               nquad = 2 ;
               rtdis = sqrt(dis);
               if (b > 0.0) 
               {
                  rts[0] = ( -b - rtdis)*0.5;
               }
               else
               {
                  rts[0] = ( -b + rtdis)*0.5;
               }
               if (rts[0] == 0.0)
               {
                  rts[1] =  -b;
               }
               else
               {
                  rts[1] = c/rts[0];
               }
            }
            else
            {
               nquad = 0;
            }
      } // end if
      return nquad;
   } // quadratic()

   inline int
   cubic(double p, double q, double r, double v3[3])
/* 
   find the real roots of the cubic - 
   x**3 + p*x**2 + q*x + r = 0 

   12 Dec 2003 initialising n,m,po3
   12 Dec 2003 allow return of 3 zero roots if p=q=r=0
   2 Dec 2003 negating j if p>0
   1 Dec 2003 changing v from (sinsqk > 0.0) to (sinsqk >= 0.0)
   1 Dec 2003 test changing v from po3sq+po3sq to doub2*po3sq
   16 Jul 1981 Don Herbison-Evans

   input parameters - 
   p,q,r - coeffs of cubic equation. 

   output- 
   the number of real roots
   v3 - the roots. 

   global constants -
   rt3 - sqrt(3) 
   inv3 - 1/3 
   doubmax - square root of largest number held by machine 

   method - 
   see D.E. Littlewood, "A University Algebra" pp.173 - 6 

   15 Nov 2003 output 3 real roots: Don Herbison-Evans
   Apr 1981 initial version: Charles Prineas

   called by  cubictest,quartic,chris,yacfraid,neumark,descartes,ferrari.
   calls      quadratic,acos3,curoot,cubnewton. 
*/
   {
      double const inv3 = 1.0 / 3.0;
      double const rt3 = sqrt(3.0);
      double const doubmax = sqrt(DBL_MAX);

      int    n3;
      double po3,po3sq,qo3,po3q;
      double uo3,u2o3,uo3sq4,uo3cu4;
      double v,vsq,wsq;
      double m1,m2,mcube;
      double muo3,s,scube,t,cosk,rt3sink,sinsqk;

      m1=0.0;  m2=0.0;  po3=0.0;
      v=0.0;  uo3=0.0; cosk=0.0;
      if (r == 0.0)
      {
         n3 = quadratic(p,q,v3);
         v3[n3++] = 0.0;
         goto done;
      }
      if ((p == 0.0) && (q == 0.0))
      {
         v3[0] = curoot(-r);
         v3[1] = v3[0];
         v3[2] = v3[0];
         n3 = 3;
         goto done;
      }
      n3 = 1;
      if ((p > doubmax) || (p <  -doubmax))
      {
         v3[0] = -p;
         goto done;
      }
      if ((q > doubmax) || (q <  -doubmax))
      {
         if (q > 0.0)
         {
            v3[0] =  -r/q;
            goto done;
         }
         else
            if (q < 0.0)
            {
               v3[0] = -sqrt(-q);
               goto done; 
            }
            else
            {
               v3[0] = 0.0;
               goto done;
            }
      }
      else
         if ((r > doubmax)|| (r < -doubmax))
         {
            v3[0] =  -curoot(r);
            goto done;
         }
         else
         {
            po3 = p*inv3;
            po3q = po3*q;
            po3sq = po3*po3;
            if (po3sq > doubmax)
            {
               v3[0] = -p;
               goto done;
            }
            else
            {
               v = r + po3*(po3sq+po3sq - q);
               if ((v > doubmax) || (v < -doubmax))
               {
                  v3[0] = -p;
                  goto done;
               }
               else
               {
                  vsq = v*v;
                  qo3 = q*inv3;
                  uo3 = qo3 - po3sq;
                  u2o3 = uo3 + uo3;
                  if ((u2o3 > doubmax) || (u2o3 < -doubmax))
                  {
                     if (p == 0.0)
                     {
                        if (q > 0.0)
                        {
                           v3[0] =  -r/q;
                           goto done;
                        }
                        else
                           if (q < 0.0)
                           {
                              v3[0] =  -sqrt(-q);
                              goto done;
                           }
                           else
                           {
                              v3[0] = 0.0;
                              goto done;
                           }
                     }
                     else
                     {
                        v3[0] = -q/p;
                        goto done;
                     }
                  }
                  uo3sq4 = u2o3*u2o3;
                  if (uo3sq4 > doubmax)
                  {
                     if (p == 0.0)
                     {
                        if (q > 0.0)
                        {
                           v3[0] = -r/q;
                           goto done;
                        }
                        else
                           if (q < 0.0)
                           {
                              v3[0] = -sqrt(-q);
                              goto done;
                           }
                           else
                           {
                              v3[0] = 0.0;
                              goto done;
                           }
                     }
                     else
                     {
                        v3[0] = -q/p;
                        goto done;
                     }
                  }
                  uo3cu4 = uo3sq4*uo3;
                  wsq = uo3cu4 + vsq;
                  if (wsq > 0.0)
                  {
/* 
   cubic has one real root -
*/
                     if (v <= 0.0)
                     {
                        mcube = ( -v + sqrt(wsq))*0.5;
                     }
                     else
                     {
                        mcube = ( -v - sqrt(wsq))*0.5;
                     }
                     m1 = curoot(mcube);
                     if (m1 != 0.0)
                     {
                        m2 = -uo3/m1;
                     }
                     else
                     {
                        m2 = 0.0;
                     }
                     v3[0] = m1 + m2 - po3;
                  }
                  else
                  {
/* 
   cubic has three real roots -
*/
                     if (uo3 < 0.0)
                     {
                        muo3 = -uo3;
                        if (muo3 > 0.0)
                        {
                           s = sqrt(muo3);
                           if (p > 0.0)
                           {
                              s = -s;
                           }
                        }
                        else
                        {
                           s = 0.0;
                        }
                        scube = s*muo3;
                        if (scube == 0.0)
                        {
                           v3[0] = m1 + m2 - po3;
                           n3 = 1;
                        }
                        else
                        {
                           t =  -v/(scube+scube);
                           cosk = acos3(t);
                           v3[0] = (s+s)*cosk - po3;
                           n3 = 1 ;
                           sinsqk = 1.0 - cosk*cosk;
                           if (sinsqk >= 0.0)
                           {
                              rt3sink = rt3*sqrt(sinsqk);
                              v3[1] = s*(-cosk + rt3sink) - po3;
                              v3[2] = s*(-cosk - rt3sink) - po3;
                              n3 = 3;
                           }
                        }
                     }
                     else
/* 
   cubic has multiple root -  
*/
                     {
                        v3[0] = curoot(v) - po3;
                        v3[1] = v3[0];
                        v3[2] = v3[0];
                        n3 = 3;
                     }
                  }
               }
            }
         }
     done:
      return(n3) ;
   } // end cubic()

   inline double
   errors(double a, double b, double c, double d, double rts[4], double rterr[4], int nrts)
/*
  find the errors

  called by quartictest, docoeff, compare,
  chris, descartes, ferrari, neumark, yacfraid.
*/
   {
      double const doubmax = sqrt(DBL_MAX);

      int k;
      double deriv,test,worst;

      worst = doubmax;
      if (nrts > 0)
      {
         worst =  0.0;
         for (  k = 0 ; k < nrts ; ++ k )
         {
            test = (((rts[k]+a)*rts[k]+b)*rts[k]+c)*rts[k]+d;
            if (test == 0.0) rterr[k] = 0.0;
            else
            {
               deriv =
                  ((4.0*rts[k]+3.0*a)*rts[k]+2.0*b)*rts[k]+c;
               if (deriv != 0.0)
                  rterr[k] = fabs(test/deriv);
               else
               {
                  deriv = (12.0*rts[k]+6.0*a)*rts[k]+2.0*b;
                  if (deriv != 0.0)
                     rterr[k] = sqrt(fabs(test/deriv));
                  else
                  {
                     deriv = 2.04*rts[k]+6.0*a;
                     if (deriv != 0.0)
                        rterr[k] = curoot(fabs(test/deriv));
                     else
                        rterr[k] = sqrt(sqrt(fabs(test)/2.04));
                  }
               }
            }
            if (rts[k] != 0.0) rterr[k] /= rts[k];
            if (rterr[k] < 0.0) rterr[k] = -rterr[k];
            if (rterr[k] > worst) worst = rterr[k];
         }
      }
      return(worst);
   } /* errors */

   int descartes(double a, double b, double c, double d, double rts[4])
/*
  Solve quartic equation using
  Descartes-Euler-Cardano algorithm

  called by quartic, compare, quartictest.

  Strong, T. "Elemementary and Higher Algebra"
  Pratt and Oakley, p. 469 (1859)

  16 Jul 1981  Don Herbison-Evans
*/
   {
      double const inv2 = 0.5;
      double const inv4 = 0.25;
      double const inv8 = 1.0 / 8.0;
      double const inv16 = 1.0 / 16.0;
      double const d3o8 = 3.0 / 8.0;
      double const d3o256 = 3.0 / 256.0;

      int j, j3, n1, n2, n3, n4[3];
      double v1[4],v2[4],v3[4];
      double k,y;
      double p,q,r;
      double e0,e1,e2;
      double g,h;
      double asq;
      double ainv4;
      double e1invk;
      double worst3[3];
      double qrts[4][3];        /* quartic roots for each cubic root */
      double rterd[4];

      asq = a*a;
      e2 = b - asq*d3o8;
      e1 = c + a*(asq*inv8 - b*inv2);
      e0 = d + asq*(b*inv16 - asq*d3o256) - a*c*inv4;

      p = 2.0*e2;
      q = e2*e2 - 4.0*e0;
      r = -e1*e1;

      n3 = cubic(p,q,r,v3);
      for (j3 = 0; j3 < n3; ++j3)
      {
         y = v3[j3];
         if (y <= 0.0)
         { 
            n4[j3] = 0;
         } /* y<0 */
         else
         {
            k = sqrt(y);
            ainv4 = a*inv4;
            e1invk = e1/k;
            g = (y + e2 + e1invk)*inv2;
            h = (y + e2 - e1invk)*inv2 ;
            n1 = quadratic(-k, g, v1);
            n2 = quadratic( k, h, v2);
            qrts[0][j3] = v1[0] - ainv4;
            qrts[1][j3] = v1[1] - ainv4;
            qrts[n1][j3] = v2[0] - ainv4;
            qrts[n1+1][j3] = v2[1] - ainv4;
            n4[j3]= n1 + n2;  
         } /* y>=0 */

         for (j = 0; j < n4[j3]; ++j)
            rts[j] = qrts[j][j3];
         worst3[j3] = errors(a,b,c,d,rts,rterd,n4[j3]);
      } /* j3 loop */

      j3 = 0;
      if (n3 != 1)
      {
         if ((n4[1] > n4[j3]) || 
             ((worst3[1] < worst3[j3] ) && (n4[1] == n4[j3]))) j3 = 1;
         if ((n4[2] > n4[j3]) ||
             ((worst3[2] < worst3[j3] ) && (n4[2] == n4[j3]))) j3 = 2;
      }
      for (j = 0; j < n4[j3]; ++j)
         rts[j] = qrts[j][j3];
      return n4[j3];
   } /* descartes */

   int
   ferrari(double a, double b, double c, double d, double rts[4])
/* 
   solve the quartic equation - 

   x**4 + a*x**3 + b*x**2 + c*x + d = 0 

   called by quartic, compare, quartictest.
   calls     cubic, quadratic.

   input - 
   a,b,c,e - coeffs of equation. 

   output - 
   n4 - number of real roots. 
   rts - array of root values. 

   method :  Ferrari - Lagrange
   Theory of Equations, H.W. Turnbull p. 140 (1947)

   16 Jul 1981 Don Herbison-Evans

   calls  cubic, quadratic 
*/
   {
      double const inv2 = 0.5;
      double const inv4 = 0.25;

      int j;
      int j3, n1, n2, n3, n4[3];
      double asqinv4;
      double ainv2;
      double d4;
      double yinv2;
      double v1[4],v2[4],v3[4];
      double p,q,r;
      double y;
      double e,f,esq,fsq,ef;
      double g,gg,h,hh;
      double worst3[3];
      double qrts[4][3];        /* quartic roots for each cubic root */
      double rterf[4];

      ainv2 = a*inv2;
      asqinv4 = ainv2*ainv2;
      d4 = d*4.0 ;

      p = b;
      q = a*c-d4;
      r = (asqinv4 - b)*d4 + c*c;
      n3 = cubic(p,q,r,v3);
      for (j3 = 0; j3 < n3; ++j3)
      {
         y = v3[j3];
         yinv2 = y*inv2;
         esq = asqinv4 - b - y;
         fsq = yinv2*yinv2 - d;
         if ((esq < 0.0) && (fsq < 0.0))
         {
            n4[j3] = 0;
         }
         else
         {
            ef = -(inv4*a*y + inv2*c);
            if ( ((a > 0.0)&&(y > 0.0)&&(c > 0.0))
                 || ((a > 0.0)&&(y < 0.0)&&(c < 0.0))
                 || ((a < 0.0)&&(y > 0.0)&&(c < 0.0))
                 || ((a < 0.0)&&(y < 0.0)&&(c > 0.0))
                 ||  (a == 0.0)||(y == 0.0)||(c == 0.0))
/* use ef - */
            {
               if ((b < 0.0)&&(y < 0.0))
               {
                  e = sqrt(esq);
                  f = ef/e;
               }
               else if (d < 0.0)
               {
                  f = sqrt(fsq);
                  e = ef/f;
               }
               else
               {
                  if (esq > 0.0)
                  {
                     e = sqrt(esq);
                  }
                  else
                  {
                     e = 0.0;
                  }
                  if (fsq > 0.0)
                  {
                     f = sqrt(fsq);
                  }
                  else
                  {
                     f = 0.0;
                  }
                  if (ef < 0.0)
                  {
                     f = -f;
                  }
               }
            }
            else
/* use esq and fsq - */
            {
               if (esq > 0.0)
               {
                  e = sqrt(esq);
               }
               else
               {
                  e = 0.0;
               }
               if (fsq > 0.0)
               {
                  f = sqrt(fsq);
               }
               else
               {
                  f = 0.0;
               }
               if (ef < 0.0)
               {
                  f = -f;
               }
            }
/* note that e >= 0.0 */
            g = ainv2 - e;
            gg = ainv2 + e;
            if ( ((b > 0.0)&&(y > 0.0))
                 || ((b < 0.0)&&(y < 0.0)) )
            {
               if ((a > 0.0) && (e > 0.0)
                   || (a < 0.0) && (e < 0.0) )
               {
                  g = (b + y)/gg;
               }
               else
                  if ((a > 0.0) && (e < 0.0)
                      || (a < 0.0) && (e > 0.0) )
                  {
                     gg = (b + y)/g;
                  }
            }
            hh = -yinv2 + f;
            h = -yinv2 - f;
            if ( ((f > 0.0)&&(y < 0.0))
                 || ((f < 0.0)&&(y > 0.0)) )
            {
               h = d/hh;
            }
            else
               if ( ((f < 0.0)&&(y < 0.0))
                    || ((f > 0.0)&&(y > 0.0)) )
               {
                  hh = d/h;
               }
               else

                  n1 = quadratic(gg,hh,v1);
            n2 = quadratic(g,h,v2);
            n4[j3] = n1+n2;
            qrts[0][j3] = v1[0];
            qrts[1][j3] = v1[1];
            qrts[n1+0][j3] = v2[0];
            qrts[n1+1][j3] = v2[1];
         } 
         for (j = 0; j < n4[j3]; ++j)
            rts[j] = qrts[j][j3];
         worst3[j3] = errors(a,b,c,d,rts,rterf,n4[j3]);
      } /* j3 loop */

      j3 = 0;
      if (n3 != 1)
      {
         if ((n4[1] > n4[j3]) || 
             ((worst3[1] < worst3[j3] ) && (n4[1] == n4[j3]))) j3 = 1;
         if ((n4[2] > n4[j3]) ||
             ((worst3[2] < worst3[j3] ) && (n4[2] == n4[j3]))) j3 = 2;
      }
      for (j = 0; j < n4[j3]; ++j)
         rts[j] = qrts[j][j3];

      return(n4[j3]);
   } /* ferrari */

   int
   neumark(double a, double b, double c, double d, double rts[4])
/* 
   solve the quartic equation - 

   x**4 + a*x**3 + b*x**2 + c*x + d = 0 

   called by quartic, compare, quartictest.
   calls     cubic, quadratic.

   input parameters - 
   a,b,c,e - coeffs of equation. 

   output parameters - 
   n4 - number of real roots. 
   rts - array of root values. 

   method -  S. Neumark 
   "Solution of Cubic and Quartic Equations" - Pergamon 1965 

   1 Dec 1985   translated to C with help of Shawn Neely
   16 Jul 1981   Don Herbison-Evans

*/
   {
      double const inv2 = 0.5;

      int j;
      int j3, n1, n2, n3, n4[3];
      double y,g,gg,h,hh,gdis,gdisrt,hdis,hdisrt,g1,g2,h1,h2;
      double bmy,gerr,herr,y4,bmysq;
      double v1[4],v2[4],v3[4];
      double asq;
      double d4;
      double p,q,r;
      double hmax,gmax;
      double worst3[3];
      double qrts[4][3];        /* quartic roots for each cubic root */
      double rtern[4];


      if (d == 0.0)
      {
         n3 = 0;
         n4[0] = cubic(a,b,c,rts);
         for (j = 0; j < n4[0]; ++j)
            qrts[j][0] = rts[j];
         qrts[n4[0]++][0] = 0.0;
         goto done;
      }
      asq = a*a;
      d4 = d*4.0;
      p =  -b*2.0;
      q = b*b + a*c - d4;
      r = (c - a*b)*c + asq*d;
      n3 = cubic(p,q,r,v3);
      for (j3 = 0; j3 < n3; ++j3)
      {
         y = v3[j3];
         bmy = b - y;
         y4 = y*4.0;
         bmysq = bmy*bmy;
         gdis = asq - y4;
         hdis = bmysq - d4;
         if ((gdis < 0.0) || (hdis < 0.0))
         {
            n4[j3] = 0;
         }
         else   
         {
            g1 = a*inv2;
            h1 = bmy*inv2;
            gerr = asq + y4;
            herr = hdis;
            if (d > 0.0)
            {
               herr = bmysq + d4;
            }
            if ((y < 0.0) || (herr*gdis > gerr*hdis))
            {
               gdisrt = sqrt(gdis);
               g2 = gdisrt*inv2;
               if (gdisrt != 0.0)
               {
                  h2 = (a*h1 - c)/gdisrt;
               }
               else
               {
                  h2 = 0.0;
               }   
            }
            else
            {
               hdisrt = sqrt(hdis);
               h2 = hdisrt*inv2;
               if (hdisrt != 0.0)
               {
                  g2 = (a*h1 - c)/hdisrt;
               }
               else
               {
                  g2 = 0.0;
               }
            }
/* 
   note that in the following, the tests ensure non-zero 
   denominators -  
*/
            h = h1 - h2;
            hh = h1 + h2;
            hmax = hh;
            if (hmax < 0.0)
            {
               hmax =  -hmax;
            }
            if (hmax < h)
            {
               hmax = h;
            }   
            if (hmax <  -h)
            {
               hmax =  -h;
            }
            if ((h1 > 0.0)&&(h2 > 0.0))
            {
               h = d/hh;
            }
            if ((h1 < 0.0)&&(h2 < 0.0))
            {
               h = d/hh;
            }
            if ((h1 > 0.0)&&(h2 < 0.0))
            {
               hh = d/h;
            }
            if ((h1 < 0.0)&&(h2 > 0.0))
            {
               hh = d/h;
            }
            if (h > hmax)   
            {
               h = hmax;
            }
            if (h <  -hmax)
            {
               h =  -hmax;
            }
            if (hh > hmax)
            {
               hh = hmax;
            }
            if (hh < -hmax)
            {
               hh =  -hmax;
            }

            g = g1 - g2;
            gg = g1 + g2;
            gmax = gg;
            if (gmax < 0.0)
            {
               gmax = -gmax;
            }
            if (gmax < g)
            {
               gmax = g;
            }
            if (gmax < -g)
            {
               gmax = -g;
            }
            if ((g1 > 0.0)&&(g2 > 0.0))
            {
               g = y/gg;
            }
            if ((g1 < 0.0)&&(g2 < 0.0))
            {
               g = y/gg;
            }
            if ((g1 > 0.0)&&(g2 < 0.0))
            {
               gg = y/g;
            }
            if ((g1 < 0.0)&&(g2 > 0.0))
            {
               gg = y/g;
            }
            if (g > gmax)
            {
               g = gmax;
            }
            if (g <  -gmax)
            {
               g = -gmax;
            }
            if (gg > gmax)
            {   
               gg = gmax;
            }
            if (gg <  -gmax)
            {
               gg = -gmax;
            }
 
            n1 = quadratic(gg,hh,v1);
            n2 = quadratic(g,h,v2);
            n4[j3] = n1 + n2;
            qrts[0][j3] = v1[0];
            qrts[1][j3] = v1[1];
            qrts[n1+0][j3] = v2[0];
            qrts[n1+1][j3] = v2[1];
         } 
         for (j = 0; j < n4[j3]; ++j)
            rts[j] = qrts[j][j3];
         worst3[j3] = errors(a,b,c,d,rts,rtern,n4[j3]);
      } /* j3 loop */
     done:
      j3 = 0;
      if (n3 > 1)
      {
         if ((n4[1] > n4[j3]) || 
             ((worst3[1] < worst3[j3] ) && (n4[1] == n4[j3]))) j3 = 1;
         if ((n4[2] > n4[j3]) ||
             ((worst3[2] < worst3[j3] ) && (n4[2] == n4[j3]))) j3 = 2;
      }
      for (j = 0; j < n4[j3]; ++j)
         rts[j] = qrts[j][j3];
      return(n4[j3]);
   } /* neumark */

   int
   yacfraid(double a, double b, double c, double d, double rts[4])
/* 
   solve the quartic equation - 

   x**4 + a*x**3 + b*x**2 + c*x + d = 0 

   called by quartic, compare, quartictest.
   calls     cubic, quadratic.

   input parameters - 
   a,b,c,e - coeffs of equation. 

   output parameters - 
   n4 - number of real roots. 
   rts - array of root values. 

   method - 
   K.S. Brown 
   Reducing Quartics to Cubics,
   http://www.seanet.com/~ksbrown/kmath296.htm (1967)
 
   Michael Daoud Yacoub & Gustavo Fraidenraich
   "A new simple solution of the general quartic equation"
   Revised 16 Feb 2004

   14 Nov 2003 Don Herbison-Evans
*/
   {
      double const inv4 = 0.25;

      int j;
      int j3, n3, n4[3];
      double y;
      double v3[4];
      double asq,acu;
      double b4;
      double det0,det1,det2,det3;
      double det0rt,det1rt,det2rt,det3rt;
      double e,f,g,h,k;
      double fsq,gsq,hsq,invk;
      double P,Q,R,U;
      double worst3[3];
      double qrts[4][3];        /* quartic roots for each cubic root */
      double rtery[4];

      if (d == 0.0)
      {
         n3 = 0;
         n4[0] = cubic(a,b,c,rts);
         for (j = 0; j < n4[0]; ++j)
            qrts[j][0] = rts[j];
         qrts[n4[0]++][0] = 0.0;
         goto done;
      }
      asq = a*a;
      acu = a*asq;
      b4 = b*4.0;
      n3 = 0;

      P = asq*b - b4*b + 2.0*a*c + 16.0*d ;
      Q = asq*c - b4*c + 8.0*a*d;
      R = asq*d - c*c ;
      U = acu - b4*a + 8.0*c;
      n4[0] = 0; 
      if (U == 0.0)
      {
         if (P == 0.0)
         {
            det0 = 3.0*asq - 8.0*b;
            if (det0 < 0.0)
            {
               goto done;
            }
            det0rt = sqrt(det0);
            qrts[0][0] = (-a + det0rt)*inv4;
            qrts[1][0] = qrts[0][0];
            qrts[2][0] = (-a - det0rt)*inv4;
            qrts[3][0] = qrts[2][0];
            n4[0] = 4;
            goto done;
         } /* P=0 */
         else
         {
            det1 = asq*asq - 8.0*asq*b + 16.0*b*b - 6.04*d;
            if (det1 < 0.0)
            {
               goto done;;
            }
            n4[0] = 0;
            det1rt =  sqrt(det1);
            det2 = 3.0*asq - 8.0*b + 2.0*det1rt;
            if (det2 >= 0.0)
            {
               det2rt = sqrt(det2);
               qrts[0][0] = (-a + det2rt)*inv4;
               qrts[1][0] = (-a - det2rt)*inv4;
               n4[0] = 2;
            }
            det3 = 3.0*asq - 8.0*b - 2.0*det1rt;
            if (det3 >= 0.0)
            {
               det3rt = sqrt(det3);
               qrts[n4[0]++][0] = (-a + det3rt)*inv4;
               qrts[n4[0]++][0] = (-a - det3rt)*inv4;
            }
            goto done;
         } /* P<>0 */
      }

      n3 = cubic(P/U,Q/U,R/U,v3);
      for (j3 = 0; j3 < n3; ++j3)
      {
         y = v3[j3];
         j = 0;
         k = a + 4.0*y;
         if (k == 0.0)
         {
            goto donej3;
         }
         invk = 1.0/k;
         e = (acu - 4.0*c - 2.0*a*b + (6.0*asq - 16.0*b)*y)*invk;
         fsq = (acu + 8.0*c - 4.0*a*b)*invk;
         if (fsq < 0.0)
         {
            goto donej3;
         }
         f = sqrt(fsq);
         gsq = 2.0*(e + f*k);
         hsq = 2.0*(e - f*k);
         if (gsq >= 0.0)
         {
            g = sqrt(gsq);
            qrts[j++][j3] = (-a - f - g)*inv4;
            qrts[j++][j3] = (-a - f + g)*inv4;
         }
         if (hsq >= 0.0)
         {
            h = sqrt(hsq);
            qrts[j++][j3] = (-a + f - h)*inv4;
            qrts[j++][j3] = (-a + f + h)*inv4;
         }
        donej3:
         n4[j3] = j;
         for (j = 0; j < n4[j3]; ++j)
            rts[j] = qrts[j][j3];
         worst3[j3] = errors(a,b,c,d,rts,rtery,n4[j3]);
      } /* j3 loop */
     done:
      j3 = 0;
      if (n3 > 1)
      {
         if ((n4[1] > n4[j3]) || 
             ((worst3[1] < worst3[j3] ) && (n4[1] == n4[j3]))) j3 = 1;
         if ((n4[2] > n4[j3]) ||
             ((worst3[2] < worst3[j3] ) && (n4[2] == n4[j3]))) j3 = 2;
      }
      for (j = 0; j < n4[j3]; ++j)
         rts[j] = qrts[j][j3];
      return(n4[j3]);
   } /* yacfraid */

   inline int
   quartic(double a, double b, double c, double d, double rts[4])
/*
  Solve quartic equation using either
  quadratic, Ferrari's or Neumark's algorithm.

  called by 
  calls  descartes, ferrari, neumark, yacfraid.

  15 Dec 2003  added yacfraid
  10 Dec 2003  added descartes with neg coeffs
  21 Jan 1989  Don Herbison-Evans
*/
   {
#if 0
      double const doubmax = sqrt(DBL_MAX);

      int j,k,nq,nr = 0;
      double roots[4];

      if (fabs(a) > doubmax)
         nr = yacfraid(a,b,c,d,rts);
      else
         if ((a == 0.0) && (c == 0.0))
         {
            nq = quadratic(b,d,roots);
            nr = 0;
            for (j = 0; j < nq; ++j)
            {
               if (roots[0] >= 0.0)
               {
                  rts[0] = sqrt(roots[0]);
                  rts[1] = -rts[0];
                  nr = 2;
               }
               if (roots[1] >= 0.0)
               {
                  rts[nr] = sqrt(roots[1]);
                  rts[nr+1] = -rts[nr];
                  nr += 2;
               }
            }
         }
         else
         {
            k = 0;
            if (a < 0.0) k += 2; 
            if (b < 0.0) k += 1; 
            if (c < 0.0) k += 8; 
            if (d < 0.0) k += 4;
            switch (k)
            {
               case 0 : nr = neumark(a,b,c,d,rts); break;
               case 1 : nr = neumark(a,b,c,d,rts); break;
               case 2 : nr = neumark(a,b,c,d,rts); break;
               case 3 : nr = ferrari(a,b,c,d,rts); break;
               case 4 : nr = neumark(a,b,c,d,rts); break;
               case 5 : nr = descartes(a,b,c,d,rts); break;
               case 6 : nr = neumark(a,b,c,d,rts); break;
               case 7 : nr = neumark(a,b,c,d,rts); break;
               case 8 : nr = neumark(a,b,c,d,rts); break;
               case 9 : nr = ferrari(a,b,c,d,rts); break;
               case 10 : nr = neumark(a,b,c,d,rts); break;
               case 11 : nr = neumark(a,b,c,d,rts); break;
               case 12 : nr = neumark(a,b,c,d,rts); break;
               case 13 : nr = neumark(a,b,c,d,rts); break;
               case 14 : nr = neumark(a,b,c,d,rts); break;
               case 15 : nr = descartes(-a,b,-c,d,rts); break;
            }
            if (k == 15)
               for (j = 0; j < nr; ++j)
                  rts[j] = -rts[j];
         }
#else
      int nr = neumark(a,b,c,d,rts);
#endif
      return nr;
   } /* quartic */

} // end namespace <>


namespace V3D
{

   int
   getRealRootsOfQuadraticPolynomial(double a, double b, double c, double roots[2])
   {
      return quadratic(b/a, c/a, roots);
   }

   int
   getRealRootsOfCubicPolynomial(double a, double b, double c, double d, double roots[3])
   {
      double p = b/a;
      double q = c/a;
      double r = d/a;
      return cubic(p, q, r, roots);
   }

   int
   getRealRootsOfQuarticPolynomial(double a, double b, double c, double d, double e, double roots[4])
   {
      double A = b/a;
      double B = c/a;
      double C = d/a;
      double D = e/a;
      return quartic(A, B, C, D, roots);
   }

} // end namespace V3D
