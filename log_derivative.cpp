#include <iostream>
#include <string>
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>

#include "./log.hpp"

#define DEBUG_VALUE_OF(variable) { _logDebug.value_of(#variable, variable); } 
template<class T> void __log__::value_of(const std::string& name, const T& value) 
{
    fill_wsp();
    std::cout << wsp << name << " (" << demangled_type_info_name(typeid(value)) << ") = [" << value << "]" << std::endl;
}

inline int delta(int i, int j) { return i == j; }

void potmat(Eigen::MatrixXd & w, const double r, const int nch)
// Example 5.2, p.114.
// The second test problem in coupled Schrodinger equations section.
// Study of special algorithms for solving Sturm-Liouville and Schroedinger equations
// Verrie Ledoux, Thesis
// The first few eigenvalues of the problem are:
// 14.94180054
// 17.04349658
// 21.38042053
// 26.92073133
// 51.82570724
// 55.80351609
{
    assert(nch == 4 && "Example excepts for nch=4");

    double cos_r = std::cos(r);

    for ( int i = 0; i < nch; ++i )
    {
        for ( int j = 0; j < nch; ++j )
        {
            w(i, j) = cos_r / std::max(i, j) + delta(i, j) / std::pow(r, i);
        }
    }
}

void logdb(Eigen::MatrixXd & z, Eigen::MatrixXd & w, Eigen::MatrixXd & wref, const int nch, const double rmin, const double rmax, const int nsteps)
// routine to initialise the log derivative matrix, y, at r = rmin, 
// and propagate it from rmin to rmax 
// ----------------------------------------------------------
// variables in call list:
// z            matrix of dimension (nch, nch) 
//              on return `z` contains the log derivative matrix at r = rmax
// w            workspace for the potential matrix
// wref         workspace for the reference potential
// nch          number of coupled equations
// rmin, rmax   integration range limits
//              rmin is assummed to lie inside the classically forbidden
//              region for all channels
// nsteps       number of sectors partitioning integration range 
{
    DEBUG_METHOD("logdb");

    double h = (rmax - rmin) / nsteps;
    DEBUG_VALUE_OF(h);

    double d3 = h / 3.0;
    double d6 = h * h / 6.0;

    double r = rmin;
    potmat(w, r, nch);

    // use diagonal approximation to WKB initial value for log derivative matrix
    // (equation 16)
    // rmin is assumed to lie inside classically forbidden region in all 
    // the channels (w(ii) > 0)
    //
    // zero out z matrix
    z = Eigen::MatrixXd::Zero(nch, nch);

    double wdiag = 0.0;
    for ( int i = 0; i < nch; ++i )
    {
        wdiag = w(i, i);
        if ( wdiag < 0.0 )
        {
            std::cout << "DIAGONAL REFERENCE POTENTIAL SHOULD BE POSITIVE! ABORT." << std::endl;
            exit( 1 );
        }

        z(i, i) = std::sqrt(wdiag);
    }

    Eigen::MatrixXd z1 = Eigen::MatrixXd::Zero(nch, nch);
    Eigen::MatrixXd z2 = Eigen::MatrixXd::Zero(nch, nch);
    Eigen::MatrixXd z3 = Eigen::MatrixXd::Zero(nch, nch);
    Eigen::MatrixXd z4 = Eigen::MatrixXd::Zero(nch, nch);

    int istep = 1;
    // propagate z from rmin to rmax
    for ( int kstep = 1; kstep <= nsteps; ++kstep, istep += 2 )
    {
        // обнулить z??
        // eq. 5.61 from Manolopoulos Thesis
        z += d3 * w; 

        // the reference potential for the sector is the diagonal of the
        // coupling matrix evaluated at the centre of the sector, r = c
        // (eqn 15 of Manolopoulos1986)
        r = rmin + 0.5 * istep * h;
        DEBUG_VALUE_OF(r);
        DEBUG_PRINT("Reference potential is evaluated at the centre of the sector (r = c)\n"); 
        potmat(w, r, nch);
        
        // the reference potential:
        for ( int ich = 0; ich < nch; ++ich )
            wref(ich, ich) = w(ich, ich);

        // adjust quadrature contribution at r = a to account for
        // sector reference potential
        // (eqn 11 of Manolopoulos1986)
        z -= d3 * wref;
        // итого мы посчитали Q(a) = h/3 U(a) = h/3(W(a) - Wref(a))

        // evaluate half sector propagators (eqn 10, Manolopoulos1986)
        // z1: y1, z2: y2, z3: y3, z4: y4
        double arg;
        for ( int ich = 0; ich < nch; ++ich )
        {
            arg = std::sqrt(std::abs(wref(ich, ich)));
            if ( wref(ich, ich) < 0.0 )
            {
                double ctn = 1.0 / std::tan(arg);
                z1(ich, ich) = arg * ctn;
                double csc = 1.0 / std::sin(arg);
                z2(ich, ich) = arg * csc;
            }
            else
            {
                double ctnh = 1.0 / std::tanh(arg);
                z1(ich, ich) = arg * ctnh;
                double csch = 1.0 / std::sinh(arg); 
                z2(ich, ich) = arg * csch;
            }

            z3(ich, ich) = z2(ich, ich);
            z4(ich, ich) = z1(ich, ich);
        } 
        
        // propagate z matrix across the first half sector
        // from r = a to r = c (eqn 14 of Manolopoulos1986)
        
        // Y(c) = z4(a,c) - z3(a,c) * (z(a) + z1(a,c))^(-1) * z2(a,c)
        z += z1;
        z.inverse();
        // z now contains (z(a) + z1(a,c))^(-1)
        z = z4 - z3 * z * z2;
        // z now contains z(c)
        
        // evaluate quadrature contribution at midpoint r = c 
        // (eqn 12 from Manolopoulos1986)
        // evaluate potential matrix at midpoint r = c
        // матрица w в точке r = c была рассчитана в самом начале цикла
        // при расчете Wref
        // уже можно поганить эту матрицу, так что делаем inplace transformations  
        w *= -d6;
        // U(c) = w(c) - wref(c), т.к. wref вычисляется как диагональная часть w(c), то U(c) имеет нулевые элементы на диагонали
        // а значит I - U(c) содержит единицы на диагонали
        for ( int ich = 0; ich < nch; ++ich )
            w(ich, ich) = 1.0;
        // w now containts I - h^2/6*U(c)
        w.inverse();
        
        // subtracting identity matrix
        for ( int ich = 0; ich < nch; ++ich )
            w(ich, ich) -= 1.0;

        w *= 4.0 / h;
        // w now contains Q(c)
        
        // apply quadrature contribution at midpoint
        // (eqn 13 from Manolopoulos1986)
        z1 += w; // z1(c, b)
        z4 += w; // z4(c, b)
        // z2(c, b) = z2(a, c)
        // z3(c, b) = z3(a, c)

        // propagate z matrix across the second half sector
        // from r = c to r = b (eqn 14 from Manolopoulos1986)
        
        // Y(b) = z4(c, b) - z3(c, b) * (z(c) + z1(c, b))^(-1) * z2(c, b)
        z += z1;
        z.inverse();
        // z now contains (z(c) + z1(c, b))^(-1)
        z = z4 - z3 * z * z2;
        // z now contains Y(b)

        // apply reference potential adjustment to quadrature at r = b
        // (eqn 11 from Manolopoulos 1986)
        z -= d3 * wref;

        r = rmin + kstep * h;
        std::cout << "(end of loop) r = " << r << std::endl;

        // apply quadrature contribution at r = b
        // (eqn 12 from Manolopoulos 1986)
        potmat(w, r, nch);
        w += d3 * z;
    }
}

int main()
{
    DEBUG_METHOD("main");

    const int nch = 4;
    Eigen::MatrixXd z = Eigen::MatrixXd::Zero(nch, nch);
    Eigen::MatrixXd w = Eigen::MatrixXd::Zero(nch, nch);
    Eigen::MatrixXd wref = Eigen::MatrixXd::Zero(nch, nch);
    
    const double rmin = 0.0;
    const double rmax = 1.0;

    const int nsteps = 2;

    logdb(z, w, wref, nch, rmin, rmax, nsteps);

    return 0;
}
