#include <iostream>
#include "Eigen/Eigen"
#include <vector>
#include <cmath>

using namespace std;
using namespace Eigen;

//Funzione che risolve il sistema Ax = b usando la decomposizione PALU
VectorXd SoluzionePALU(const MatrixXd& A, const VectorXd& b) {
    PartialPivLU<MatrixXd> lu(A);  //PartialPivLU è una classe fornita da Eigen
    return lu.solve(b);
}

//Funzione che risolve il sistema Ax = b usando la decomposizione QR
VectorXd SoluzioneQR(const MatrixXd& A, const VectorXd& b) {
    HouseholderQR<MatrixXd> qr(A);  //HouseholderQR è una classe fornita da Eigen
    return qr.solve(b);
}

//Funzione che calcola l'errore relativo ||Ax - b|| / ||b||
double ErrRel(const MatrixXd& A, const VectorXd& x, const VectorXd& b) {
    VectorXd residuo = A * x - b;
    return residuo.norm() / b.norm();
}

int main() {
    Vector2d x_atteso(-1.0, -1.0);
    cout << "Valore atteso per i tre sistemi: " << x_atteso.transpose() << "\n";
    
/*
    //creo un vettore in cui riporto le matrici A da utilizzare nel ciclo
    vector<MatrixXd> matriciA = {
        (MatrixXd(2, 2) << 0.554701962252291, -0.03770900990025203,
                           0.8320502943378437, -0.9992878623566787).finished(),

        (MatrixXd(2, 2) << 0.554701962252291, -0.5540673164667656,
                           0.8320502943378437, -0.8324762492991315).finished(),

        (MatrixXd(2, 2) << 0.554701962252291, -0.5547019558519056,
                           0.8320502943378437, -0.8320502947645361).finished()
    };
    
    //creo un vettore in cui riporto i vettori b da utilizzare nel ciclo
    vector<VectorXd> vettorib = {
        (Vector2d() << -0.5169911863249772, 0.1672384680188350).finished(),
        (Vector2d() << -0.0006394645785530173, 0.0004259549612877223).finished(),
        (Vector2d() << -6.400391328043042e-10, 4.266924591433961e-10).finished()
    };
*/

    vector<MatrixXd> matriciA;
    vector<VectorXd> vettorib;
    
    MatrixXd A1(2, 2);
    A1 << 0.554701962252291, -0.03770900990025203, 
            0.8320502943378437, -0.9992878623566787;
    matriciA.push_back(A1);
    
    MatrixXd A2(2, 2);
    A2 << 0.554701962252291, -0.5540673164667656,
            0.8320502943378437, -0.8324762492991315;
     matriciA.push_back(A2);
    
    MatrixXd A3(2, 2);
    A3 << 0.554701962252291, -0.5547019558519056,
            0.8320502943378437, -0.8320502947645361;
    matriciA.push_back(A3);
    
    Vector2d b1;
    b1 << -0.5169911863249772, 0.1672384680188350;
    vettorib.push_back(b1);
    
    Vector2d b2;
    b2 << -0.0006394645785530173, 0.0004259549612877223;
    vettorib.push_back(b2);
    
    Vector2d b3;
    b3 << -6.400391328043042e-10, 4.266924591433961e-10;
    vettorib.push_back(b3);

    for (size_t i = 0; i < matriciA.size(); ++i) {
        cout << "Sistema " << i + 1 << ":\n";

        VectorXd x_palu = SoluzionePALU(matriciA[i], vettorib[i]);
        VectorXd x_qr = SoluzioneQR(matriciA[i], vettorib[i]);

        cout << "  Soluzione sistema con PALU: " << x_palu.transpose() << "\n";
        cout << "  Errore relativo con PALU: " << ErrRel(matriciA[i], x_palu, vettorib[i]) << "\n";

        cout << "  Soluzione sistema con QR: " << x_qr.transpose() << "\n";
        cout << "  Errore relativo con QR: " << ErrRel(matriciA[i], x_qr, vettorib[i]) << "\n";

    }

    return 0;
}

