#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count) {
    // составление однородной системы + SVD
    // без подвохов
    Eigen::MatrixXd A(2 * count, 3);
    Eigen::VectorXd b(2 * count);

    for (int i_pair = 0; i_pair < count; i_pair++) {
        double x = ms[i_pair][0];
        double y = ms[i_pair][1];
        double z = ms[i_pair][2];

        for (int i = 0; i < 3; i++) {
            A(i_pair * 2, i) = x * Ps[i_pair](2, i) - z * Ps[i_pair](0, i);
            A(i_pair * 2 + 1, i) = y * Ps[i_pair](2, i) - z * Ps[i_pair](1, i);
        }

        b(i_pair * 2) = z * Ps[i_pair](0, 3) - x * Ps[i_pair](2, 3);
        b(i_pair * 2 + 1) = z * Ps[i_pair](1, 3) - y * Ps[i_pair](2, 3);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd s_inv = Eigen::MatrixXd::Zero(3, 2 * count);
    for (int i = 0; i < 3; i++) {
        s_inv(i, i) = 1 / svd.singularValues()[i];
    }
    Eigen::VectorXd ans = svd.matrixV() * s_inv * svd.matrixU().transpose() * b;
    return {ans[0], ans[1], ans[2], 1};
}
