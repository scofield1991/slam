#include "g2o_types.h"


Vector3d EdgeStereoSE3ProjectXYZOnlyPose::cam_project(const Vector3d & trans_xyz) const{
  const float invz = 1.0f/trans_xyz[2];
  Vector3d res;
  res[0] = trans_xyz[0]*invz*fx + cx;
  res[1] = trans_xyz[1]*invz*fy + cy;
  res[2] = res[0] - bf*invz;
  return res;
}

void EdgeStereoSE3ProjectXYZOnlyPose::linearizeOplus() {
  VertexSE3Expmap * vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
  Vector3d xyz_trans = vi->estimate().map(Xw);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double invz = 1.0/xyz_trans[2];
  double invz_2 = invz*invz;

  _jacobianOplusXi(0,0) =  x*y*invz_2 *fx;
  _jacobianOplusXi(0,1) = -(1+(x*x*invz_2)) *fx;
  _jacobianOplusXi(0,2) = y*invz *fx;
  _jacobianOplusXi(0,3) = -invz *fx;
  _jacobianOplusXi(0,4) = 0;
  _jacobianOplusXi(0,5) = x*invz_2 *fx;

  _jacobianOplusXi(1,0) = (1+y*y*invz_2) *fy;
  _jacobianOplusXi(1,1) = -x*y*invz_2 *fy;
  _jacobianOplusXi(1,2) = -x*invz *fy;
  _jacobianOplusXi(1,3) = 0;
  _jacobianOplusXi(1,4) = -invz *fy;
  _jacobianOplusXi(1,5) = y*invz_2 *fy;

  _jacobianOplusXi(2,0) = _jacobianOplusXi(0,0)-bf*y*invz_2;
  _jacobianOplusXi(2,1) = _jacobianOplusXi(0,1)+bf*x*invz_2;
  _jacobianOplusXi(2,2) = _jacobianOplusXi(0,2);
  _jacobianOplusXi(2,3) = _jacobianOplusXi(0,3);
  _jacobianOplusXi(2,4) = 0;
  _jacobianOplusXi(2,5) = _jacobianOplusXi(0,5)-bf*invz_2;
}

bool EdgeStereoSE3ProjectXYZOnlyPose::read(std::istream& is){
  for (int i=0; i<=3; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<=2; i++)
    for (int j=i; j<=2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeStereoSE3ProjectXYZOnlyPose::write(std::ostream& os) const {

  for (int i=0; i<=3; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<=2; i++)
    for (int j=i; j<=2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

Vector2d project2d(const Vector3d& v)  {
  Vector2d res;
  res(0) = v(0)/v(2);
  res(1) = v(1)/v(2);
  return res;
}

Vector2d EdgeSE3ProjectXYZOnlyPose::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}

void EdgeSE3ProjectXYZOnlyPose::linearizeOplus() {
  VertexSE3Expmap * vi = static_cast<VertexSE3Expmap *>(_vertices[0]);
  Vector3d xyz_trans = vi->estimate().map(Xw);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double invz = 1.0/xyz_trans[2];
  double invz_2 = invz*invz;

  _jacobianOplusXi(0,0) =  x*y*invz_2 *fx;
  _jacobianOplusXi(0,1) = -(1+(x*x*invz_2)) *fx;
  _jacobianOplusXi(0,2) = y*invz *fx;
  _jacobianOplusXi(0,3) = -invz *fx;
  _jacobianOplusXi(0,4) = 0;
  _jacobianOplusXi(0,5) = x*invz_2 *fx;

  _jacobianOplusXi(1,0) = (1+y*y*invz_2) *fy;
  _jacobianOplusXi(1,1) = -x*y*invz_2 *fy;
  _jacobianOplusXi(1,2) = -x*invz *fy;
  _jacobianOplusXi(1,3) = 0;
  _jacobianOplusXi(1,4) = -invz *fy;
  _jacobianOplusXi(1,5) = y*invz_2 *fy;
}

bool EdgeSE3ProjectXYZOnlyPose::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectXYZOnlyPose::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}

void EdgeSE3ProjectXYZ::linearizeOplus() {
  VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  SE3Quat T(vj->estimate());
  VertexSBAPointXYZ* vi = static_cast<VertexSBAPointXYZ*>(_vertices[0]);
  Vector3d xyz = vi->estimate();
  Vector3d xyz_trans = T.map(xyz);

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double z_2 = z*z;

  Matrix<double,2,3> tmp;
  tmp(0,0) = fx;
  tmp(0,1) = 0;
  tmp(0,2) = -x/z*fx;

  tmp(1,0) = 0;
  tmp(1,1) = fy;
  tmp(1,2) = -y/z*fy;

  _jacobianOplusXi =  -1./z * tmp * T.rotation().toRotationMatrix();

  _jacobianOplusXj(0,0) =  x*y/z_2 *fx;
  _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *fx;
  _jacobianOplusXj(0,2) = y/z *fx;
  _jacobianOplusXj(0,3) = -1./z *fx;
  _jacobianOplusXj(0,4) = 0;
  _jacobianOplusXj(0,5) = x/z_2 *fx;

  _jacobianOplusXj(1,0) = (1+y*y/z_2) *fy;
  _jacobianOplusXj(1,1) = -x*y/z_2 *fy;
  _jacobianOplusXj(1,2) = -x/z *fy;
  _jacobianOplusXj(1,3) = 0;
  _jacobianOplusXj(1,4) = -1./z *fy;
  _jacobianOplusXj(1,5) = y/z_2 *fy;
}

Vector2d EdgeSE3ProjectXYZ::cam_project(const Vector3d & trans_xyz) const{
  Vector2d proj = project2d(trans_xyz);
  Vector2d res;
  res[0] = proj[0]*fx + cx;
  res[1] = proj[1]*fy + cy;
  return res;
}

bool EdgeSE3ProjectXYZ::read(std::istream& is){
  for (int i=0; i<2; i++){
    is >> _measurement[i];
  }
  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++) {
      is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
  return true;
}

bool EdgeSE3ProjectXYZ::write(std::ostream& os) const {

  for (int i=0; i<2; i++){
    os << measurement()[i] << " ";
  }

  for (int i=0; i<2; i++)
    for (int j=i; j<2; j++){
      os << " " <<  information()(i,j);
    }
  return os.good();
}