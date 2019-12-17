// RUN: %cxxamp -emit-llvm -S -c %s -o -|%cppfilt|%FileCheck %s
// RUN: %gtest_amp %s -DUSING_GTEST=1 -o %t && %t
// XFAIL: *
#include <cstdlib> //for size_t
//Serialization object decl
namespace hc {
class Serialize {
 public:
  Serialize():x(0) {}
  void Append(size_t sz, const void *s) {
    x++;
  }
  int x;
};
template<typename T>
class gmac_array {
 public:
  __attribute__((annotate("serialize")))/* For compiler */
   void __cxxamp_serialize(Serialize& s) const {
     s.Append(0, NULL);
   }
   T t;
};
}
class nontemplate {
  public:
  __attribute__((annotate("serialize")))/* For compiler */
    void __cxxamp_serialize(hc::Serialize& s) const {
      s.Append(0, NULL);
    }
};
class baz {
 public:
  __attribute__((annotate("serialize")))/* For compiler */
  void __cxxamp_serialize(hc::Serialize& s) const;
 private:
  hc::gmac_array<float> foo;
  hc::gmac_array<float> bar;
  nontemplate nt;
};

int kerker(void) [[cpu, hc]] {
  baz b1;
  hc::Serialize s;
  b1.__cxxamp_serialize(s);
  return 1;
}
#ifdef USING_GTEST
// The definition should be generated by clang
// CHECK: call {{.*}}void @hc::gmac_array<float>::__cxxamp_serialize
// Executable tests
#include <gtest/gtest.h>
TEST(Serialization, Call) {
  baz bl;
  hc::Serialize s;
  bl.__cxxamp_serialize(s);
  EXPECT_EQ(3, s.x);
}
#endif