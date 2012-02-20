// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -fexceptions -std=c++11 | FileCheck %s

// CHECK: @var = internal global
auto var = [](int i) { return i+1; };

int a() { return []{ return 1; }(); }
// CHECK: define i32 @_Z1av
// CHECK: call i32 @_ZZ1avENKUlvE_clEv
// CHECK: define internal i32 @_ZZ1avENKUlvE_clEv
// CHECK: ret i32 1

int b(int x) { return [x]{return x;}(); }
// CHECK: define i32 @_Z1bi
// CHECK: store i32
// CHECK: load i32*
// CHECK: store i32
// CHECK: call i32 @_ZZ1biENKUlvE_clEv
// CHECK: define internal i32 @_ZZ1biENKUlvE_clEv
// CHECK: load i32*
// CHECK: ret i32

int c(int x) { return [&x]{return x;}(); }
// CHECK: define i32 @_Z1ci
// CHECK: store i32
// CHECK: store i32*
// CHECK: call i32 @_ZZ1ciENKUlvE_clEv
// CHECK: define internal i32 @_ZZ1ciENKUlvE_clEv
// CHECK: load i32**
// CHECK: load i32*
// CHECK: ret i32

struct D { D(); D(const D&); int x; };
int d(int x) { D y[10]; [x,y] { return y[x].x; }(); }

// CHECK: define i32 @_Z1di
// CHECK: call void @_ZN1DC1Ev
// CHECK: icmp ult i64 %{{.*}}, 10
// CHECK: call void @_ZN1DC1ERKS_
// CHECK: call i32 @_ZZ1diENKUlvE_clEv
// CHECK: define internal i32 @_ZZ1diENKUlvE_clEv
// CHECK: load i32*
// CHECK: load i32*
// CHECK: ret i32

struct E { E(); E(const E&); ~E(); int x; };
int e(E a, E b, bool cond) { [a,b,cond](){ return (cond ? a : b).x; }(); }
// CHECK: define i32 @_Z1e1ES_b
// CHECK: call void @_ZN1EC1ERKS_
// CHECK: invoke void @_ZN1EC1ERKS_
// CHECK: invoke i32 @_ZZ1e1ES_bENKUlvE_clEv
// CHECK: call void @_ZZ1e1ES_bENUlvE_D1Ev
// CHECK: call void @_ZZ1e1ES_bENUlvE_D1Ev

// CHECK: define internal i32 @_ZZ1e1ES_bENKUlvE_clEv
// CHECK: trunc i8
// CHECK: load i32*
// CHECK: ret i32

void f() {
  // CHECK: define void @_Z1fv()
  // CHECK: {{call.*@_ZZ1fvENKUliiE_cvPFiiiEEv}}
  // CHECK-NEXT: store i32 (i32, i32)*
  // CHECK-NEXT: ret void
  int (*fp)(int, int) = [](int x, int y){ return x + y; };
}

// CHECK: define internal i32 @_ZZ1fvENUliiE_8__invokeEii
// CHECK: store i32
// CHECK-NEXT: store i32
// CHECK-NEXT: load i32*
// CHECK-NEXT: load i32*
// CHECK-NEXT: call i32 @_ZZ1fvENKUliiE_clEii
// CHECK-NEXT: ret i32

// CHECK: define internal void @_ZZ1e1ES_bENUlvE_D2Ev
