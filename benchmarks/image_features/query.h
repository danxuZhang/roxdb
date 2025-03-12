#pragma once
#include "io.h"
#include "roxdb/db.h"

constexpr const size_t kIters = 10;

inline auto GetQueries(const Dataset& dataset) -> std::vector<rox::Query> {
  const size_t k = 100;

  // Q1: Single KNN on SIFT vectors
  rox::Query q1;
  q1.AddVector("sift", dataset.sift[0]);
  q1.WithLimit(k);

  // Q2: Single KNN on GIST vectors
  rox::Query q2;
  q2.AddVector("gist", dataset.gist[0]);
  q2.WithLimit(k);

  // Q3: Single KNN on SIFT vectors with filters
  rox::Query q3;
  q3.AddVector("sift", dataset.sift[0]);
  q3.AddScalarFilter("category", rox::ScalarFilter::Op::kEq, 5);
  q3.AddScalarFilter("confidence", rox::ScalarFilter::Op::kLt, 0.5);
  q3.WithLimit(k);

  // Q4: Single KNN on GIST vectors with filters
  rox::Query q4;
  q4.AddVector("gist", dataset.gist[0]);
  q4.AddScalarFilter("category", rox::ScalarFilter::Op::kEq, 5);
  q4.AddScalarFilter("confidence", rox::ScalarFilter::Op::kLt, 0.5);
  q4.WithLimit(k);

  // Q5: Multi KNN on SIFT and GIST vectors
  rox::Query q5;
  q5.AddVector("sift", dataset.sift[0]);
  q5.AddVector("gist", dataset.gist[0]);
  q5.WithLimit(k);

  // Q6: Multi KNN on SIFT and GIST vectors with filters
  rox::Query q6;
  q6.AddVector("sift", dataset.sift[0]);
  q6.AddVector("gist", dataset.gist[0]);
  q6.AddScalarFilter("category", rox::ScalarFilter::Op::kEq, 5);
  q6.AddScalarFilter("confidence", rox::ScalarFilter::Op::kLt, 0.5);
  q6.WithLimit(k);

  return {q1, q2, q3, q4, q5, q6};
}
