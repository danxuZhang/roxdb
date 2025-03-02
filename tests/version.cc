#include <gtest/gtest.h>

#include "roxdb/db.h"

// Dummy Test
TEST(DummyTest, Dummy) { EXPECT_EQ(1, 1); }

TEST(DummyTest, Version) { EXPECT_EQ(rox::DB::GetVersion(), "0.1.0"); }
