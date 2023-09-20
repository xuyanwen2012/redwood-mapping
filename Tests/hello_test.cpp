#include <gtest/gtest.h>

class Foo {
 public:
  int Bar(const std::string& input_filepath,
          const std::string& output_filepath) {
    return 0;
  }
};

namespace {

// The fixture for testing class Foo.
class FooTest : public ::testing::Test {
 protected:
  FooTest() {}
};

TEST_F(FooTest, MethodBarDoesAbc) {
  const std::string input_filepath = "this/package/testdata/myinputfile.dat";
  const std::string output_filepath = "this/package/testdata/myoutputfile.dat";
  Foo f;
  EXPECT_EQ(f.Bar(input_filepath, output_filepath), 0);
}

}  // namespace
