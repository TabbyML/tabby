#import <CoreML/CoreML.h>
#import <Vision/Vision.h>

int main() {
  NSURL *modelUrl = [NSURL URLWithString: @"./Model.mlpackage"];
  MLModel *model = [MLModel modelWithContentsOfURL:modelUrl error:nil];
  return 0;
}
