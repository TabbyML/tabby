#include <string>
#include <vector>

#import <CoreML/CoreML.h>
#import <Vision/Vision.h>

int compileModel() {
  // NSURL *modelUrl = [NSURL fileURLWithPath: @"/Users/meng/Projects/playground/coreml/j-350m/Model.mlpackage"];
  NSURL *modelUrl = [NSURL fileURLWithPath: @"/Users/meng/Projects/playground/exporters/tiny-gptj/Model.mlpackage"];
  NSURL *compiledUrl = [MLModel compileModelAtURL: modelUrl error:nil];
  NSLog(@"absoluteURL = %@", [compiledUrl absoluteURL]);
  return 0;
}

void SetInput(NSMutableDictionary* dict, const std::string& key, std::vector<uint32_t> ids) {
  // Build shape.
  NSMutableArray* shape = [[NSMutableArray alloc] init];
  [shape addObject: [NSNumber numberWithInteger: 1]];
  [shape addObject: [NSNumber numberWithInteger: ids.size()]];

  MLMultiArrayDataType dataType = MLMultiArrayDataTypeInt32;
  MLMultiArray* dest = [[MLMultiArray alloc] initWithShape:shape dataType:dataType error:nil];
  memcpy(dest.dataPointer, ids.data(), sizeof(uint32_t) * ids.size());

  NSString* nsKey = [NSString stringWithUTF8String:key.c_str()];
  [dict setObject: dest forKey: nsKey];
}

MLModel* InitModel() {
  NSURL *modelUrl = [NSURL URLWithString: @"file:///Users/meng/Projects/tabby/crates/tabby-coreml/cc/j-350.mlmodelc"];
  // NSURL *modelUrl = [NSURL URLWithString: @"file:///Users/meng/Projects/tabby/crates/tabby-coreml/cc/tiny-gptj"];
  return [MLModel modelWithContentsOfURL:modelUrl error:nil];
}

int main() {
#if 0
  compileModel();
  return 0;
#endif

  NSMutableDictionary* input_dict = [NSMutableDictionary dictionary];

  std::vector<uint32_t> input_ids = { 4299, 12900,     7,    77,  2599,   198, 50284,   361,   299,  1279, 352,    25,   198};
  std::vector<uint32_t> attention_mask = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  SetInput(input_dict, "input_ids", input_ids);
  SetInput(input_dict, "attention_mask", attention_mask);
  NSLog(@"inputs %@", input_dict);

  id<MLFeatureProvider> input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:input_dict error:nil];

  auto* model = InitModel();
  id<MLFeatureProvider> output = [model predictionFromFeatures:input error:nil];
  MLMultiArray* token_scores = [output featureValueForName: @"token_scores"].multiArrayValue;
  NSLog(@"Shape: %@, Strides: %@", token_scores, token_scores.strides);

  // 0, shape[1] - 1, 0
  const int offset = (token_scores.shape[1].intValue - 1) * token_scores.strides[1].intValue;
  const int size = token_scores.shape[2].intValue;
  printf("offset: %d size: %d\n", offset, size);

  // start
  float* start = &((float*)token_scores.dataPointer)[offset];

  uint32_t argmax = std::distance(start, std::max_element(start, start + size));

  printf("%d\n", argmax);
  return 0;
}
